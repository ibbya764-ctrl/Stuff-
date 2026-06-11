"""
request_manager.py
==================

Handles concurrent users cleanly.

Two components:

  RateLimiter    — per-user and global request rate limiting.
                   Prevents one user consuming all compute.
                   Configurable limits: questions per hour,
                   questions per day, max concurrent requests.

  RequestQueue   — async queue for LLM requests.
                   When the LLM is busy, requests wait in queue
                   rather than failing. Users see their position.
                   Processes one (or N) requests at a time.

Together these make the system robust under multi-user load
without needing to scale to multiple GPU instances immediately.
"""

import asyncio
import time
import json
import os
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import defaultdict


# ============================================================
# RateLimiter
# ============================================================

@dataclass
class RateLimit:
    """Rate limit configuration."""
    per_hour:      int   = 20    # max questions per user per hour
    per_day:       int   = 100   # max questions per user per day
    max_concurrent: int  = 5     # max simultaneous requests across all users
    queue_timeout:  int  = 300   # seconds to wait in queue before giving up


class RateLimiter:
    """
    Per-user sliding window rate limiter.

    Tracks request timestamps per user. When a user exceeds their
    limit, their request is either queued (if within timeout) or
    rejected with a clear message about when they can next ask.
    """

    def __init__(self, limits: Optional[RateLimit] = None):
        self.limits  = limits or RateLimit()
        # user_id → list of timestamps
        self._history: dict[str, list[float]] = defaultdict(list)
        self._concurrent = 0

    def check(self, user_id: str) -> tuple[bool, str]:
        """
        Check if user can make a request.
        Returns (allowed, reason_if_not).
        """
        now    = time.time()
        history = self._history[user_id]

        # Clean old entries
        hour_ago = now - 3600
        day_ago  = now - 86400
        self._history[user_id] = [t for t in history if t > day_ago]
        history  = self._history[user_id]

        # Check per-hour
        last_hour = [t for t in history if t > hour_ago]
        if len(last_hour) >= self.limits.per_hour:
            oldest   = min(last_hour)
            reset_in = int(3600 - (now - oldest))
            return False, (
                f"Hourly limit reached ({self.limits.per_hour}/hour). "
                f"Resets in {reset_in//60}m {reset_in%60}s."
            )

        # Check per-day
        if len(history) >= self.limits.per_day:
            oldest   = min(history)
            reset_in = int(86400 - (now - oldest))
            return False, (
                f"Daily limit reached ({self.limits.per_day}/day). "
                f"Resets in {reset_in//3600}h."
            )

        # Check global concurrent
        if self._concurrent >= self.limits.max_concurrent:
            return False, (
                f"Server busy ({self._concurrent} requests in progress). "
                f"Please wait a moment."
            )

        return True, ""

    def record(self, user_id: str) -> None:
        """Record a request as started."""
        self._history[user_id].append(time.time())
        self._concurrent = min(
            self.limits.max_concurrent,
            self._concurrent + 1
        )

    def release(self, user_id: str) -> None:
        """Record a request as completed."""
        self._concurrent = max(0, self._concurrent - 1)

    def usage_for(self, user_id: str) -> dict:
        """Current usage for a user."""
        now      = time.time()
        history  = self._history[user_id]
        last_hour = [t for t in history if t > now - 3600]
        return {
            "this_hour": len(last_hour),
            "today":     len(history),
            "limit_hour": self.limits.per_hour,
            "limit_day":  self.limits.per_day,
            "concurrent": self._concurrent,
        }

    def admin_stats(self) -> dict:
        """Global usage stats for admin."""
        now = time.time()
        return {
            "concurrent":    self._concurrent,
            "active_users":  sum(
                1 for h in self._history.values()
                if any(t > now - 300 for t in h)
            ),
            "total_users":   len(self._history),
            "questions_today": sum(
                len([t for t in h if t > now - 86400])
                for h in self._history.values()
            ),
        }


# ============================================================
# RequestQueue
# ============================================================

@dataclass
class QueuedRequest:
    request_id:  str
    user_id:     str
    question:    str
    domain:      str
    session_id:  str
    added_at:    float = field(default_factory=time.time)
    result:      Any   = None
    error:       str   = ""
    done:        bool  = False

    async def wait(self, timeout: int = 300) -> Any:
        """Wait for this request to complete."""
        deadline = time.time() + timeout
        while not self.done:
            if time.time() > deadline:
                raise asyncio.TimeoutError(
                    "Request timed out in queue after "
                    f"{timeout}s"
                )
            await asyncio.sleep(0.5)
        if self.error:
            raise RuntimeError(self.error)
        return self.result


class RequestQueue:
    """
    Async queue for LLM requests.

    When the LLM is busy processing a request, new requests
    wait in a FIFO queue. Users are told their position.

    The queue processes requests one at a time (or N at a time
    if parallel processing is configured). This prevents the
    LLM from being called concurrently, which would cause
    quality degradation and memory issues.

    Status updates are sent via WebSocket so users see
    live queue position updates while waiting.
    """

    def __init__(
        self,
        max_concurrent: int = 1,    # process N requests simultaneously
        max_queue:      int = 20,   # max requests waiting in queue
        verbose:        bool = True,
    ):
        self.max_concurrent = max_concurrent
        self.max_queue      = max_queue
        self.verbose        = verbose

        self._queue:   list[QueuedRequest]  = []
        self._active:  list[QueuedRequest]  = []
        self._lock     = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running  = False
        self._stats = {
            "total_processed": 0,
            "total_errors":    0,
            "avg_wait_sec":    0.0,
        }

    async def start(self) -> None:
        """Start the queue processor."""
        self._running = True
        asyncio.create_task(self._process_loop())

    async def stop(self) -> None:
        self._running = False

    async def enqueue(
        self,
        user_id:    str,
        question:   str,
        domain:     str,
        session_id: str,
        request_id: Optional[str] = None,
    ) -> QueuedRequest:
        """
        Add a request to the queue.
        Returns the QueuedRequest — caller can await request.wait().
        """
        import uuid
        rid = request_id or str(uuid.uuid4())[:8]

        async with self._lock:
            if len(self._queue) >= self.max_queue:
                raise RuntimeError(
                    f"Queue full ({self.max_queue} requests waiting). "
                    f"Please try again in a few minutes."
                )

            req = QueuedRequest(
                request_id=rid,
                user_id=user_id,
                question=question,
                domain=domain,
                session_id=session_id,
            )
            self._queue.append(req)

        if self.verbose:
            pos = len(self._queue)
            print(f"[queue] Enqueued {rid} — position {pos}")

        return req

    def queue_position(self, request_id: str) -> int:
        """Return 1-based position in queue, or 0 if not found."""
        for i, req in enumerate(self._queue):
            if req.request_id == request_id:
                return i + 1
        return 0

    def status(self) -> dict:
        return {
            "queue_length":    len(self._queue),
            "active":          len(self._active),
            "max_concurrent":  self.max_concurrent,
            "max_queue":       self.max_queue,
            "total_processed": self._stats["total_processed"],
            "avg_wait_sec":    round(self._stats["avg_wait_sec"], 1),
        }

    async def _process_loop(self) -> None:
        """Main processing loop — runs continuously."""
        while self._running:
            async with self._lock:
                if not self._queue:
                    await asyncio.sleep(0.2)
                    continue
                req = self._queue.pop(0)
                self._active.append(req)

            asyncio.create_task(self._process(req))
            await asyncio.sleep(0.1)

    async def _process(self, req: QueuedRequest) -> None:
        """Process one request using the registered handler."""
        wait_sec = time.time() - req.added_at
        try:
            async with self._semaphore:
                if self._handler:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._handler, req.question, req.domain,
                        req.session_id, req.user_id
                    )
                    req.result = result
                else:
                    req.error = "No handler registered"
        except Exception as e:
            req.error = str(e)
            self._stats["total_errors"] += 1
        finally:
            req.done = True
            if req in self._active:
                self._active.remove(req)
            self._stats["total_processed"] += 1
            # Update rolling average
            n   = self._stats["total_processed"]
            avg = self._stats["avg_wait_sec"]
            self._stats["avg_wait_sec"] = (avg * (n-1) + wait_sec) / n

    _handler: Optional[Callable] = None

    def register_handler(self, fn: Callable) -> None:
        """Register the function that processes each request."""
        self._handler = fn


# ============================================================
# Simple combined middleware for FastAPI
# ============================================================

class RequestManager:
    """
    Combines rate limiter and request queue.
    Single entry point for the web API.
    """

    def __init__(
        self,
        limits:      Optional[RateLimit] = None,
        max_queue:   int  = 20,
        max_concurrent: int = 1,
        verbose:     bool = True,
    ):
        self.limiter = RateLimiter(limits or RateLimit())
        self.queue   = RequestQueue(max_concurrent, max_queue, verbose)
        self.verbose = verbose

    async def start(self) -> None:
        await self.queue.start()

    def register_handler(self, fn: Callable) -> None:
        self.queue.register_handler(fn)

    async def submit(
        self,
        user_id:    str,
        question:   str,
        domain:     str,
        session_id: str,
    ) -> dict:
        """
        Submit a request. Handles rate limiting and queuing.
        Returns result dict or error dict.
        """
        # Rate limit check
        allowed, reason = self.limiter.check(user_id)
        if not allowed:
            return {
                "success": False,
                "error":   reason,
                "rate_limited": True,
                "usage": self.limiter.usage_for(user_id),
            }

        try:
            self.limiter.record(user_id)
            req = await self.queue.enqueue(user_id, question, domain, session_id)

            # Wait for processing
            result = await req.wait(timeout=self.queue._semaphore._value * 300)

            return {"success": True, "result": result}

        except asyncio.TimeoutError as e:
            return {"success": False, "error": str(e), "timeout": True}
        except RuntimeError as e:
            return {"success": False, "error": str(e)}
        finally:
            self.limiter.release(user_id)

    def admin_stats(self) -> dict:
        return {
            "rate_limiter": self.limiter.admin_stats(),
            "queue":        self.queue.status(),
        }
