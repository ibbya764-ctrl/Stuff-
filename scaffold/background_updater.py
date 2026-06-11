"""
background_updater.py
=====================

Decouples module updates from response delivery.

Currently all post-reasoning updates (brain, psych, plastic, calib,
persona, DMN, collective, training) happen synchronously before the
response reaches the user. This adds 200-800ms of unnecessary latency
to every response.

The fix: return the response immediately, queue all updates to run
in a background thread. The user gets their answer faster; the
system still learns from every interaction.

Usage:
    updater = BackgroundUpdater(scaffold)
    
    # After getting the response — deliver it to user immediately,
    # then queue updates
    updater.queue(result, question, domain, response_text, user_ip, collective)
"""

import threading
import queue
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class UpdateJob:
    result:         Any
    question:       str
    domain:         str
    response_text:  str
    user_ip:        str
    collective:     Any
    scaffold:       Any
    timestamp:      float = field(default_factory=time.time)


class BackgroundUpdater:
    """
    Runs all post-response module updates in a background thread.
    
    The response is delivered to the user immediately.
    Updates happen within a few hundred milliseconds, invisibly.
    """

    def __init__(self, scaffold: dict, verbose: bool = False):
        self.scaffold = scaffold
        self.verbose  = verbose
        self._queue   = queue.Queue(maxsize=100)
        self._stats   = {"processed": 0, "errors": 0, "avg_ms": 0.0}
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, daemon=True, name="BriUpdater"
        )
        self._thread.start()

    def queue(
        self,
        result:       Any,
        question:     str,
        domain:       str,
        response_text: str,
        user_ip:      str,
        collective:   Any,
    ) -> None:
        """Queue a set of module updates. Non-blocking."""
        try:
            self._queue.put_nowait(UpdateJob(
                result=result, question=question, domain=domain,
                response_text=response_text, user_ip=user_ip,
                collective=collective, scaffold=self.scaffold,
            ))
        except queue.Full:
            if self.verbose:
                print("[updater] Queue full — dropping update job")

    def stop(self):
        self._running = False

    def stats(self) -> dict:
        return {**self._stats, "queue_size": self._queue.qsize()}

    def _worker(self):
        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
                t0  = time.time()
                self._run_updates(job)
                elapsed = (time.time() - t0) * 1000
                n = self._stats["processed"] + 1
                self._stats["processed"] = n
                self._stats["avg_ms"] = (
                    self._stats["avg_ms"] * (n - 1) + elapsed
                ) / n
                if self.verbose:
                    print(f"[updater] Updates done in {elapsed:.0f}ms")
            except queue.Empty:
                continue
            except Exception as e:
                self._stats["errors"] += 1
                if self.verbose:
                    print(f"[updater] Error: {e}")

    def _run_updates(self, job: UpdateJob):
        sc       = job.scaffold
        r        = job.result.get("reasoning") if isinstance(job.result, dict) else None
        response = job.response_text

        if not r:
            return

        # Brain modules
        try:
            sc["brain"].post_reasoning_update(
                r, job.question, job.domain, success=r.verified
            )
        except Exception: pass

        # Psychological evaluation
        try:
            fb = sc["psych"].evaluate(r, response, job.question)
        except Exception:
            fb = None

        # Plastic core
        try:
            sc["plastic"].process_run(r, response, job.question)
        except Exception: pass

        # Calibrator
        try:
            sc["calib"].record_claim(r.confidence, r.verified, "derived", job.domain)
        except Exception: pass

        # Personality
        try:
            sc["persona"].update_from_response(response)
        except Exception: pass

        # Integration score
        try:
            if fb:
                signals = {
                    "values":    fb.value_score,
                    "aesthetic": fb.aesthetic_score,
                    "social":    fb.social_score,
                    "engagement": fb.engagement,
                }
                sc["integr"].compute_integration(signals)
        except Exception: pass

        # DMN integration
        try:
            sc["dmn"].integrate_task_result(job.result, job.question)
        except Exception: pass

        # Knowledge base — store learned facts
        try:
            if r.result:
                sc["knowledge"].store_learned_fact(
                    r.result[:200], job.domain, "reasoning",
                    confidence=0.65 if r.verified else 0.45
                )
        except Exception: pass

        # Collective contribution
        try:
            if r.verified:
                job.collective.contribute(r, job.question, job.user_ip, {
                    "messages": [
                        {"role": "user",      "content": job.question},
                        {"role": "assistant", "content": response},
                    ]
                })
                job.collective.update_session_activity(job.user_ip, True)
        except Exception: pass

        # Continual trainer
        try:
            if sc.get("trainer") and sc.get("episodic"):
                rec = sc["episodic"].query_recent(n=1)
                if rec:
                    sc["trainer"].on_run_complete(job.result, rec[0])
        except Exception: pass

        # From-scratch trainer
        try:
            if sc.get("from_scratch") and r.verified:
                sc["from_scratch"].add_verified_run(r, job.question)
        except Exception: pass
