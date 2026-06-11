"""
streaming_pipeline.py
=====================

Streams the Language module's output token by token via WebSocket.

Current flow: user waits ~30-60s for full response → receives it all at once.
Streaming flow: user sees first token in <1s → text appears word by word.

For a paid product this is the single biggest perceived performance improvement.
Groq's API supports streaming natively. We thread it through the WebSocket.

WebSocket protocol (what the frontend receives):
    {"type": "reasoning_start"}          ← reasoning underway
    {"type": "token", "content": "The"}  ← token stream starts
    {"type": "token", "content": " thread"}
    ...
    {"type": "response_end", "verified": true, "confidence": "MODERATE", ...}

Usage:
    streamer = StreamingPipeline(scaffold, groq_client)
    await streamer.process_and_stream(websocket, question, domain, user_tier)
"""

import json
import asyncio
import time
from typing import Optional, AsyncGenerator


class StreamingPipeline:
    """
    Handles the full Bri pipeline with streaming final output.
    
    The Reasoner runs to completion (needed for correctness).
    The Language module output streams token by token.
    Module updates are queued to background (via BackgroundUpdater).
    """

    def __init__(
        self,
        scaffold:         dict,
        groq_client       = None,    # groq.Groq instance
        background_updater = None,   # BackgroundUpdater instance
        verbose:          bool = False,
    ):
        self.scaffold   = scaffold
        self.groq       = groq_client
        self.updater    = background_updater
        self.verbose    = verbose

    async def process_and_stream(
        self,
        websocket,
        question:   str,
        domain:     str,
        user_ip:    str,
        collective,
        user_tier:  str = "free",
    ) -> dict:
        """
        Full pipeline with streaming Language module output.
        Returns metadata dict (verified, confidence, phi, etc.)
        """
        sc = self.scaffold

        # 1. Tell frontend reasoning is starting
        await self._send(websocket, {"type": "reasoning_start"})

        # 2. Assemble integrated state
        try:
            from integrated_state import StateAssembler
            assembler = StateAssembler(
                brain=sc.get("brain"), psych=sc.get("psych"),
                plastic=sc.get("plastic"), dmn=sc.get("dmn"),
                persona=sc.get("persona"), knowledge=sc.get("knowledge"),
            )
            state = assembler.assemble(question, domain)
        except ImportError:
            state = None

        # 3. Check procedural shortcut (instant)
        try:
            auto = sc["brain"].try_automatic(question, domain)
            if auto:
                _, ar = auto
                sc["brain"].post_reasoning_update(ar, question, domain)
                response = ar.result or ""
                # Stream the procedural result
                await self._stream_text(websocket, response)
                await self._send(websocket, {
                    "type": "response_end",
                    "verified": True,
                    "confidence": ar.confidence,
                    "automatic": True,
                    "phi": sc["dmn"].workspace.phi_estimate,
                })
                return {"response": response, "verified": True, "automatic": True}
        except Exception:
            pass

        # 4. Build context (smart — only what's relevant)
        ctx = self._build_smart_context(question, domain, sc, collective, user_ip)

        sc["dmn"].pause_for_task()
        result = {}
        try:
            # 5. Run the reasoning pipeline (non-streaming — correctness first)
            loop = asyncio.get_event_loop()
            full_q = f"{question}\n\n{ctx}" if ctx else question
            result = await loop.run_in_executor(
                None, sc["pipeline"].ask, full_q, domain
            )

            r        = result.get("reasoning")
            response = ""

            if r and state:
                # 6. Update state with result
                if hasattr(assembler, 'update_with_result'):
                    assembler.update_with_result(state, r)

                # 7. Stream the Language module output
                if self.groq and r.result:
                    response = await self._stream_language_output(
                        websocket, state, r, question
                    )
                else:
                    # Fallback: stream the raw result
                    response = r.result or result.get("response", "")
                    await self._stream_text(websocket, response)
            else:
                response = result.get("response", "")
                await self._stream_text(websocket, response)

            # 8. Send metadata
            phi = sc["dmn"].workspace.phi_estimate
            await self._send(websocket, {
                "type":       "response_end",
                "verified":   getattr(r, "verified", False) if r else False,
                "confidence": getattr(r, "confidence", "UNCERTAIN") if r else "UNCERTAIN",
                "phi":        phi,
                "reasoning_trace": self._format_trace(r) if r else "",
            })

            # 9. Queue all module updates to background (non-blocking)
            result["response"] = response
            if self.updater:
                self.updater.queue(result, question, domain, response, user_ip, collective)

            return {"response": response, "verified": getattr(r,"verified",False)}

        except Exception as e:
            await self._send(websocket, {"type": "error", "content": str(e)})
            return {"response": "", "verified": False, "error": str(e)}
        finally:
            sc["dmn"].resume()

    async def _stream_language_output(
        self, websocket, state, reasoning_result, question: str
    ) -> str:
        """
        Call Groq with streaming=True, send each token via WebSocket.
        Returns the full assembled text.
        """
        try:
            from language_module import LanguageModule, IDENTITY_SYSTEM, RESULT_CONTEXT
            from integrated_state import StateAssembler

            # Build the prompts
            result_text = getattr(reasoning_result, "result", "") or ""
            if not result_text and getattr(reasoning_result, "steps", None):
                for step in reversed(reasoning_result.steps):
                    if getattr(step, "label", "") in ("DERIVED",) and step.content:
                        result_text = step.content; break

            integrated_ctx = ""
            ws = self.scaffold["dmn"].workspace
            parts = []
            if ws.cross_domain_insight: parts.append(f"Background connection: {ws.cross_domain_insight[:100]}")
            if ws.spontaneous_question:  parts.append(f"You've been wondering: {ws.spontaneous_question[:80]}")
            if state.questioning_beliefs: parts.append(f"Belief in question: '{state.questioning_beliefs[0][:60]}'")
            integrated_ctx = "\n".join(parts)

            note = " and reached a verified conclusion" if state.verified else " but couldn't fully verify"
            system_prompt = IDENTITY_SYSTEM.format(
                character=state.character_description(),
                question=question,
                reasoning_note=note,
            )
            user_prompt = RESULT_CONTEXT.format(
                result=result_text[:500] or "Reasoning was exploratory.",
                integrated=integrated_ctx[:300] or "None",
            )

            # Stream from Groq
            full_text = ""
            loop = asyncio.get_event_loop()

            def do_groq_stream():
                return self.groq.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    stream=True,
                    max_tokens=1024,
                    temperature=0.7,
                )

            stream = await loop.run_in_executor(None, do_groq_stream)

            def iter_stream():
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta

            # Send tokens via WebSocket
            for token in iter_stream():
                full_text += token
                await self._send(websocket, {"type": "token", "content": token})
                await asyncio.sleep(0)  # yield to event loop

            return full_text

        except Exception as e:
            if self.verbose:
                print(f"[streaming] Groq stream error: {e}")
            # Fallback to non-streaming
            response = getattr(reasoning_result, "result", "") or ""
            await self._stream_text(websocket, response)
            return response

    async def _stream_text(self, websocket, text: str, chunk_size: int = 4) -> None:
        """Stream text word by word as fallback (no Groq streaming)."""
        words = text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            await self._send(websocket, {"type": "token", "content": token})
            if i % 8 == 0:
                await asyncio.sleep(0.01)

    def _build_smart_context(self, question, domain, sc, collective, user_ip) -> str:
        """
        Build context using only what's relevant to this question type.
        
        Simple questions: just DMN context (fast, low token count)
        Technical questions: add brain + knowledge context
        Complex questions: full context
        """
        lower = question.lower()
        word_count = len(lower.split())

        # Classify question complexity
        is_complex = any(w in lower for w in [
            "derive", "prove", "show that", "analyse", "analyze",
            "what are the implications", "from first principles", "why does"
        ])
        is_simple = word_count < 8 and not is_complex
        is_technical = any(w in lower for w in [
            "calculate", "compute", "what is the", "define", "formula"
        ])

        parts = []

        # Always include DMN workspace (cheap, high value)
        try:
            dmn_ctx = sc["dmn"].get_context()
            if dmn_ctx: parts.append(dmn_ctx)
        except Exception: pass

        # Skip heavy context for simple questions
        if is_simple:
            return "\n\n".join(parts)

        # Technical and complex get brain context
        if is_technical or is_complex:
            try:
                brain_ctx = sc["brain"].pre_reasoning_context(question, domain)
                if brain_ctx: parts.append(brain_ctx)
            except Exception: pass

        # Complex questions get everything
        if is_complex:
            try:
                psych_ctx = sc["psych"].pre_reasoning_context(question, domain)
                if psych_ctx: parts.append(psych_ctx)
            except Exception: pass
            try:
                plastic_ctx = sc["plastic"].pre_reasoning_context(question, domain)
                if plastic_ctx: parts.append(plastic_ctx)
            except Exception: pass

        # Knowledge context for all non-simple questions
        if not is_simple:
            try:
                know_ctx = sc["knowledge"].pre_reasoning_context(question, domain)
                if know_ctx: parts.append(know_ctx)
            except Exception: pass

        return "\n\n".join(parts)

    @staticmethod
    async def _send(websocket, data: dict) -> None:
        try:
            await websocket.send_json(data)
        except Exception:
            pass

    @staticmethod
    def _format_trace(r) -> str:
        if not r: return ""
        lines = [f"Method: {getattr(r,'method','?')}",
                 f"Verified: {r.verified} | Confidence: {r.confidence}"]
        for step in getattr(r, "steps", []):
            lines.append(f"  [{step.label}] {step.content[:80]}")
        return "\n".join(lines)
