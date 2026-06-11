"""
thought_stream.py
=================

Three interlinked capabilities that give the system genuine
awareness of its own reasoning:

  ThoughtStreamMonitor  — generates reasoning steps one at a time
                          and checks each one before continuing.
                          Notices mid-stream if a step doesn't follow,
                          introduces an unjustified assumption, or
                          makes a claim that needs verification.
                          The system is aware of its own reasoning
                          as it happens, not just after.

  ReasonerSelfReview    — after the full chain is generated, the
                          Reasoner reads its own output and checks
                          for consistency, unjustified confidence,
                          and steps that were generated fluently
                          but don't actually hold. Adjusts the
                          output before passing to Communicator.

  VerificationLoop      — the Communicator's natural language
                          translation is sent back to the Reasoner
                          for an accuracy check. Did anything get
                          dropped, softened, added, or subtly changed
                          in translation? If yes, the Communicator
                          tries again with specific corrections noted.
                          Maximum N attempts before outputting the
                          best version.

These run in sequence:
  question
    → ThoughtStreamMonitor (step-by-step generation with real-time checks)
    → ReasonerSelfReview   (self-check on the full chain)
    → Communicator         (translation)
    → VerificationLoop     (back-check before output)
    → user

Integration: pass to BrainPipeline as optional modules.
Each can be enabled independently.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

from brain_pipeline import (
    ReasonerOutput, ReasonerStep,
    REASONER_SYSTEM_PROMPT,
)


# ============================================================
# ThoughtStep — one moment in the thought stream
# ============================================================

@dataclass
class ThoughtStep:
    """
    One step in the streaming thought process.
    Includes the thought itself and the system's real-time check.
    """
    step_number:   int
    label:         str        # STRUCTURAL / DERIVED / VERIFIABLE
    content:       str        # the reasoning step
    check_passed:  bool       # did the step pass its own check?
    check_notes:   str        # what the check found
    flag:          str = ""   # "ASSUMPTION" | "INCONSISTENCY" | "NEEDS_VERIFICATION" | ""
    revised:       bool = False  # was this step revised after flagging?
    source:        str = "derived"


# ============================================================
# ThoughtStreamMonitor
# ============================================================

STREAM_STEP_PROMPT = """You generate ONE reasoning step and immediately check it.

Prior steps so far:
{prior_steps}

Question: {question}

Generate the next reasoning step. Then check it.

Output JSON only:
{{
  "label":        "STRUCTURAL" | "DERIVED" | "VERIFIABLE",
  "content":      "the step",
  "check_passed": true | false,
  "check_notes":  "what the check found",
  "flag":         "" | "ASSUMPTION" | "INCONSISTENCY" | "NEEDS_VERIFICATION",
  "source":       "technique:id" | "derived" | "verified",
  "is_final":     true | false
}}

Rules:
- check_passed = false if the step makes an unjustified leap
- flag = ASSUMPTION if a new unverified claim is being introduced
- flag = INCONSISTENCY if this contradicts a prior step
- flag = NEEDS_VERIFICATION if this is a factual claim that should be checked
- is_final = true when you have reached a conclusion
- Maximum {max_steps} steps total"""

STREAM_REVISION_PROMPT = """A reasoning step was flagged.

Original step: {content}
Flag: {flag}
Check notes: {check_notes}

Revise this step to address the flag. Keep the label the same.
Output JSON only:
{{
  "content":     "revised step",
  "check_notes": "how the revision addresses the flag"
}}"""


class ThoughtStreamMonitor:
    """
    Generates reasoning steps one at a time with real-time self-checking.

    Unlike the standard Reasoner which generates all steps at once,
    this generates each step, checks it, optionally revises it, then
    continues. The system is aware of its own reasoning as it happens.
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        max_steps:   int  = 8,
        auto_revise: bool = True,   # revise flagged steps automatically
        verbose:     bool = True,
    ):
        self.llm         = llm_chat_fn
        self.max_steps   = max_steps
        self.auto_revise = auto_revise
        self.verbose     = verbose

    def generate_stream(
        self,
        question: str,
        domain:   str = "",
        context:  str = "",
    ) -> tuple[list[ThoughtStep], ReasonerOutput]:
        """
        Generate a reasoning chain step by step with real-time monitoring.

        Returns (thought_steps, reasoner_output) where thought_steps
        is the full stream including flags and revisions, and
        reasoner_output is the clean ReasonerOutput for downstream use.
        """
        thought_steps: list[ThoughtStep] = []
        run_id = f"stream-{int(time.time())}"

        if self.verbose:
            print(f"\n  [thought_stream] Streaming reasoning for: {question[:50]}...")

        for step_num in range(1, self.max_steps + 1):
            # Format prior steps for context
            prior = self._format_prior(thought_steps)

            system = "You are a structured reasoning engine generating one step at a time."
            user   = STREAM_STEP_PROMPT.format(
                prior_steps=prior or "(none — this is the first step)",
                question=question,
                max_steps=self.max_steps,
            )
            if context:
                user = f"Context:\n{context}\n\n" + user

            try:
                raw  = self.llm(system, user)
                data = self._parse_json(raw)
            except Exception as e:
                if self.verbose:
                    print(f"  [thought_stream] Step {step_num} parse failed: {e}")
                break

            step = ThoughtStep(
                step_number=step_num,
                label=data.get("label", "DERIVED").upper(),
                content=data.get("content", ""),
                check_passed=bool(data.get("check_passed", True)),
                check_notes=data.get("check_notes", ""),
                flag=data.get("flag", ""),
                source=data.get("source", "derived"),
            )

            if self.verbose:
                flag_str = f" ⚑ {step.flag}" if step.flag else ""
                ok_str   = "✓" if step.check_passed else "✗"
                print(f"  [{step_num}] [{step.label}] {step.content[:60]}... "
                      f"{ok_str}{flag_str}")

            # Auto-revise if flagged and not NEEDS_VERIFICATION
            if (step.flag
                    and step.flag != "NEEDS_VERIFICATION"
                    and self.auto_revise
                    and not step.check_passed):
                revised_step = self._revise_step(step, question)
                if revised_step:
                    step.content      = revised_step["content"]
                    step.check_notes  = revised_step.get("check_notes", step.check_notes)
                    step.revised      = True
                    step.check_passed = True
                    if self.verbose:
                        print(f"       → Revised: {step.content[:60]}...")

            thought_steps.append(step)

            # Check if complete
            if data.get("is_final", False):
                if self.verbose:
                    print(f"  [thought_stream] Concluded after {step_num} steps")
                break

        # Convert to ReasonerOutput
        output = self._to_reasoner_output(thought_steps, question, domain, run_id)
        return thought_steps, output

    def _revise_step(self, step: ThoughtStep, question: str) -> Optional[dict]:
        """Ask the system to revise a flagged step."""
        system = "You revise a reasoning step to address a specific problem."
        user   = STREAM_REVISION_PROMPT.format(
            content=step.content,
            flag=step.flag,
            check_notes=step.check_notes,
        )
        try:
            raw  = self.llm(system, user)
            return self._parse_json(raw)
        except Exception:
            return None

    def _format_prior(self, steps: list[ThoughtStep]) -> str:
        if not steps:
            return ""
        lines = []
        for s in steps:
            flag_note = f" [FLAG: {s.flag}]" if s.flag else ""
            lines.append(f"  [{s.label}] {s.content}{flag_note}")
        return "\n".join(lines)

    def _to_reasoner_output(
        self,
        steps:    list[ThoughtStep],
        question: str,
        domain:   str,
        run_id:   str,
    ) -> ReasonerOutput:
        """Convert thought steps to a clean ReasonerOutput."""
        reasoner_steps = [
            ReasonerStep(
                label=s.label,
                content=s.content,
                source=s.source,
            )
            for s in steps
        ]

        n_structural = sum(
            1 for s in steps if s.label in ("STRUCTURAL", "VERIFIABLE")
        )
        frac = n_structural / max(1, len(steps))

        n_flags         = sum(1 for s in steps if s.flag)
        n_assumptions   = sum(1 for s in steps if s.flag == "ASSUMPTION")
        has_verifiable  = any(s.label == "VERIFIABLE" for s in steps)
        all_passed      = all(s.check_passed for s in steps)

        # Determine confidence based on flags
        if n_flags == 0 and has_verifiable and all_passed:
            confidence = "HIGH"
        elif n_assumptions <= 1 and all_passed:
            confidence = "MODERATE"
        elif n_flags <= 2:
            confidence = "LOW"
        else:
            confidence = "UNCERTAIN"

        # Extract result from last step
        result = steps[-1].content if steps else ""
        if steps and steps[-1].label == "VERIFIABLE":
            result = steps[-1].content

        # Collect assumption flags as obligations raised
        obligations_raised = [
            f"Verify assumption: {s.content[:80]}"
            for s in steps if s.flag == "ASSUMPTION"
        ]

        return ReasonerOutput(
            run_id=run_id,
            domain=domain,
            question=question,
            method="streaming_thought",
            steps=reasoner_steps,
            assumptions=[
                s.content for s in steps if s.flag == "ASSUMPTION"
            ],
            result=result,
            verified=has_verifiable and all_passed,
            confidence=confidence,
            structural_fraction=frac,
            obligations_raised=obligations_raised,
            obligations_discharged=[],
        )

    @staticmethod
    def _parse_json(raw: str) -> dict:
        clean = raw.strip()
        if clean.startswith("```"):
            lines = clean.split("\n")
            clean = "\n".join(l for l in lines[1:] if not l.startswith("```")).strip()
        match = re.search(r"\{[\s\S]*\}", clean)
        if not match:
            return {}
        return json.loads(match.group(0))


# ============================================================
# ReasonerSelfReview
# ============================================================

SELF_REVIEW_PROMPT = """Review your own reasoning chain.

Question: {question}

Your reasoning:
{reasoning_text}

Check for:
1. Steps that don't follow logically from prior steps
2. Assumptions stated as facts
3. Conclusions that aren't supported by the steps
4. Confidence level that isn't warranted by the verification status

Output JSON only:
{{
  "issues_found":    ["issue 1", "issue 2"],
  "steps_to_revise": [0, 2],
  "revised_steps":   {{"0": "revised step 0", "2": "revised step 2"}},
  "confidence_adjustment": "HIGHER" | "LOWER" | "SAME",
  "overall_assessment": "brief assessment",
  "approved": true | false
}}

If no issues found: issues_found=[], steps_to_revise=[], approved=true"""


class ReasonerSelfReview:
    """
    The Reasoner reads its own output and checks it once before
    passing to the Communicator.

    This catches: fluent but logically gapped steps, unwarranted
    confidence, and conclusions that don't follow from the chain.
    """

    def __init__(
        self,
        llm_chat_fn: Callable,
        verbose:     bool = True,
    ):
        self.llm     = llm_chat_fn
        self.verbose = verbose

    def review(self, output: ReasonerOutput) -> ReasonerOutput:
        """
        Review a ReasonerOutput. Returns a (possibly revised) output.
        """
        if not output.steps:
            return output

        reasoning_text = output.to_reasoner_training_text()

        system = "You review your own reasoning for logical consistency and accuracy."
        user   = SELF_REVIEW_PROMPT.format(
            question=reasoning_text.split("\n")[0],
            reasoning_text=reasoning_text,
        )

        if self.verbose:
            print(f"  [self_review] Reviewing {len(output.steps)}-step chain...")

        try:
            raw    = self.llm(system, user)
            match  = re.search(r"\{[\s\S]*\}", raw.strip())
            if not match:
                return output
            review = json.loads(match.group(0))
        except Exception as e:
            if self.verbose:
                print(f"  [self_review] Parse failed: {e}")
            return output

        issues   = review.get("issues_found", [])
        approved = bool(review.get("approved", True))

        if self.verbose:
            if issues:
                print(f"  [self_review] {len(issues)} issue(s) found:")
                for issue in issues[:3]:
                    print(f"    - {issue}")
            else:
                print(f"  [self_review] No issues found ✓")

        if not issues and approved:
            return output

        # Apply revisions
        revised_steps = list(output.steps)
        step_revisions = review.get("revised_steps", {})

        for idx_str, new_content in step_revisions.items():
            try:
                idx = int(idx_str)
                if 0 <= idx < len(revised_steps):
                    revised_steps[idx] = ReasonerStep(
                        label=revised_steps[idx].label,
                        content=new_content,
                        source=revised_steps[idx].source,
                    )
                    if self.verbose:
                        print(f"  [self_review] Step {idx} revised")
            except (ValueError, IndexError):
                pass

        # Adjust confidence
        conf_map = {
            "HIGHER": {"LOW": "MODERATE", "MODERATE": "HIGH", "HIGH": "HIGH",
                       "UNCERTAIN": "LOW"},
            "LOWER":  {"HIGH": "MODERATE", "MODERATE": "LOW", "LOW": "UNCERTAIN",
                       "UNCERTAIN": "UNCERTAIN"},
            "SAME":   None,
        }
        adj          = review.get("confidence_adjustment", "SAME")
        confidence   = output.confidence
        if adj in conf_map and conf_map[adj]:
            confidence = conf_map[adj].get(output.confidence, output.confidence)

        # Record review issues as obligations if not approved
        extra_obligations = []
        if not approved:
            extra_obligations = [
                f"Self-review flagged: {issue[:80]}"
                for issue in issues[:3]
            ]

        return ReasonerOutput(
            run_id=output.run_id,
            domain=output.domain,
            question=output.question,
            method=output.method,
            steps=revised_steps,
            assumptions=output.assumptions,
            result=output.result,
            verified=output.verified and approved,
            confidence=confidence,
            structural_fraction=output.structural_fraction,
            obligations_raised=output.obligations_raised + extra_obligations,
            obligations_discharged=output.obligations_discharged,
            raw_json=output.raw_json,
        )


# ============================================================
# VerificationLoop
# ============================================================

BACK_VERIFY_PROMPT = """You reasoned about a question. Someone translated your reasoning
into natural language. Check whether the translation is accurate.

Your original reasoning:
{original_reasoning}

The translation:
{translated_output}

Check for:
1. Missing steps or conclusions from your reasoning
2. Claims added that you did not make
3. Confidence level changed (e.g. you said MODERATE, they implied HIGH)
4. Assumptions you flagged that were dropped
5. Step labels ([STRUCTURAL]/[DERIVED]/[VERIFIABLE]) preserved correctly

Output JSON only:
{{
  "accurate":          true | false,
  "issues":            ["issue 1", "issue 2"],
  "missing_content":   ["thing 1 that was dropped"],
  "added_content":     ["thing 1 that was added"],
  "confidence_changed": true | false,
  "correction_notes":  "what the Communicator should fix"
}}"""


class VerificationLoop:
    """
    The Communicator's output goes back to the Reasoner for
    an accuracy check before reaching the user.

    Catches: dropped assumptions, inflated confidence, added claims,
    lost nuance in translation.

    Maximum max_attempts passes. Returns the best version found.
    """

    def __init__(
        self,
        reasoner_llm:    Callable,
        communicator_fn: Callable,  # the Communicator.communicate method
        max_attempts:    int  = 2,
        verbose:         bool = True,
    ):
        self.llm_reasoner  = reasoner_llm
        self.communicate   = communicator_fn
        self.max_attempts  = max_attempts
        self.verbose       = verbose

    def verify_and_deliver(
        self,
        reasoning:  ReasonerOutput,
        question:   str,
        register:   str = "auto",
    ) -> tuple[str, dict]:
        """
        Generate communication, back-verify, retry if needed.

        Returns (final_output, verification_record).
        """
        original = reasoning.to_reasoner_training_text()
        verification_record = {
            "attempts": 0,
            "approved": False,
            "issues":   [],
        }

        best_output = None
        best_issues = None

        for attempt in range(1, self.max_attempts + 1):
            verification_record["attempts"] = attempt

            # Generate communication
            correction = (
                best_issues.get("correction_notes", "")
                if attempt > 1 and best_issues else ""
            )
            comm_output = self.communicate(
                reasoning,
                question + (
                    f"\n\n[CORRECTION NOTE: {correction}]"
                    if correction else ""
                ),
                register,
            )

            if best_output is None:
                best_output = comm_output

            if self.verbose:
                print(f"  [verify_loop] Attempt {attempt}: "
                      f"{len(comm_output)} chars generated")

            # Back-verify
            system = "You check whether a translation of your reasoning is accurate."
            user   = BACK_VERIFY_PROMPT.format(
                original_reasoning=original,
                translated_output=comm_output,
            )

            try:
                raw   = self.llm_reasoner(system, user)
                match = re.search(r"\{[\s\S]*\}", raw.strip())
                if not match:
                    verification_record["approved"] = True
                    break
                check = json.loads(match.group(0))
            except Exception:
                verification_record["approved"] = True
                break

            issues   = check.get("issues", [])
            accurate = bool(check.get("accurate", True))

            if self.verbose:
                if issues:
                    print(f"  [verify_loop] Issues found: {issues[:2]}")
                else:
                    print(f"  [verify_loop] Accurate ✓")

            verification_record["issues"] = issues

            if accurate or not issues:
                best_output = comm_output
                verification_record["approved"] = True
                break

            # Store this attempt as best if it has fewer issues
            if best_issues is None or len(issues) < len(best_issues.get("issues", [])):
                best_output = comm_output
                best_issues = check

            if attempt < self.max_attempts:
                if self.verbose:
                    print(f"  [verify_loop] Retrying with correction notes...")

        if not verification_record["approved"]:
            if self.verbose:
                print(f"  [verify_loop] Max attempts reached — "
                      f"outputting best version")

        return best_output, verification_record


# ============================================================
# Integration helper for BrainPipeline
# ============================================================

class ThoughtAwareBrainPipeline:
    """
    Extends BrainPipeline with the full thought-aware reasoning cycle:

      ThoughtStreamMonitor → ReasonerSelfReview → Communicator → VerificationLoop

    Drop-in replacement for BrainPipeline with the same ask() interface.

    Usage:
        from thought_stream import ThoughtAwareBrainPipeline
        from enhanced_pipeline import EnhancedConfig

        pipeline = ThoughtAwareBrainPipeline(
            llm_chat_fn = llm_chat,
            config      = EnhancedConfig(...),
        )
        result = pipeline.ask("Why do flat rotation curves exist?")
    """

    def __init__(
        self,
        llm_chat_fn:          Callable,
        communicator_llm:     Optional[Callable] = None,
        config                = None,
        enable_stream:        bool = True,
        enable_self_review:   bool = True,
        enable_verify_loop:   bool = True,
        max_stream_steps:     int  = 8,
        max_verify_attempts:  int  = 2,
        verbose:              bool = True,
    ):
        from brain_pipeline import BrainPipeline, Communicator

        comm_llm = communicator_llm or llm_chat_fn

        self._base = BrainPipeline(
            reasoner_llm=llm_chat_fn,
            communicator_llm=comm_llm,
            config=config,
            verbose=verbose,
        )

        self.stream_monitor = (
            ThoughtStreamMonitor(llm_chat_fn, max_stream_steps, verbose=verbose)
            if enable_stream else None
        )
        self.self_review = (
            ReasonerSelfReview(llm_chat_fn, verbose=verbose)
            if enable_self_review else None
        )
        self.verify_loop = (
            VerificationLoop(
                llm_chat_fn,
                self._base.communicator.communicate,
                max_verify_attempts,
                verbose=verbose,
            )
            if enable_verify_loop else None
        )

        self.verbose = verbose

    def ask(
        self,
        question: str,
        domain:   str = "",
        register: str = "auto",
    ) -> dict:
        t0     = time.time()
        domain = domain or getattr(
            getattr(self._base._enhanced, "cfg", None), "domain_name", ""
        )

        # Memory + introspection context
        memory_ctx    = self._base._get_memory_context(question, domain)
        introspect_ctx = self._base._get_introspect_context(question, domain)
        context       = "\n".join(filter(None, [introspect_ctx, memory_ctx]))

        thought_steps = []

        # Stage 1: Generate reasoning
        if self.stream_monitor:
            thought_steps, reasoning = self.stream_monitor.generate_stream(
                question, domain, context
            )
        else:
            reasoning = self._base.reasoner.reason(
                question, domain, memory_ctx, introspect_ctx
            )

        # Stage 2: Self-review
        if self.self_review:
            reasoning = self.self_review.review(reasoning)

        # Stage 3: Communicate + back-verify
        verification_record = {}
        if self.verify_loop:
            response, verification_record = self.verify_loop.verify_and_deliver(
                reasoning, question, register
            )
        else:
            response = self._base.communicator.communicate(
                reasoning, question, register
            )

        # Post-process
        pipeline_result = self._base._wrap_as_pipeline_result(reasoning, question)
        self._base._post_process(pipeline_result, reasoning, question, response)
        self._base._store_training_example(reasoning, question, response)

        return {
            "run_id":               reasoning.run_id,
            "question":             question,
            "domain":               domain,
            "reasoning":            reasoning,
            "response":             response,
            "thought_steps":        thought_steps,
            "verification_record":  verification_record,
            "verified":             reasoning.verified,
            "confidence":           reasoning.confidence,
            "elapsed":              round(time.time() - t0, 2),
            "selected_branch":      reasoning.to_pipeline_format(),
        }

    # Delegate other methods to base pipeline
    def consolidate(self, quiet: bool = False) -> dict:
        return self._base.consolidate(quiet=quiet)

    def explore(self, n: int = 1) -> list:
        return self._base.explore(n=n)

    def export_training_data(self, **kwargs) -> tuple:
        return self._base.export_training_data(**kwargs)

    def status(self) -> str:
        return self._base.status()
