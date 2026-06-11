#!/usr/bin/env python3
"""
mac_run.py — Full architecture, all modules wired.

Modules active:
  ThoughtAwareBrainPipeline   Reasoner + Communicator + self-review + back-verify
  BrainModulesBundle          Working memory, affective signal, procedural learning
  FullPsychologicalBundle     Values, aesthetic, social, emotional, narrative, wonder,
                              flow, temporal self, moral reasoning, intrinsic motivation,
                              identity resilience, emotional resonance, self-compassion,
                              intellectual humility, consciousness proxy
  PlasticPsychologicalCore    Belief system, assumption auditing, philosophical inquiry,
                              epistemic plasticity monitoring
  EpistemicHumilityCalibrator Confidence calibration over time
  PersonalityConsistency      Trait tracking across interactions
  PsychologicalIntegration    Module alignment monitoring
  FromScratchTrainer          Small model (Qwen2-0.5B) trains from run 1
  ContinualScaffoldTrainer    Continual LoRA fine-tuning every 5 verified runs

Run: python3 mac_run.py
"""

import sys
import os
import json
import time
import requests


# ── LLM ───────────────────────────────────────────────────

def llm_chat(system: str, user: str) -> str:
    try:
        r = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "qwen2.5:7b",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream":  False,
                "options": {"temperature": 0.7, "num_predict": 2000},
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()["message"]["content"]
    except Exception:
        pass

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":     "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "system":    system,
                "messages":  [{"role": "user", "content": user}],
            },
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()["content"][0]["text"]

    raise RuntimeError(
        "\nNo LLM available.\n"
        "  Start Ollama: ollama serve\n"
        "  Or set:       export ANTHROPIC_API_KEY=your_key"
    )


# ── Setup check ───────────────────────────────────────────

def check() -> bool:
    print("Checking setup...\n")
    ok = True

    modules = [
        ("pipeline",              "original pipeline"),
        ("obligation_store",      "original pipeline"),
        ("technique_library",     "original pipeline"),
        ("enhanced_pipeline",     "scaffold"),
        ("brain_pipeline",        "scaffold"),
        ("thought_stream",        "scaffold"),
        ("brain_modules",         "scaffold"),
        ("continual_trainer",     "scaffold"),
        ("memory_system",         "scaffold"),
        ("psychology",            "scaffold"),
        ("psychology_extended",   "scaffold"),
        ("belief_system",         "scaffold"),
        ("psych_modules",         "scaffold"),
        ("from_scratch_trainer",  "scaffold"),
    ]

    for module, kind in modules:
        try:
            __import__(module)
            print(f"  ok  {module}")
        except ImportError as e:
            print(f"  MISSING  {module}  ({e})")
            if kind == "original pipeline":
                print(f"           → Download from your Colab notebook")
            ok = False

    print()

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        if any("qwen2.5" in m for m in models):
            print("  ok  Ollama + qwen2.5")
        else:
            print("  WARNING  qwen2.5 not found. Run: ollama pull qwen2.5:7b")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                ok = False
    except Exception:
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("  ok  Claude API key set")
        else:
            print("  MISSING  Ollama not running AND no ANTHROPIC_API_KEY")
            ok = False

    print()
    return ok


# ── Main ──────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Full Architecture")
    print("=" * 60 + "\n")

    if not check():
        print("Fix the issues above and run again.")
        sys.exit(1)

    # ── Imports ───────────────────────────────────────────

    from enhanced_pipeline import EnhancedConfig
    from thought_stream import ThoughtAwareBrainPipeline
    from brain_modules import BrainModulesBundle
    from continual_trainer import ContinualScaffoldTrainer
    from from_scratch_trainer import FromScratchTrainer
    from psychology_extended import FullPsychologicalBundle
    from belief_system import PlasticPsychologicalCore
    from psych_modules import (
        EpistemicHumilityCalibrator,
        PersonalityConsistency,
        PsychologicalIntegration,
    )

    BASE = "./scaffold_data"
    os.makedirs(BASE, exist_ok=True)

    print("Initialising modules...\n")

    # ── Brain modules ──────────────────────────────────────
    brain = BrainModulesBundle(base_dir=BASE)
    print("  ✓ Brain modules  (working memory, affective signal, procedural)")

    # ── Pipeline ───────────────────────────────────────────
    # enable_stream=False for MacBook speed (~2-3 min/question)
    # Change to True on 4060 Ti for full thought streaming
    pipeline = ThoughtAwareBrainPipeline(
        llm_chat_fn=llm_chat,
        config=EnhancedConfig(
            base_dir=BASE,
            domain_name="physics_mond",
            verbose=False,
        ),
        enable_stream=False,
        enable_self_review=True,
        enable_verify_loop=True,
        verbose=True,
    )
    print("  ✓ Brain pipeline (Reasoner → self-review → Communicator → back-verify)")

    # ── Psychology ─────────────────────────────────────────
    psych   = FullPsychologicalBundle(base_dir=BASE)
    plastic = PlasticPsychologicalCore(base_dir=BASE)
    calib   = EpistemicHumilityCalibrator(os.path.join(BASE, "calib.json"))
    persona = PersonalityConsistency(os.path.join(BASE, "personality.json"))
    integr  = PsychologicalIntegration(os.path.join(BASE, "integration.json"))
    print("  ✓ Psychology     (values, aesthetic, social, emotional, belief system)")
    print("  ✓ Plasticity     (belief revision, assumption auditing, philosophical inquiry)")

    # ── Training ───────────────────────────────────────────
    episodic = None
    trainer  = None
    from_scratch = None
    try:
        episodic = pipeline._base._enhanced._episodic
    except Exception:
        pass

    if episodic:
        trainer = ContinualScaffoldTrainer(
            episodic_store=episodic,
            base_dir=BASE,
            model_name="Qwen/Qwen2.5-7B-Instruct",
            trigger_every=5,
            verbose=False,
        )
        from_scratch = FromScratchTrainer(
            base_model_name="Qwen/Qwen2-0.5B",
            output_dir=os.path.join(BASE, "small_model"),
            data_dir=os.path.join(BASE, "scratch_data"),
            verbose=False,
        )
        print(f"  ✓ Trainers       (continual LoRA + from-scratch Qwen2-0.5B)")

    print()

    # ── Per-question runner ────────────────────────────────

    def run(question: str, domain: str = "physics_mond") -> dict:

        # 1. Check for procedural shortcut (fully automatic pattern)
        auto = brain.try_automatic(question, domain)
        if auto:
            pattern, auto_result = auto
            brain.post_reasoning_update(auto_result, question, domain)
            response = auto_result.result
            print(f"\n  [AUTOMATIC] Procedural pattern applied — no full chain needed")
            return {
                "selected_branch": auto_result.to_pipeline_format(),
                "reasoning":       auto_result,
                "response":        response,
                "verified":        True,
                "confidence":      auto_result.confidence,
                "automatic":       True,
            }

        # 2. Assemble context from all modules
        brain_ctx   = brain.pre_reasoning_context(question, domain)
        psych_ctx   = psych.pre_reasoning_context(question, domain)
        plastic_ctx = plastic.pre_reasoning_context(question, domain)
        calib_ctx   = calib.context_for_reasoning()
        persona_ctx = persona.context_for_reasoning()

        full_ctx = "\n\n".join(
            c for c in [brain_ctx, psych_ctx, plastic_ctx, calib_ctx, persona_ctx]
            if c.strip()
        )
        q_with_ctx = f"{question}\n\n{full_ctx}" if full_ctx else question

        # 3. Run pipeline
        result = pipeline.ask(q_with_ctx, domain=domain)

        reasoning = result.get("reasoning")
        response  = result.get("response", "")

        # 4. Update all modules
        if reasoning:

            # Brain modules
            brain.post_reasoning_update(
                reasoning, question, domain,
                success=reasoning.verified,
            )

            # Psychological evaluation
            feedback = psych.evaluate(reasoning, response, question)

            # Belief system
            belief_events = plastic.process_run(reasoning, response, question)

            # Confidence calibration
            calib.record_claim(
                claimed_confidence=reasoning.confidence,
                actual_verified=reasoning.verified,
                source="derived" if reasoning.structural_fraction > 0.5 else "analogical",
                domain=domain,
            )

            # Personality
            persona.update_from_response(response)

            # Integration monitoring
            signals = {
                "values":           feedback.value_score,
                "aesthetic":        feedback.aesthetic_score,
                "social":           feedback.social_score,
                "emotional_state":  feedback.engagement,
                "confidence":       0.85 if reasoning.confidence == "HIGH"
                                    else 0.6 if reasoning.confidence == "MODERATE"
                                    else 0.3,
            }
            integration, conflicts = integr.compute_integration(signals)
            if conflicts:
                result["integration_conflicts"] = conflicts

            # Temporal milestone if verified
            if reasoning.verified:
                psych.temporal.record_milestone(
                    domain=domain,
                    description=f"verified run ({reasoning.method})",
                    metric="structural_fraction",
                    value=reasoning.structural_fraction,
                )

            # Training
            if trainer and episodic:
                try:
                    recent = episodic.query_recent(n=1)
                    if recent:
                        trainer.on_run_complete(result, recent[0])
                except Exception:
                    pass

            if from_scratch and reasoning.verified:
                from_scratch.add_verified_run(reasoning, question)

            result["feedback"]    = feedback
            result["integration"] = integration

        return result

    # ── First question ────────────────────────────────────

    print("=" * 60)
    print("  First question")
    print("=" * 60 + "\n")

    first_q = (
        "In the CP2 framework, why do flat rotation curves require "
        "axisymmetric disk geometry rather than spherical symmetry?"
    )
    print(f"Q: {first_q}\n")

    result = run(first_q)

    if result.get("response"):
        print(f"\n{result['response'][:400]}")

    vr = result.get("verification_record", {})
    fb = result.get("feedback")
    print(f"\nVerified:    {result.get('verified')}  |  "
          f"Confidence:  {result.get('confidence')}  |  "
          f"Integration: {result.get('integration', 0):.2f}")
    if fb:
        print(f"Psychology:  quality={fb.internal_quality:.2f}  "
              f"emotion={fb.emotional_state}  "
              f"values={fb.value_score:.2f}")

    print(f"\n{brain.status()}")
    print(f"\n{psych.status()}")
    print(f"\n{plastic.status()}")
    if from_scratch:
        print(f"\n{from_scratch.status()}")

    # ── Interactive loop ──────────────────────────────────

    print("\n" + "=" * 60)
    print("  Commands")
    print("  ─────────────────────────────────────────────────────")
    print("  [question]      Ask anything")
    print("  explore         System picks its own question")
    print("  consolidate     Maintenance + full module sync")
    print("  status          Full status of all modules")
    print("  beliefs         Show current belief system state")
    print("  personality     Show personality profile")
    print("  stream on/off   Toggle thought streaming (slow on Mac)")
    print("  Ctrl+C          Stop and export training data")
    print("=" * 60 + "\n")

    try:
        while True:
            q = input("> ").strip()
            if not q:
                continue

            # Commands
            if q.lower() in ("explore", "x"):
                pipeline.explore(n=1)
                continue

            if q.lower() in ("consolidate", "c"):
                print("Running consolidation...")
                pipeline.consolidate()
                obligation_store = None
                try:
                    obligation_store = pipeline._base._enhanced._core.obligation_store
                except Exception:
                    pass
                brain.consolidate(
                    obligation_store=obligation_store,
                    episodic_store=episodic,
                )
                print(brain.status())
                print(psych.status())
                continue

            if q.lower() in ("status", "s"):
                print(brain.status())
                print(psych.status())
                print(plastic.status())
                print(f"Calibration error: {calib.calibration_score():.3f}")
                print(integr.status())
                if trainer:
                    print(trainer.status())
                if from_scratch:
                    print(from_scratch.status())
                continue

            if q.lower() == "beliefs":
                print(plastic.status())
                for b in plastic.beliefs.questioning_beliefs():
                    print(f"  questioning: '{b.statement[:70]}' ({b.confidence:.2f})")
                continue

            if q.lower() == "personality":
                print(persona.context_for_reasoning())
                top = persona.dominant_traits(5)
                for name, strength in top:
                    bar = "█" * int(strength * 10) + "░" * (10 - int(strength * 10))
                    print(f"  {bar} {strength:.2f}  {name}")
                continue

            if q.lower() == "stream on":
                from thought_stream import ThoughtStreamMonitor
                pipeline.stream_monitor = ThoughtStreamMonitor(
                    llm_chat, max_steps=8, verbose=True
                )
                print("Thought streaming ON (~6-10 min/question on Mac)")
                continue

            if q.lower() == "stream off":
                pipeline.stream_monitor = None
                print("Thought streaming OFF")
                continue

            # Ask
            result = run(q)
            response = result.get("response", "")
            if response:
                print(f"\n{response[:500]}")

            fb = result.get("feedback")
            conflicts = result.get("integration_conflicts", [])

            status_parts = [
                f"Verified: {result.get('verified')}",
                f"Confidence: {result.get('confidence')}",
                f"WM: {brain.working_memory.n_slots} slots",
                f"Procedural: {brain.procedural.stats()['n_automatic']} automatic",
            ]
            if fb:
                status_parts.append(f"Quality: {fb.internal_quality:.2f}")
            if conflicts:
                status_parts.append(
                    f"⚑ {len(conflicts)} integration conflict(s)"
                )
            print("\n" + " | ".join(status_parts))

    except KeyboardInterrupt:
        print(f"\n\nSession ended.")

        # Export training data
        if trainer:
            n = trainer.export_dataset(
                os.path.join(BASE, "training_data.jsonl")
            )
            if n:
                print(f"Exported {n} continual training examples")

        if from_scratch:
            n = len(from_scratch._examples)
            if n:
                print(f"From-scratch training buffer: {n} examples")
                print(f"  Saved to {BASE}/scratch_data/all_examples.jsonl")

        pipeline.consolidate(quiet=True)

        # Final psychological state
        print(f"\nFinal state:")
        print(psych.status())
        print(plastic.status())
        print(f"Personality (top 3):")
        for name, strength in persona.dominant_traits(3):
            print(f"  {name}: {strength:.2f}")
        print(f"\nAll data saved to {BASE}/")


if __name__ == "__main__":
    main()
