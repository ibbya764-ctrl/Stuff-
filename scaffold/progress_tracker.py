"""
progress_tracker.py
===================

Tracks and stores system improvement over time.

Takes hourly snapshots of key metrics and stores them persistently.
Provides comparison between now vs yesterday vs last week.
Surfaces: verified answer rate, belief stability, Phi trend,
personality development, domain depth, small model progress.

Usage:
    from progress_tracker import ProgressTracker, start_tracker

    tracker = start_tracker(scaffold, collective)

    # Get full history for dashboard
    tracker.full_history()

    # Get latest snapshot
    tracker.latest_snapshot()

    # Compare progress
    tracker.compare(days=7)
"""

import os
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

_global_tracker: Optional["ProgressTracker"] = None


def get_tracker() -> Optional["ProgressTracker"]:
    return _global_tracker


def start_tracker(
    scaffold:   dict,
    collective  = None,
    base_dir:   str  = "./scaffold_data",
    interval:   int  = 3600,   # snapshot every hour
) -> "ProgressTracker":
    global _global_tracker
    _global_tracker = ProgressTracker(scaffold, collective, base_dir, interval)
    _global_tracker.start()
    return _global_tracker


@dataclass
class Snapshot:
    timestamp:          float
    verified_total:     int     # total verified answers ever
    belief_count:       int     # total beliefs
    questioning_count:  int     # beliefs currently under revision
    phi:                float   # current Φ
    dmn_cycles:         int
    dominant_emotion:   str
    engagement:         float
    collective_total:   int     # shared pool size
    domain_depths:      dict    # {domain: depth_level}
    top_traits:         list    # [(name, strength)]
    # Small model
    model_stage:        str
    model_examples:     int
    model_verify_rate:  float   # 0-1, last test
    # Autonomous
    autonomous_total:   int
    autonomous_verified: int

    def to_dict(self) -> dict:
        return {
            "timestamp":         self.timestamp,
            "verified_total":    self.verified_total,
            "belief_count":      self.belief_count,
            "questioning_count": self.questioning_count,
            "phi":               round(self.phi, 4),
            "dmn_cycles":        self.dmn_cycles,
            "dominant_emotion":  self.dominant_emotion,
            "engagement":        round(self.engagement, 2),
            "collective_total":  self.collective_total,
            "domain_depths":     self.domain_depths,
            "top_traits":        self.top_traits,
            "model_stage":       self.model_stage,
            "model_examples":    self.model_examples,
            "model_verify_rate": round(self.model_verify_rate, 3),
            "autonomous_total":  self.autonomous_total,
            "autonomous_verified": self.autonomous_verified,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(**{k: d.get(k, v) for k, v in {
            "timestamp": 0, "verified_total": 0, "belief_count": 0,
            "questioning_count": 0, "phi": 0, "dmn_cycles": 0,
            "dominant_emotion": "neutral", "engagement": 0.5,
            "collective_total": 0, "domain_depths": {}, "top_traits": [],
            "model_stage": "TRAINING", "model_examples": 0,
            "model_verify_rate": 0, "autonomous_total": 0,
            "autonomous_verified": 0,
        }.items()})


class ProgressTracker:
    """Hourly snapshots of system state for progress tracking."""

    def __init__(
        self,
        scaffold:   dict,
        collective  = None,
        base_dir:   str = "./scaffold_data",
        interval:   int = 3600,
    ):
        self.scaffold   = scaffold
        self.collective = collective
        self.base_dir   = base_dir
        self.interval   = interval
        self._snapshots: list[dict] = []
        self._running   = False
        self._thread:   Optional[threading.Thread] = None
        self._path      = os.path.join(base_dir, "progress.json")
        self._load()

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="BriProgress"
        )
        self._thread.start()
        print("[progress] Tracker started — snapshots every hour")

    def stop(self) -> None:
        self._running = False

    def snapshot_now(self) -> dict:
        """Take a snapshot right now and store it."""
        s = self._take_snapshot()
        self._snapshots.append(s.to_dict())
        if len(self._snapshots) > 720:   # 30 days of hourly snapshots
            self._snapshots = self._snapshots[-720:]
        self._save()
        return s.to_dict()

    def latest_snapshot(self) -> dict:
        if self._snapshots:
            return self._snapshots[-1]
        return {}

    def full_history(self) -> dict:
        return {
            "snapshots": self._snapshots,
            "count":     len(self._snapshots),
            "oldest":    self._snapshots[0]["timestamp"] if self._snapshots else None,
            "newest":    self._snapshots[-1]["timestamp"] if self._snapshots else None,
            "summary":   self._summary(),
        }

    def compare(self, days: int = 7) -> dict:
        """Compare current state to N days ago."""
        if len(self._snapshots) < 2:
            return {}
        now   = self._snapshots[-1]
        cutoff = time.time() - days * 86400
        past  = next(
            (s for s in reversed(self._snapshots) if s["timestamp"] < cutoff),
            self._snapshots[0]
        )
        return {
            "days":              days,
            "phi_change":        round(now["phi"] - past["phi"], 4),
            "verified_gained":   now["verified_total"] - past["verified_total"],
            "beliefs_gained":    now["belief_count"] - past["belief_count"],
            "model_examples_gained": now["model_examples"] - past["model_examples"],
            "model_stage_now":   now["model_stage"],
            "collective_gained": now["collective_total"] - past["collective_total"],
        }

    def _summary(self) -> dict:
        if not self._snapshots:
            return {}
        first = self._snapshots[0]
        last  = self._snapshots[-1]
        hours = (last["timestamp"] - first["timestamp"]) / 3600
        return {
            "hours_running":     round(hours, 1),
            "verified_total":    last["verified_total"],
            "model_stage":       last["model_stage"],
            "phi_now":           last["phi"],
            "phi_start":         first["phi"],
            "collective_total":  last["collective_total"],
        }

    def _loop(self) -> None:
        time.sleep(30)   # short initial wait
        self.snapshot_now()
        while self._running:
            time.sleep(self.interval)
            try:
                self.snapshot_now()
            except Exception as e:
                print(f"[progress] Snapshot error: {e}")

    def _take_snapshot(self) -> Snapshot:
        sc = self.scaffold

        # Beliefs
        belief_count = questioning = 0
        try:
            plastic = sc.get("plastic")
            if plastic:
                all_b = plastic.beliefs.all_beliefs()
                belief_count = len(all_b)
                questioning  = len(plastic.beliefs.questioning_beliefs())
        except Exception: pass

        # DMN / Phi
        phi = dmn_cycles = 0
        emotion = "neutral"; engagement = 0.5
        try:
            dmn = sc.get("dmn")
            if dmn:
                phi        = dmn.mean_phi()
                dmn_cycles = dmn._cycle_count
                emotion    = dmn.workspace.dominant_emotion
                engagement = dmn.workspace.engagement_level
        except Exception: pass

        # Collective
        collective_total = 0
        try:
            if self.collective:
                collective_total = self.collective.stats().get("total_contributions", 0)
        except Exception: pass

        # Personality
        top_traits = []
        try:
            persona = sc.get("persona")
            if persona:
                top_traits = [
                    (n, round(t["strength"], 3))
                    for n, t in sorted(
                        persona.traits.items(),
                        key=lambda x: -x[1]["strength"]
                    )[:4]
                ]
        except Exception: pass

        # Domain depths
        domain_depths = {}
        try:
            knowledge = sc.get("knowledge")
            if knowledge:
                for domain in ["physics","mathematics","biology","economics","philosophy"]:
                    p = knowledge.knowledge_base.domain_profile(domain)
                    domain_depths[domain] = p.get("depth","none")
        except Exception: pass

        # Small model
        model_stage = "TRAINING"; model_examples = 0; model_vr = 0.0
        try:
            fs = sc.get("from_scratch")
            if fs:
                model_stage    = fs._stage
                model_examples = len(fs._examples)
                import os, json
                comp_path = os.path.join(fs.output_dir, "competency.json")
                if os.path.exists(comp_path):
                    records = json.load(open(comp_path)).get("records", [])
                    if records:
                        model_vr = records[-1].get("verify_rate", 0)
        except Exception: pass

        # Verified total (from episodic store)
        verified_total = 0
        try:
            episodic = sc.get("episodic")
            if episodic:
                all_records  = episodic.query_recent(n=10000)
                verified_total = sum(1 for r in all_records if r.verified)
        except Exception: pass

        # Autonomous stats
        auto_total = auto_verified = 0
        try:
            from autonomous_loop import get_loop
            loop = get_loop()
            if loop:
                auto_total    = loop._stats.get("total_asked", 0)
                auto_verified = loop._stats.get("total_verified", 0)
        except Exception: pass

        return Snapshot(
            timestamp=time.time(),
            verified_total=verified_total,
            belief_count=belief_count,
            questioning_count=questioning,
            phi=phi, dmn_cycles=dmn_cycles,
            dominant_emotion=emotion, engagement=engagement,
            collective_total=collective_total,
            domain_depths=domain_depths,
            top_traits=top_traits,
            model_stage=model_stage,
            model_examples=model_examples,
            model_verify_rate=model_vr,
            autonomous_total=auto_total,
            autonomous_verified=auto_verified,
        )

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                self._snapshots = json.load(f).get("snapshots", [])
        except Exception: pass

    def _save(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump({"snapshots": self._snapshots}, f)
