"""
meta_optimizer.py  v2  —  Level 2 + self-improving strategy logic
==================================================================

Level 1: task performance (reasoning, verifying answers)
Level 2: learning parameter optimisation (belief rates, thresholds)
Level 2+: improvement strategy optimisation (which rules to use)
Level 2++: rule generation prompt optimisation
Ceiling: the LLM (external, fixed capability)

The key insight: the improvement logic doesn't have to be Python code.
It can be a set of learnable rules stored in JSON:

  "if verify_rate_30 < 0.45 and domain is physics,
   decrease belief_update_lr_physics"

Each rule has a success rate — how often following it actually improved
things. Rules are selected using UCB1 (Upper Confidence Bound) which
balances exploiting known-good rules with exploring underused ones.
When existing rules consistently fail, the LLM generates new ones.

This is the improvement logic improving itself:
  - Rules determine what the meta-optimizer does
  - Rule success rates are updated from outcomes
  - Weak rules are pruned; new rules are generated
  - The rule generation prompts are themselves A/B tested

Everything that moves is data, not code. No Python self-modification.
"""

import os, json, math, time, random, threading, secrets
from dataclasses import dataclass, field
from typing import Optional, Callable


# ============================================================
# Parameter registry (what Level 2 can adjust)
# ============================================================

PARAMETER_DEFAULTS = {
    "belief_update_lr_physics":        0.30,
    "belief_update_lr_mathematics":    0.30,
    "belief_update_lr_biology":        0.30,
    "belief_update_lr_economics":      0.30,
    "belief_update_lr_philosophy":     0.30,
    "belief_update_lr_general":        0.30,
    "assumption_auditor_threshold":    3,
    "philosophical_inquiry_threshold": 6,
    "automatization_threshold":        7,
    "training_recency_weight":         0.60,
    "training_min_confidence":         0.50,
    "context_weight_brain":            0.70,
    "context_weight_psych":            0.50,
    "context_weight_plastic":          0.50,
    "context_weight_knowledge":        0.80,
    "context_weight_dmn":              0.90,
    "context_weight_collective":       0.60,
    "dmn_cycle_seconds":               45,
    "reasoning_prompt_variant":        0,
}

PARAMETER_BOUNDS = {
    "belief_update_lr_physics":        (0.05, 0.80, 0.05),
    "belief_update_lr_mathematics":    (0.05, 0.80, 0.05),
    "belief_update_lr_biology":        (0.05, 0.80, 0.05),
    "belief_update_lr_economics":      (0.05, 0.80, 0.05),
    "belief_update_lr_philosophy":     (0.05, 0.80, 0.05),
    "belief_update_lr_general":        (0.05, 0.80, 0.05),
    "assumption_auditor_threshold":    (2, 8, 1),
    "philosophical_inquiry_threshold": (4, 12, 1),
    "automatization_threshold":        (4, 15, 1),
    "training_recency_weight":         (0.20, 0.95, 0.05),
    "training_min_confidence":         (0.30, 0.80, 0.05),
    "context_weight_brain":            (0.10, 1.0, 0.10),
    "context_weight_psych":            (0.10, 1.0, 0.10),
    "context_weight_plastic":          (0.10, 1.0, 0.10),
    "context_weight_knowledge":        (0.10, 1.0, 0.10),
    "context_weight_dmn":              (0.10, 1.0, 0.10),
    "context_weight_collective":       (0.10, 1.0, 0.10),
    "dmn_cycle_seconds":               (20, 120, 5),
    "reasoning_prompt_variant":        (0, 2, 1),
}


# ============================================================
# MetaRule — one learnable improvement strategy
# ============================================================

@dataclass
class MetaRule:
    """
    One improvement strategy. Data, not code.
    Can be generated, evaluated, and pruned.
    """
    rule_id:      str
    condition:    dict    # {"metric": str, "comparison": "<"|">"|"==", "threshold": float}
    action:       dict    # {"parameter": str, "direction": "increase"|"decrease"}
    description:  str
    n_trials:     int   = 0
    n_successes:  int   = 0
    success_rate: float = 0.5
    generated_by: str   = "default"   # "default" | "llm" | "evolved"
    created_at:   float = field(default_factory=time.time)
    last_used:    float = 0.0

    def matches(self, metrics: dict) -> bool:
        """Does this rule's condition hold given current metrics?"""
        v  = metrics.get(self.condition.get("metric",""), None)
        if v is None: return False
        t  = self.condition.get("threshold", 0)
        op = self.condition.get("comparison", "<")
        if op == "<":  return v < t
        if op == ">":  return v > t
        if op == "==": return abs(v - t) < 0.01
        return False

    def ucb1_score(self, total_trials: int) -> float:
        """
        Upper Confidence Bound score.
        Balances exploiting high-success rules with exploring underused ones.
        Unexplored rules (n_trials=0) get infinite priority.
        """
        if self.n_trials == 0:
            return float("inf")
        return (self.success_rate
                + math.sqrt(2.0 * math.log(max(1, total_trials))
                            / self.n_trials))

    def record_outcome(self, improved: bool) -> None:
        self.n_trials    += 1
        self.n_successes += int(improved)
        self.success_rate = self.n_successes / self.n_trials
        self.last_used    = time.time()

    def is_weak(self) -> bool:
        """True if this rule has been tried enough to declare it ineffective."""
        return self.n_trials >= 8 and self.success_rate < 0.25

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "MetaRule":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# Default rule set — the starting improvement logic
def default_rules() -> list[MetaRule]:
    return [
        MetaRule("r001",
            {"metric": "verify_rate_30", "comparison": "<", "threshold": 0.45},
            {"parameter": "reasoning_prompt_variant", "direction": "increase"},
            "low verify rate → try different reasoning prompt",
            generated_by="default"),
        MetaRule("r002",
            {"metric": "calibration_error", "comparison": ">", "threshold": 0.35},
            {"parameter": "belief_update_lr_general", "direction": "increase"},
            "high calibration error → faster belief updates",
            generated_by="default"),
        MetaRule("r003",
            {"metric": "calibration_error", "comparison": ">", "threshold": 0.40},
            {"parameter": "belief_update_lr_general", "direction": "decrease"},
            "very high calibration error → slower belief updates (overreacting)",
            generated_by="default"),
        MetaRule("r004",
            {"metric": "phi", "comparison": "<", "threshold": 0.20},
            {"parameter": "dmn_cycle_seconds", "direction": "decrease"},
            "low integration → run DMN more frequently",
            generated_by="default"),
        MetaRule("r005",
            {"metric": "belief_stability", "comparison": "<", "threshold": 0.50},
            {"parameter": "assumption_auditor_threshold", "direction": "increase"},
            "unstable beliefs → raise threshold before questioning assumptions",
            generated_by="default"),
        MetaRule("r006",
            {"metric": "auto_rate", "comparison": "<", "threshold": 0.10},
            {"parameter": "automatization_threshold", "direction": "decrease"},
            "few automatic answers → lower threshold for automatization",
            generated_by="default"),
        MetaRule("r007",
            {"metric": "verify_rate_30", "comparison": ">", "threshold": 0.70},
            {"parameter": "context_weight_dmn", "direction": "increase"},
            "high verify rate → lean more on DMN context",
            generated_by="default"),
        MetaRule("r008",
            {"metric": "verify_rate_30", "comparison": "<", "threshold": 0.40},
            {"parameter": "training_recency_weight", "direction": "increase"},
            "low verify rate → weight recent training examples more",
            generated_by="default"),
    ]


# ============================================================
# MetaRuleSystem — the self-improving strategy layer
# ============================================================

class MetaRuleSystem:
    """
    Manages the set of improvement strategies.

    Rules are selected via UCB1, evaluated after outcomes,
    pruned when ineffective, and regenerated via LLM when the
    full set is underperforming.

    This is the improvement logic improving itself.
    """

    PRUNE_INTERVAL     = 20    # check for weak rules every N evaluations
    GENERATE_THRESHOLD = 0.35  # generate new rules when mean success rate < this
    MAX_RULES          = 30

    def __init__(
        self,
        llm_fn:   Optional[Callable] = None,
        path:     str = "./scaffold_data/meta_rules.json",
        verbose:  bool = True,
    ):
        self.llm     = llm_fn
        self.path    = path
        self.verbose = verbose
        self.rules:  list[MetaRule] = []
        self._total_trials = 0
        self._eval_count   = 0
        self._load()
        if not self.rules:
            self.rules = default_rules()
            self._save()

    def select_rule(self, metrics: dict) -> Optional[MetaRule]:
        """Select the best applicable rule using UCB1."""
        applicable = [r for r in self.rules if r.matches(metrics)]
        if not applicable:
            return None
        return max(applicable, key=lambda r: r.ucb1_score(self._total_trials))

    def record_outcome(self, rule: MetaRule, improved: bool) -> None:
        """Update rule success rate and trigger maintenance."""
        rule.record_outcome(improved)
        self._total_trials += 1
        self._eval_count   += 1

        if self._eval_count % self.PRUNE_INTERVAL == 0:
            self._prune_weak_rules()
            self._maybe_generate_new_rules(metrics={})   # triggers if overall SR is low
            self._save()

    def mean_success_rate(self) -> float:
        tried = [r for r in self.rules if r.n_trials > 0]
        if not tried: return 0.5
        return sum(r.success_rate for r in tried) / len(tried)

    def generate_new_rules_from_history(self, metrics: dict, history: list) -> list[MetaRule]:
        """
        Ask the LLM to generate new improvement strategies.
        This is Level 2++ — the rule generation logic using external intelligence.
        """
        if not self.llm:
            return self._evolve_existing_rules()

        tried = [(r.description, round(r.success_rate, 2), r.n_trials)
                 for r in self.rules if r.n_trials >= 3][:8]
        tried_str = "\n".join(f"  - {d}: {sr:.0%} success ({n} trials)"
                               for d, sr, n in tried)

        system = (
            "You generate new meta-learning rules for an AI self-improvement system. "
            "Each rule specifies: when to change a learning parameter, which one, and how. "
            "Output ONLY valid JSON array. No explanation."
        )
        user = (
            f"Current metrics: {json.dumps({k: round(v,3) for k,v in metrics.items()})}\n\n"
            f"Rules tried so far:\n{tried_str}\n\n"
            f"Available parameters to adjust:\n"
            f"{list(PARAMETER_DEFAULTS.keys())}\n\n"
            f"Available metrics: verify_rate_30, verify_rate_100, calibration_error, "
            f"phi, belief_stability, auto_rate\n\n"
            "Generate 2-3 new improvement rules that haven't been tried. "
            "Format:\n"
            '[{"condition":{"metric":"verify_rate_30","comparison":"<","threshold":0.5},'
            '"action":{"parameter":"belief_update_lr_general","direction":"decrease"},'
            '"description":"..."}]'
        )
        try:
            raw  = self.llm(system, user).strip()
            import re
            m = re.search(r'\[.*\]', raw, re.DOTALL)
            if not m: return []
            data = json.loads(m.group())
            new_rules = []
            for d in data[:3]:
                r = MetaRule(
                    rule_id=f"llm_{secrets.token_hex(3)}",
                    condition=d.get("condition", {}),
                    action=d.get("action", {}),
                    description=d.get("description", "LLM-generated rule"),
                    generated_by="llm",
                )
                # Validate
                if (r.condition.get("metric") in
                    ["verify_rate_30","verify_rate_100","calibration_error",
                     "phi","belief_stability","auto_rate"]
                    and r.action.get("parameter") in PARAMETER_DEFAULTS
                    and r.action.get("direction") in ("increase","decrease")):
                    new_rules.append(r)
            if self.verbose and new_rules:
                print(f"  [meta-rules] LLM generated {len(new_rules)} new rules")
            return new_rules
        except Exception as e:
            if self.verbose:
                print(f"  [meta-rules] LLM generation failed: {e}")
            return self._evolve_existing_rules()

    def _evolve_existing_rules(self) -> list[MetaRule]:
        """
        Fallback: mutate existing high-performing rules slightly.
        Poor man's LLM when no API available.
        """
        good = sorted(self.rules, key=lambda r: r.success_rate, reverse=True)[:3]
        evolved = []
        for r in good:
            bounds = PARAMETER_BOUNDS.get(r.action.get("parameter",""), None)
            if not bounds: continue
            new_threshold = r.condition.get("threshold", 0.5)
            # Slightly shift threshold
            new_threshold = round(new_threshold + random.choice([-0.05, 0.05]), 3)
            evolved.append(MetaRule(
                rule_id=f"evo_{secrets.token_hex(3)}",
                condition={**r.condition, "threshold": new_threshold},
                action=r.action.copy(),
                description=f"Evolved from: {r.description}",
                generated_by="evolved",
            ))
        return evolved[:2]

    def _prune_weak_rules(self) -> None:
        weak = [r for r in self.rules if r.is_weak()]
        for r in weak:
            self.rules.remove(r)
            if self.verbose:
                print(f"  [meta-rules] Pruned: '{r.description[:50]}' "
                      f"(success={r.success_rate:.0%} after {r.n_trials} trials)")

    def _maybe_generate_new_rules(self, metrics: dict) -> None:
        if (self.mean_success_rate() < self.GENERATE_THRESHOLD
                and len(self.rules) < self.MAX_RULES):
            new = self.generate_new_rules_from_history(metrics, [])
            self.rules.extend(new)

    def _load(self) -> None:
        if not os.path.exists(self.path): return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.rules         = [MetaRule.from_dict(d) for d in data.get("rules", [])]
            self._total_trials = data.get("total_trials", 0)
        except Exception: pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "rules":         [r.to_dict() for r in self.rules],
                "total_trials":  self._total_trials,
                "mean_sr":       round(self.mean_success_rate(), 3),
            }, f, indent=2)


# ============================================================
# MetricsTracker
# ============================================================

class MetricsTracker:
    def __init__(self, scaffold: dict, path: str):
        self.scaffold  = scaffold
        self.path      = path
        self._outcomes: list[dict] = []
        self._load()

    def record_outcome(self, domain, verified, confidence, automatic=False):
        conf_val = {"HIGH":0.85,"MODERATE":0.6,"LOW":0.35,"UNCERTAIN":0.2}.get(confidence,0.5)
        self._outcomes.append({"t":time.time(),"domain":domain,"verified":verified,
                                "conf_val":conf_val,"automatic":automatic})
        if len(self._outcomes) > 2000: self._outcomes = self._outcomes[-2000:]
        self._save()

    def rolling_verify_rate(self, n=30, domain=""):
        o = [x for x in self._outcomes if not domain or x["domain"]==domain][-n:]
        return sum(1 for x in o if x["verified"])/max(1,len(o))

    def calibration_error(self, n=50):
        r = [o for o in self._outcomes[-n:] if not o["automatic"]]
        if len(r) < 5: return 0.3
        return round(sum(abs(o["conf_val"]-(1.0 if o["verified"] else 0.0)) for o in r)/len(r),4)

    def current_metrics(self) -> dict:
        sc = self.scaffold
        phi = 0.0
        try: phi = sc["dmn"].mean_phi() if sc.get("dmn") else 0.0
        except Exception: pass
        belief_stability = 0.7
        try:
            p = sc.get("plastic")
            if p:
                all_b = p.beliefs.all_beliefs()
                q     = p.beliefs.questioning_beliefs()
                belief_stability = 1.0 - len(q)/max(1,len(all_b))
        except Exception: pass
        auto_rate = len([o for o in self._outcomes[-50:] if o.get("automatic")])/max(1,min(50,len(self._outcomes)))
        return {
            "verify_rate_30":  self.rolling_verify_rate(30),
            "verify_rate_100": self.rolling_verify_rate(100),
            "calibration_error": self.calibration_error(),
            "phi":             phi,
            "belief_stability": belief_stability,
            "auto_rate":       auto_rate,
        }

    def outcomes_since(self, t):
        return [o for o in self._outcomes if o["t"] > t]

    def _load(self):
        if not os.path.exists(self.path): return
        try:
            with open(self.path) as f: self._outcomes = json.load(f).get("outcomes",[])
        except Exception: pass

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path,"w") as f: json.dump({"outcomes":self._outcomes[-2000:]},f)


# ============================================================
# ParameterChange
# ============================================================

@dataclass
class ParameterChange:
    change_id:   str
    parameter:   str
    old_value:   float
    new_value:   float
    timestamp:   float
    baseline_vr: float
    n_before:    int
    rule_id:     str   = ""
    post_vr:     float = 0.0
    n_after:     int   = 0
    effect:      float = 0.0
    adopted:     bool  = False
    evaluated:   bool  = False
    evaluate_after: int = 40


# ============================================================
# MetaOptimizer — orchestrates everything
# ============================================================

class MetaOptimizer:
    """
    Orchestrates the full self-improving stack.

    Level 1 (scaffold): learns from task outcomes
    Level 2 (this):     adjusts which parameters govern Level 1 learning
    Level 2+ (rules):   adjusts which strategies govern Level 2 decisions
    Level 2++ (LLM):    generates new strategies when existing ones fail
    """

    CYCLE_SECONDS   = 1800
    EVAL_THRESHOLD  = 40
    ADOPT_THRESHOLD = 0.04

    def __init__(self, scaffold, llm_fn=None, base_dir="./scaffold_data", verbose=True):
        self.scaffold = scaffold
        self.verbose  = verbose
        self.params   = dict(PARAMETER_DEFAULTS)
        self.metrics  = MetricsTracker(scaffold, os.path.join(base_dir,"meta_outcomes.json"))
        self.rules    = MetaRuleSystem(
            llm_fn=llm_fn,
            path=os.path.join(base_dir,"meta_rules.json"),
            verbose=verbose,
        )
        self._changes: list[ParameterChange] = []
        self._pending: list[ParameterChange] = []
        self._log:     list[dict] = []
        self._running  = False
        self._thread   = None
        self._path     = os.path.join(base_dir,"meta_optimizer.json")
        self._load()

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True, name="MetaOpt")
        self._thread.start()
        if self.verbose: print("[meta] Level 2+ optimizer started")

    def stop(self): self._running = False

    def record_outcome(self, domain, verified, confidence, automatic=False):
        self.metrics.record_outcome(domain, verified, confidence, automatic)
        self._try_evaluate_pending()

    def get_param(self, name, default=None):
        return self.params.get(name, default if default is not None
                               else PARAMETER_DEFAULTS.get(name))

    def status(self) -> dict:
        m = self.metrics.current_metrics()
        rule_stats = [{
            "description": r.description[:60],
            "trials":      r.n_trials,
            "success_rate": round(r.success_rate, 2),
            "generated_by": r.generated_by,
        } for r in sorted(self.rules.rules, key=lambda x:-x.success_rate)[:5]]
        return {
            "verify_rate_30":   round(m["verify_rate_30"],3),
            "verify_rate_100":  round(m["verify_rate_100"],3),
            "calibration_err":  m["calibration_error"],
            "n_changes_tried":  len(self._changes),
            "n_adopted":        sum(1 for c in self._changes if c.adopted),
            "n_rules":          len(self.rules.rules),
            "rules_mean_sr":    round(self.rules.mean_success_rate(),3),
            "top_rules":        rule_stats,
            "pending":          len(self._pending),
        }

    # ── Main loop ──────────────────────────────────────────

    def _loop(self):
        time.sleep(120)
        while self._running:
            try: self._one_cycle()
            except Exception as e:
                if self.verbose: print(f"[meta] Cycle error: {e}")
            time.sleep(self.CYCLE_SECONDS)

    def _one_cycle(self):
        if len(self.metrics._outcomes) < 20: return
        if self._pending: return   # wait for current test to finish

        metrics = self.metrics.current_metrics()
        if self.verbose:
            print(f"\n[meta] Cycle — vr30={metrics['verify_rate_30']:.3f} "
                  f"cal={metrics['calibration_error']:.3f} "
                  f"rules_sr={self.rules.mean_success_rate():.2f}")

        # Select rule using UCB1
        rule = self.rules.select_rule(metrics)
        if not rule:
            if self.verbose: print("  [meta] No applicable rule found")
            return

        if self.verbose:
            print(f"  [meta] Rule: '{rule.description[:60]}' "
                  f"(sr={rule.success_rate:.0%}, trials={rule.n_trials})")

        # Propose change from rule
        param     = rule.action.get("parameter","")
        direction = rule.action.get("direction","increase")
        change    = self._propose_change(param, direction, metrics, rule.rule_id)
        if not change: return

        self._apply_change(change)
        self._pending.append(change)
        self._changes.append(change)
        self._save()

    def _propose_change(self, param, direction, metrics, rule_id="") -> Optional[ParameterChange]:
        if param not in PARAMETER_BOUNDS: return None
        current = self.params.get(param, PARAMETER_DEFAULTS.get(param, 0))
        lo, hi, step = PARAMETER_BOUNDS[param]
        new_val = min(hi, current+step) if direction=="increase" else max(lo, current-step)
        if new_val == current: return None
        return ParameterChange(
            change_id=secrets.token_hex(4), parameter=param,
            old_value=current, new_value=new_val,
            timestamp=time.time(), baseline_vr=metrics.get("verify_rate_30",0.5),
            n_before=len(self.metrics._outcomes), rule_id=rule_id,
        )

    def _apply_change(self, change: ParameterChange):
        self.params[change.parameter] = change.new_value
        sc = self.scaffold
        try:
            if change.parameter.startswith("belief_update_lr_"):
                domain = change.parameter.replace("belief_update_lr_","")
                p = sc.get("plastic")
                if p and hasattr(p.beliefs,"set_update_rate"):
                    p.beliefs.set_update_rate(domain, change.new_value)
            elif change.parameter == "dmn_cycle_seconds":
                d = sc.get("dmn")
                if d: d.cycle_seconds = int(change.new_value)
            elif change.parameter == "assumption_auditor_threshold":
                p = sc.get("plastic")
                if p and hasattr(p,"assumption_auditor"):
                    p.assumption_auditor.threshold = int(change.new_value)
            elif change.parameter == "automatization_threshold":
                b = sc.get("brain")
                if b and hasattr(b,"procedural"):
                    b.procedural.AUTOMATISATION_THRESHOLD = int(change.new_value)
        except Exception: pass

    def _try_evaluate_pending(self):
        if not self._pending: return
        change = self._pending[0]
        since  = self.metrics.outcomes_since(change.timestamp)
        if len(since) < change.evaluate_after: return

        post_vr = sum(1 for o in since if o["verified"]) / len(since)
        change.post_vr   = post_vr
        change.n_after   = len(since)
        change.effect    = post_vr - change.baseline_vr
        change.evaluated = True

        # Find the rule that triggered this
        rule = next((r for r in self.rules.rules if r.rule_id==change.rule_id), None)

        if change.effect > self.ADOPT_THRESHOLD:
            change.adopted = True
            if rule: self.rules.record_outcome(rule, improved=True)
            if self.verbose:
                print(f"\n[meta] ADOPTED {change.parameter}: "
                      f"{change.old_value}→{change.new_value} "
                      f"(+{change.effect:.3f})")
        else:
            change.adopted = False
            # Revert
            self.params[change.parameter] = change.old_value
            self._apply_change(ParameterChange(
                change_id="revert", parameter=change.parameter,
                old_value=change.new_value, new_value=change.old_value,
                timestamp=time.time(), baseline_vr=0, n_before=0,
            ))
            if rule: self.rules.record_outcome(rule, improved=False)
            if self.verbose:
                print(f"\n[meta] REVERTED {change.parameter} "
                      f"(effect={change.effect:.3f})")

        self._pending.remove(change)
        self._save()

    def _load(self):
        if not os.path.exists(self._path): return
        try:
            with open(self._path) as f: d = json.load(f)
            self.params.update(d.get("params",{}))
            self._log = d.get("log",[])
        except Exception: pass

    def _save(self):
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path,"w") as f:
            json.dump({"params":self.params,"log":self._log[-200:],
                       "changes":[vars(c) for c in self._changes[-100:]]},f,indent=2)


# ── Global instance ────────────────────────────────────────

_global_meta: Optional[MetaOptimizer] = None

def get_meta() -> Optional[MetaOptimizer]: return _global_meta

def start_meta_optimizer(scaffold, llm_fn=None, base_dir="./scaffold_data", verbose=True):
    global _global_meta
    _global_meta = MetaOptimizer(scaffold, llm_fn=llm_fn, base_dir=base_dir, verbose=verbose)
    _global_meta.start()
    return _global_meta
