"""
cross_domain_composer.py
========================

Cross-domain analogical pattern matching for the technique library.

The base TechniqueComposer matches problems to techniques within the
same domain. This extends it with a layer that recognises STRUCTURAL
ISOMORPHISMS across domains — the same abstract reasoning move
appearing in physics, economics, biology, and code.

The principle: surface vocabulary is domain-specific, but abstract
structure is not. A "mean-field approximation" in spin systems, a
"representative agent" model in macroeconomics, and a "well-mixed
population" assumption in epidemiology are the same move under
different names. A "perturbative expansion in small parameter" appears
identically in quantum field theory, asymptotic analysis, and
sensitivity analysis of ML models.

This module:

  1. Defines ABSTRACT_STRUCTURES — a domain-independent vocabulary of
     reasoning patterns. Each pattern has trigger keywords drawn from
     multiple domains, plus example instantiations.

  2. Computes an abstract-structure fingerprint for techniques and
     problems alongside the existing concrete fingerprint.

  3. Surfaces cross-domain ANALOGIES: techniques from other domains
     whose abstract structure matches the current problem, with
     explicit framing so the LLM evaluates whether the mapping is
     valid rather than blindly applying it.

This is the step from "remembered moves in a domain" toward
"recognising the same move across domains" — the kind of pattern
matching that drove most of the major scientific unifications of the
last century (Boltzmann's stat mech as information theory; Wiener's
cybernetics as control + biology; modern ML as statistical physics).

Drop alongside `technique_composer.py`. No changes to that file
required — this wraps it.
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional

from technique_composer import (
    StructuralFingerprint, TechniqueComposer,
    fingerprint_technique, fingerprint_problem,
    _tokenise, _cosine,
)


# ============================================================
# Abstract structure vocabulary
#
# Each pattern is defined by:
#   - description : what the move does at the abstract level
#   - triggers    : keywords drawn from MULTIPLE domains so the same
#                   pattern is detected regardless of vocabulary
#   - examples    : concrete instantiations in different domains
#                   (used to help the LLM understand the abstraction
#                    when an analogy is surfaced)
# ============================================================

ABSTRACT_STRUCTURES: dict[str, dict] = {

    # --- Generative / postulation moves ---

    "parametrized_ansatz_with_constraint": {
        "description": (
            "Introduce a parametrized functional form for an unknown "
            "object, then fix parameters by demanding consistency with "
            "boundary behaviour, conservation, or known limits."
        ),
        "triggers": {
            "ansatz", "trial form", "parametric", "parametrize",
            "parametrise", "guess form", "postulate form",
            "trial function", "test function", "candidate solution",
            "functional form", "hypothesis function",
            "model class", "specification", "structural assumption",
        },
        "examples": [
            "Physics: ansatz for halo coefficient h(M) = M^alpha, fix "
            "alpha by demanding Newtonian limit",
            "Economics: assume utility u(c) = c^(1-sigma)/(1-sigma), fix "
            "sigma by matching observed risk-aversion",
            "ML: posit a model class, fit parameters by matching "
            "training distribution",
            "Biology: posit a Hill function for binding response, fit "
            "Hill coefficient to dose-response data",
        ],
    },

    # --- Approximation by separation of scales ---

    "perturbative_expansion": {
        "description": (
            "Identify a small parameter; expand the problem in powers "
            "of it; retain leading-order behaviour while dropping "
            "higher-order corrections."
        ),
        "triggers": {
            "perturbation", "perturbative", "small parameter",
            "leading order", "leading-order", "first order", "order epsilon",
            "weak coupling", "weak field", "weak-field", "small amplitude",
            "linearise", "linearize", "expansion in", "small deviation",
            "Taylor expand", "asymptotic", "asymptotics",
        },
        "examples": [
            "Physics: weak-field expansion of Einstein equations",
            "Economics: log-linearise around steady state for DSGE models",
            "Biology: small-perturbation analysis of gene-regulatory "
            "networks near a fixed point",
            "ML: Taylor expand the loss around current weights for "
            "second-order optimisation",
        ],
    },

    # --- Reduction by aggregation ---

    "mean_field_aggregation": {
        "description": (
            "Replace pairwise or local interactions with the effect of "
            "an averaged or representative interaction, reducing a "
            "many-body problem to a one-body problem in the average."
        ),
        "triggers": {
            "mean field", "mean-field", "average", "averaged",
            "representative agent", "well mixed", "well-mixed",
            "aggregate", "aggregation", "homogeneous", "ensemble average",
            "effective field", "self-consistent", "molecular field",
            "background field", "average opponent",
        },
        "examples": [
            "Physics: mean-field for Ising; replace neighbour spins "
            "with average magnetisation",
            "Economics: representative-agent macro models",
            "Epidemiology: well-mixed SIR — no spatial structure",
            "Game theory: assume opponents play the population mix",
        ],
    },

    # --- Separation of timescales ---

    "fast_slow_separation": {
        "description": (
            "When a system has variables evolving on very different "
            "timescales, treat fast variables as slaved to slow ones "
            "(adiabatic elimination) and derive effective dynamics for "
            "the slow variables alone."
        ),
        "triggers": {
            "adiabatic", "fast slow", "fast-slow", "slow manifold",
            "slow variable", "fast variable", "quasi-steady", "quasi steady",
            "quasi-static", "separation of timescales", "timescale",
            "integrate out", "marginalise out", "marginalize out",
            "eliminate fast", "slaved", "centre manifold", "center manifold",
        },
        "examples": [
            "Chemistry: quasi-steady-state approximation for enzyme "
            "intermediates",
            "Physics: Born–Oppenheimer separation of nuclear and "
            "electronic motion",
            "Economics: long-run vs short-run equilibrium in "
            "macroeconomic models",
            "Neuroscience: separate fast spiking dynamics from slow "
            "synaptic weight changes",
        ],
    },

    # --- Linearisation around equilibrium ---

    "linear_stability_around_fixed_point": {
        "description": (
            "Find an equilibrium or fixed point; expand the dynamics "
            "around it linearly; analyse eigenvalues of the linearised "
            "operator to determine stability and characteristic modes."
        ),
        "triggers": {
            "fixed point", "equilibrium", "steady state", "stable",
            "unstable", "stability analysis", "linear stability",
            "eigenvalue", "eigenmode", "Jacobian", "linearisation",
            "linearization", "small deviation", "normal mode",
        },
        "examples": [
            "Physics: linear stability of a stationary plasma "
            "configuration via MHD eigenmodes",
            "Ecology: stability of predator–prey equilibrium via "
            "Jacobian eigenvalues",
            "Economics: local stability of macro equilibrium",
            "ML: Hessian eigenvalues at a loss-landscape minimum",
        ],
    },

    # --- Conservation / symmetry arguments ---

    "symmetry_to_conservation": {
        "description": (
            "Identify a symmetry of the system; derive an associated "
            "conserved quantity or invariant that constrains the "
            "answer without solving the full dynamics."
        ),
        "triggers": {
            "symmetry", "invariant", "conserved", "conservation",
            "Noether", "gauge", "translation invariance", "rotation",
            "scale invariance", "invariant under", "preserved",
        },
        "examples": [
            "Physics: time translation → energy conservation (Noether)",
            "Mathematics: solving ODEs by finding an integral of motion",
            "Economics: invariance under monetary unit → homogeneity "
            "constraints on demand functions",
            "CS: loop invariants constraining algorithm correctness",
        ],
    },

    # --- Dimensional / scaling arguments ---

    "scaling_argument": {
        "description": (
            "Use dimensional analysis or scaling relations to constrain "
            "the form of an answer up to a dimensionless function, "
            "without solving the full problem."
        ),
        "triggers": {
            "dimensional analysis", "dimensions", "scale", "scaling",
            "scale invariance", "self-similar", "self similar", "Buckingham",
            "characteristic scale", "natural scale", "dimensionless",
            "rescale", "non-dimensionalise", "non-dimensionalize",
        },
        "examples": [
            "Physics: Reynolds number constrains fluid drag",
            "Biology: Kleiber's metabolic scaling law from area/volume",
            "Economics: scaling of city productivity with population",
            "ML: scaling laws relating compute, data, model size",
        ],
    },

    # --- Verification / falsification ---

    "verification_by_sampling": {
        "description": (
            "Test a derived or claimed relation by sampling the "
            "parameter space numerically; surface specific failures "
            "rather than reasoning about generic correctness."
        ),
        "triggers": {
            "sample", "sampling", "Monte Carlo", "numerical check",
            "counterexample", "falsify", "falsification", "test",
            "validation", "cross validation", "out-of-sample",
            "stress test", "empirical check", "robustness check",
        },
        "examples": [
            "Physics: Monte Carlo sampling of parameter space to "
            "falsify a derived inequality",
            "Statistics: bootstrap to test confidence interval coverage",
            "ML: out-of-sample validation",
            "Engineering: stress testing a design at corner cases",
        ],
    },

    # --- Decomposition ---

    "decomposition_to_subproblems": {
        "description": (
            "Split a hard problem into structurally simpler "
            "subproblems whose solutions combine to solve the whole. "
            "The key is choosing a decomposition that respects the "
            "problem's natural structure."
        ),
        "triggers": {
            "decompose", "decomposition", "split", "partition",
            "divide and conquer", "modular", "factor", "factorise",
            "factorize", "subproblem", "subsystem", "submodule",
            "isolate", "separate variable", "separation of variables",
        },
        "examples": [
            "Physics: separation of variables in PDEs",
            "Economics: decomposing utility into income and "
            "substitution effects",
            "CS: divide-and-conquer algorithms",
            "Biology: decomposing a phenotype into modular pathways",
        ],
    },

    # --- Coarse-graining ---

    "coarse_graining_renormalisation": {
        "description": (
            "Average over fine-scale degrees of freedom to derive "
            "effective dynamics at a larger scale; iterate to find "
            "scale-invariant or fixed-point behaviour."
        ),
        "triggers": {
            "coarse grain", "coarse-grain", "renormalisation",
            "renormalization", "RG", "block", "blocking", "effective theory",
            "effective dynamics", "integrate out", "scale transformation",
            "RG flow",
        },
        "examples": [
            "Physics: Wilsonian RG for critical phenomena",
            "ML: feature hierarchy in deep networks as coarse-graining",
            "Sociology: aggregating micro-behaviours into population-"
            "level statistics",
            "Network theory: graph coarsening preserving spectral "
            "properties",
        ],
    },

    # --- Variational / optimisation ---

    "variational_principle": {
        "description": (
            "Cast the problem as the extremum (minimum/maximum) of a "
            "functional or objective; characterise solutions by "
            "stationarity conditions instead of solving directly."
        ),
        "triggers": {
            "variational", "minimise", "minimize", "maximise", "maximize",
            "stationary", "extremum", "Lagrangian", "Hamilton's principle",
            "least action", "objective function", "loss", "cost function",
            "optimal", "optimum",
        },
        "examples": [
            "Physics: Lagrangian mechanics — paths as stationary points "
            "of action",
            "ML: training as loss minimisation",
            "Economics: utility maximisation under constraint",
            "Biology: evolutionarily stable strategies as fitness optima",
        ],
    },

    # --- Feedback / recursive structure ---

    "feedback_loop_analysis": {
        "description": (
            "Identify a closed loop where output influences input; "
            "characterise the loop's stability, gain, and "
            "fixed-point behaviour to predict system response."
        ),
        "triggers": {
            "feedback", "loop", "self reinforcing", "self-reinforcing",
            "vicious cycle", "virtuous cycle", "positive feedback",
            "negative feedback", "circular", "homeostasis",
            "recursive", "fixed point iteration",
        },
        "examples": [
            "Engineering: control-loop stability analysis",
            "Biology: homeostatic regulation of body temperature",
            "Economics: wage-price spirals; market expectations",
            "Psychology: confirmation bias as evidence-belief feedback",
        ],
    },

    # --- Reduction to a known case ---

    "reduction_to_known_case": {
        "description": (
            "Map the current problem onto one whose solution is "
            "already known, by establishing an equivalence or "
            "transformation that preserves the relevant structure."
        ),
        "triggers": {
            "reduce to", "reduction", "equivalent to", "isomorphic",
            "map to", "transformation", "map onto", "recast as",
            "in disguise", "is just", "same as", "equivalent",
        },
        "examples": [
            "Mathematics: solving a recurrence by mapping to a known "
            "generating function",
            "Physics: showing a problem is equivalent to a harmonic "
            "oscillator",
            "CS: reducing one NP problem to another to prove hardness",
            "Economics: showing a market is equivalent to a known "
            "auction format",
        ],
    },
}


# ============================================================
# Abstract structure fingerprint
# ============================================================

@dataclass
class AbstractFingerprint:
    """The abstract-structure component of a fingerprint."""

    structures: dict[str, float]  # structure_name -> [0, 1]

    def dominant(self, threshold: float = 0.3) -> list[str]:
        return [s for s, v in self.structures.items() if v >= threshold]

    def to_dict(self) -> dict:
        return {"structures": self.structures}


def fingerprint_abstract(text: str) -> AbstractFingerprint:
    """
    Score text against each abstract structure by trigger keyword hits,
    using a saturating function so a few clear hits beat many marginal
    ones.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}
    for name, spec in ABSTRACT_STRUCTURES.items():
        hits = 0
        for term in spec["triggers"]:
            if " " in term or "-" in term:
                hits += text_lower.count(term)
            else:
                hits += len(re.findall(rf"\b{re.escape(term)}\b", text_lower))
        scores[name] = 1.0 - math.exp(-0.5 * hits) if hits > 0 else 0.0
    return AbstractFingerprint(structures=scores)


def abstract_similarity(
    a: AbstractFingerprint, b: AbstractFingerprint,
) -> float:
    """Cosine over abstract-structure vectors."""
    return _cosine(a.structures, b.structures)


# ============================================================
# Cross-domain analogy
# ============================================================

@dataclass
class CrossDomainAnalogy:
    """A technique from one domain proposed as analogous to a problem
    in another domain."""

    technique_id: str
    technique_name: str
    technique_domain: str
    problem_domain: str
    shared_structure: str            # the dominant abstract structure
    structure_strength: float        # how strong the structural match is
    abstract_similarity: float       # full cosine match
    structure_description: str       # human-readable description
    structure_examples: list[str]    # concrete examples in other domains

    def to_dict(self) -> dict:
        return {
            "technique_id":         self.technique_id,
            "technique_name":       self.technique_name,
            "technique_domain":     self.technique_domain,
            "problem_domain":       self.problem_domain,
            "shared_structure":     self.shared_structure,
            "structure_strength":   self.structure_strength,
            "abstract_similarity":  self.abstract_similarity,
            "structure_description": self.structure_description,
            "structure_examples":   self.structure_examples,
        }


# ============================================================
# CrossDomainComposer
# ============================================================

class CrossDomainComposer:
    """
    Wraps a TechniqueComposer with cross-domain analogical matching.

    Uses the underlying composer for within-domain composition (the
    safer default), then adds a separate channel that surfaces
    cross-domain analogies via abstract-structure matching.

    Usage:
        from technique_composer import TechniqueComposer
        from cross_domain_composer import CrossDomainComposer

        base = TechniqueComposer(library, "/path/to/cooccurrence.json")
        cross = CrossDomainComposer(base)

        # Within-domain compositions (as before)
        comps = cross.compose_for_problem(question, "physics_mond")

        # Cross-domain analogies (new)
        analogies = cross.find_cross_domain_analogies(
            question, "physics_mond", top_n=3,
        )
        prompt_fragment = cross.format_for_prompt(comps, analogies)
    """

    def __init__(self, base_composer: TechniqueComposer):
        self.base = base_composer
        self._abstract_cache: dict[str, AbstractFingerprint] = {}

    # -------------------- Abstract fingerprinting --------------------

    def _abstract_for_technique(self, tid: str) -> Optional[AbstractFingerprint]:
        if tid in self._abstract_cache:
            return self._abstract_cache[tid]
        t_dict = self.base.library._data["techniques"].get(tid)
        if t_dict is None:
            return None
        text = " ".join([
            t_dict.get("name", ""),
            t_dict.get("description", ""),
            t_dict.get("when_to_use", ""),
            t_dict.get("example_text", ""),
        ])
        fp = fingerprint_abstract(text)
        self._abstract_cache[tid] = fp
        return fp

    def _abstract_for_problem(
        self, question: str, obligations: Optional[list[dict]] = None,
    ) -> AbstractFingerprint:
        text = question
        if obligations:
            text += " " + " ".join(o.get("text", "") for o in obligations)
        return fingerprint_abstract(text)

    # -------------------- Within-domain (delegated) --------------------

    def compose_for_problem(self, *args, **kwargs):
        return self.base.compose_for_problem(*args, **kwargs)

    # -------------------- Cross-domain analogies --------------------

    def find_cross_domain_analogies(
        self,
        question: str,
        problem_domain: str,
        top_n: int = 3,
        min_structure_strength: float = 0.35,
        min_abstract_similarity: float = 0.4,
        obligations: Optional[list[dict]] = None,
    ) -> list[CrossDomainAnalogy]:
        """
        Find techniques from OTHER domains whose abstract structure
        matches this problem's abstract structure.

        Filtering:
          - Only techniques in domains other than `problem_domain`
          - Both problem and technique must have at least one strong
            abstract structure (>= min_structure_strength)
          - Their abstract fingerprint cosine must be >= min_abstract_similarity
        """
        problem_abstract = self._abstract_for_problem(question, obligations)
        problem_dominant = problem_abstract.dominant(
            threshold=min_structure_strength
        )
        if not problem_dominant:
            # Problem doesn't strongly instantiate any abstract pattern;
            # cross-domain matching is unreliable here.
            return []

        analogies: list[CrossDomainAnalogy] = []
        seen_signatures: set[tuple] = set()

        for tid, t in self.base.library._data["techniques"].items():
            t_domain = t.get("domain", "")
            if t_domain == problem_domain or not t_domain:
                continue

            t_abstract = self._abstract_for_technique(tid)
            if t_abstract is None:
                continue

            t_dominant = t_abstract.dominant(threshold=min_structure_strength)
            if not t_dominant:
                continue

            # Find shared dominant structures
            shared = [s for s in t_dominant if s in problem_dominant]
            if not shared:
                continue

            full_sim = abstract_similarity(problem_abstract, t_abstract)
            if full_sim < min_abstract_similarity:
                continue

            # Pick the strongest shared structure as the headline
            headline = max(
                shared,
                key=lambda s: min(
                    problem_abstract.structures.get(s, 0.0),
                    t_abstract.structures.get(s, 0.0),
                ),
            )
            strength = min(
                problem_abstract.structures.get(headline, 0.0),
                t_abstract.structures.get(headline, 0.0),
            )

            # Dedupe: don't suggest two analogies that share the same
            # (technique, headline) — keep the strongest.
            sig = (tid, headline)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            spec = ABSTRACT_STRUCTURES[headline]
            analogies.append(CrossDomainAnalogy(
                technique_id=tid,
                technique_name=t.get("name", "(unknown)"),
                technique_domain=t_domain,
                problem_domain=problem_domain,
                shared_structure=headline,
                structure_strength=strength,
                abstract_similarity=full_sim,
                structure_description=spec["description"],
                structure_examples=spec["examples"],
            ))

        # Rank by combined signal: the strongest single-structure match
        # plus the overall vector similarity
        analogies.sort(
            key=lambda a: 0.6 * a.structure_strength + 0.4 * a.abstract_similarity,
            reverse=True,
        )
        return analogies[:top_n]

    # -------------------- Combined prompt formatting --------------------

    def format_for_prompt(
        self,
        compositions: Optional[list] = None,
        analogies: Optional[list[CrossDomainAnalogy]] = None,
    ) -> str:
        """
        Format both within-domain compositions and cross-domain analogies
        into a single prompt fragment with clearly distinct framing.

        The framing matters: cross-domain analogies are presented as
        candidates the LLM should evaluate, not techniques to apply
        blindly. The LLM has to do the analogy-validation work.
        """
        parts: list[str] = []

        if compositions:
            parts.append(self.base.format_compositions_for_prompt(compositions))

        if analogies:
            parts.append(self._format_analogies(analogies))

        return "\n\n".join(p for p in parts if p)

    def _format_analogies(self, analogies: list[CrossDomainAnalogy]) -> str:
        lines = [
            "CROSS-DOMAIN ANALOGIES — techniques from other domains whose "
            "ABSTRACT STRUCTURE matches this problem.",
            "",
            "These are CANDIDATE mappings, not direct prescriptions. For each, "
            "consider whether the structural correspondence actually holds in "
            "this problem's context. A strong structural fingerprint is "
            "evidence the analogy is worth examining — not proof it applies.",
        ]
        for i, a in enumerate(analogies, 1):
            lines.append(f"\n  Analogy {i}: '{a.technique_name}' "
                         f"(from {a.technique_domain})")
            lines.append(f"    Shared abstract structure: '{a.shared_structure}' "
                         f"(strength {a.structure_strength:.0%})")
            lines.append(f"    What this structure does: {a.structure_description}")
            lines.append(f"    Examples across domains:")
            for ex in a.structure_examples[:4]:
                lines.append(f"      - {ex}")
            lines.append(
                f"    Evaluate: does the move that worked in "
                f"'{a.technique_domain}' map cleanly onto your "
                f"'{a.problem_domain}' problem? If the structural mapping "
                f"is valid, adapt the move; if not, note why the analogy "
                f"breaks down (that itself is informative)."
            )
        return "\n".join(lines)

    # -------------------- Inspection --------------------

    def summary(self) -> dict:
        return {
            "base_composer":      self.base.summary(),
            "abstract_fp_cached": len(self._abstract_cache),
            "n_abstract_structures": len(ABSTRACT_STRUCTURES),
        }
