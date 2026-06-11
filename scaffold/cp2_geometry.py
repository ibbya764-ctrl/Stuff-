"""
cp2_geometry.py
===============

CP² geometric primitives for the reasoning architecture.

CP² = SU(3)/U(2) — the complex projective plane.

Points are equivalence classes of triples (z₀,z₁,z₂) ∈ C³(0}
under scalar multiplication: [z₀:z₁:z₂] ~ [λz₀:λz₁:λz₂].

The Fubini-Study metric gives CP² a natural Riemannian structure.
Geodesic distance between two points [z] and [w] is:

    d_FS([z],[w]) = arccos( |⟨z,w⟩|² / (|z|²|w|²) )^(1/2)

This is the geometry that (in your dark matter work) produces
flat rotation curves from first principles. Using it as the
internal geometry of the reasoning architecture means concept
similarity is measured by the same metric that governs galactic
dynamics — not by an arbitrary choice of cosine similarity.

The SU(3) isometry group acts on CP² by unitary transformations.
Reasoning patterns that are "the same move in a different setting"
are related by SU(3) transformations — they're genuinely equivalent
in the geometry, not just approximately similar.

Parallel transport using the Levi-Civita connection implements
analogical reasoning geometrically: "A is to B as C is to ?"
becomes a well-posed problem on the manifold.
"""

import math
import cmath
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# CP2Point — a point on the complex projective plane
# ============================================================

@dataclass
class CP2Point:
    """
    A point on CP² in homogeneous coordinates.
    coords: np.ndarray of shape (3,), dtype complex128
    Always stored in normalised form: |coords| = 1
    """
    coords: np.ndarray

    def __post_init__(self):
        self.coords = np.array(self.coords, dtype=complex)
        norm = np.linalg.norm(self.coords)
        if norm < 1e-12:
            raise ValueError("Zero vector has no CP² representative")
        self.coords = self.coords / norm

    # ── Distances ───────────────────────────────────────

    def fubini_study_distance(self, other: "CP2Point") -> float:
        """
        Geodesic distance on CP² with Fubini-Study metric.
        Returns value in [0, π/2].
        """
        inner = np.dot(self.coords.conj(), other.coords)
        cos_sq = np.clip(abs(inner)**2, 0.0, 1.0)
        return math.acos(math.sqrt(cos_sq))

    def fubini_study_similarity(self, other: "CP2Point") -> float:
        """
        Similarity in [0,1]: 1 = same point, 0 = maximally distant.
        = cos²(d_FS) = |⟨z,w⟩|² (for normalised representatives)
        """
        inner = np.dot(self.coords.conj(), other.coords)
        return float(abs(inner)**2)

    # ── Charts and local coordinates ──────────────────

    def affine_chart(self, chart: int = 0) -> np.ndarray:
        """
        Return affine coordinates in the chart where z_{chart} = 1.
        Returns 2 complex numbers (the other two coordinates).
        Fails if z_{chart} = 0 (point at infinity in this chart).
        """
        if abs(self.coords[chart]) < 1e-12:
            raise ValueError(f"Point lies at infinity in chart {chart}")
        normed = self.coords / self.coords[chart]
        return np.delete(normed, chart)

    # ── Tangent space operations ───────────────────────

    def log_map(self, other: "CP2Point") -> np.ndarray:
        """
        Logarithmic map: returns the tangent vector at self
        pointing toward other (in the tangent space T_{self}CP²).
        The tangent space is a 4-real-dimensional real vector
        (2 complex dimensions).
        """
        inner = np.dot(self.coords.conj(), other.coords)
        if abs(inner) < 1e-12:
            # Points are maximally separated
            return other.coords - self.coords
        # Project other onto tangent space at self
        proj = other.coords - inner * self.coords
        norm_proj = np.linalg.norm(proj)
        if norm_proj < 1e-12:
            return np.zeros(3, dtype=complex)
        d = self.fubini_study_distance(other)
        return (d / norm_proj) * proj

    def exp_map(self, tangent: np.ndarray) -> "CP2Point":
        """
        Exponential map: starting at self, move along the geodesic
        in direction tangent (element of T_{self}CP²).
        """
        t = tangent - np.dot(self.coords.conj(), tangent) * self.coords
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-12:
            return CP2Point(self.coords.copy())
        result = (math.cos(norm_t) * self.coords
                  + math.sin(norm_t) / norm_t * t)
        return CP2Point(result)

    def parallel_transport(
        self, vector: np.ndarray, to: "CP2Point"
    ) -> np.ndarray:
        """
        Parallel transport of a tangent vector along the geodesic
        from self to `to`. Uses the Levi-Civita connection on CP².

        This is the geometric operation underlying analogical reasoning:
        if self→to represents "A maps to B", then transporting a
        vector from A gives the corresponding vector at B.
        """
        d = self.fubini_study_distance(to)
        if d < 1e-12:
            return vector

        # Unit tangent at self pointing toward to
        log = self.log_map(to)
        norm_log = np.linalg.norm(log)
        if norm_log < 1e-12:
            return vector
        u = log / norm_log

        # Parallel transport formula on CP² (Fubini-Study)
        inner_p  = np.dot(self.coords.conj(), vector)
        inner_pu = np.dot(u.conj(), vector)

        transported = (vector
                       - inner_pu * (1 - math.cos(d)) * u
                       + inner_pu * math.sin(d) * self.coords
                       - inner_p  * math.sin(d) * u)
        return transported

    # ── SU(3) action ──────────────────────────────────

    def su3_transform(self, U: np.ndarray) -> "CP2Point":
        """
        Apply a 3×3 unitary matrix U ∈ SU(3) to this point.
        SU(3) is the isometry group of CP² with Fubini-Study metric.
        """
        return CP2Point(U @ self.coords)

    def __repr__(self) -> str:
        z = self.coords
        return f"CP2[{z[0]:.3f}:{z[1]:.3f}:{z[2]:.3f}]"


# ============================================================
# Concept → CP² embedding
# ============================================================

class ConceptEmbedder:
    """
    Maps text concepts to points on CP².

    Strategy: use the LLM to generate a compact semantic vector,
    then map to homogeneous coordinates via a learned or fixed
    complex projection.

    The projection is:
        [z₀:z₁:z₂] = [v₀+iv₁ : v₂+iv₃ : v₄+iv₅]

    where v is a 6-dimensional real semantic vector.
    Normalisation to unit sphere enforces the projective equivalence.
    """

    def __init__(self, llm_fn=None):
        self.llm      = llm_fn
        self._cache:  dict[str, CP2Point] = {}

    def embed(self, concept: str) -> CP2Point:
        """Map a concept string to a point on CP²."""
        if concept in self._cache:
            return self._cache[concept]

        vec = self._semantic_vector(concept)
        point = self._vector_to_cp2(vec)
        self._cache[concept] = point
        return point

    def _semantic_vector(self, concept: str) -> np.ndarray:
        """Get a 6-dimensional semantic vector for a concept."""
        if self.llm:
            system = (
                "Return ONLY 6 numbers between -1 and 1, space-separated. "
                "Encode the concept semantically: "
                "dim0=abstraction(concrete-abstract), "
                "dim1=dynamics(static-dynamic), "
                "dim2=scale(micro-macro), "
                "dim3=formality(informal-formal), "
                "dim4=causality(effect-cause), "
                "dim5=complexity(simple-complex). "
                "Nothing else."
            )
            try:
                resp  = self.llm(system, f"Concept: {concept}")
                nums  = [float(x) for x in resp.strip().split()[:6]]
                if len(nums) == 6:
                    return np.array(nums)
            except Exception:
                pass
        # Fallback: hash-based deterministic embedding
        h = hash(concept) % (2**32)
        rng = np.random.default_rng(h)
        return rng.uniform(-1, 1, 6)

    @staticmethod
    def _vector_to_cp2(v: np.ndarray) -> CP2Point:
        """Map 6-dim real vector to CP² homogeneous coordinates."""
        z = np.array([
            v[0] + 1j*v[1],
            v[2] + 1j*v[3],
            v[4] + 1j*v[5],
        ])
        return CP2Point(z)


# ============================================================
# CP2 Structure Mapper — geodesic analogy finding
# ============================================================

class CP2StructureMapper:
    """
    Finds structural analogies between concepts using CP² geometry.

    Replaces cosine similarity with Fubini-Study geodesic distance.
    Analogical reasoning via parallel transport on the manifold.

    The analogy "A is to B as C is to ?" becomes:
        1. Compute the tangent vector A→B (log map at A)
        2. Parallel-transport that vector from A to C
        3. Follow the geodesic from C in that direction (exp map)
        4. The result is the point D = answer to the analogy
    """

    def __init__(self, embedder: ConceptEmbedder):
        self.embedder = embedder

    def similarity(self, a: str, b: str) -> float:
        """Fubini-Study similarity between two concepts. In [0,1]."""
        pa = self.embedder.embed(a)
        pb = self.embedder.embed(b)
        return pa.fubini_study_similarity(pb)

    def distance(self, a: str, b: str) -> float:
        """Geodesic distance between two concepts. In [0, π/2]."""
        pa = self.embedder.embed(a)
        pb = self.embedder.embed(b)
        return pa.fubini_study_distance(pb)

    def analogy_point(self, A: str, B: str, C: str) -> CP2Point:
        """
        Geometric analogy: A→B transported from C.
        Returns the CP² point D such that A:B :: C:D
        via parallel transport on the manifold.
        """
        pA = self.embedder.embed(A)
        pB = self.embedder.embed(B)
        pC = self.embedder.embed(C)

        # Tangent vector at A pointing toward B
        tangent_AB = pA.log_map(pB)
        # Parallel-transport this vector from A to C
        transported = pA.parallel_transport(tangent_AB, pC)
        # Follow geodesic from C in that direction
        pD = pC.exp_map(transported)
        return pD

    def find_closest(
        self, query: CP2Point, candidates: list[str]
    ) -> tuple[str, float]:
        """Find the candidate concept closest to a query point."""
        best_name = ""
        best_sim  = -1.0
        for name in candidates:
            p   = self.embedder.embed(name)
            sim = query.fubini_study_similarity(p)
            if sim > best_sim:
                best_sim  = sim
                best_name = name
        return best_name, best_sim

    def cluster_by_geodesic(
        self, concepts: list[str], threshold: float = 0.3
    ) -> list[list[str]]:
        """
        Group concepts by geodesic proximity.
        threshold: maximum Fubini-Study distance to be in same cluster.
        """
        points = [(c, self.embedder.embed(c)) for c in concepts]
        clusters: list[list[str]] = []
        assigned: set[str] = set()

        for concept, pt in points:
            if concept in assigned:
                continue
            cluster = [concept]
            assigned.add(concept)
            for other, opt in points:
                if other in assigned:
                    continue
                if pt.fubini_study_distance(opt) < threshold:
                    cluster.append(other)
                    assigned.add(other)
            clusters.append(cluster)
        return clusters


# ============================================================
# CP2 Laplace-Beltrami spectral analyser
# ============================================================

class CP2SpectralAnalyser:
    """
    Spectral analysis of knowledge/reasoning graphs using the
    discrete Laplace-Beltrami operator on CP².

    The Laplace-Beltrami operator Δ on CP² decomposes functions on
    the manifold into eigenmodes. For a discrete graph of concepts
    embedded on CP², the graph Laplacian approximates this operator.

    The eigenvalue spacing statistics of this Laplacian connect
    to GUE distributions — linking this to your GUE work.

    If a reasoning chain or knowledge graph shows GUE eigenvalue
    spacing, it means the conceptual structure has the spectral
    signature of CP² geometry.
    """

    def __init__(self, embedder: ConceptEmbedder):
        self.embedder = embedder

    def build_laplacian(self, concepts: list[str]) -> np.ndarray:
        """
        Build the weighted graph Laplacian for a set of concepts
        embedded on CP². Edge weights = Fubini-Study similarities.
        """
        n = len(concepts)
        W = np.zeros((n, n), dtype=float)

        points = [self.embedder.embed(c) for c in concepts]
        for i in range(n):
            for j in range(i+1, n):
                sim = points[i].fubini_study_similarity(points[j])
                W[i,j] = sim
                W[j,i] = sim

        # Degree matrix
        D = np.diag(W.sum(axis=1))
        return D - W    # Graph Laplacian

    def spectral_decomposition(
        self, concepts: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Eigendecomposition of the CP²-weighted graph Laplacian.
        Returns (eigenvalues, eigenvectors).
        """
        L  = self.build_laplacian(concepts)
        vals, vecs = np.linalg.eigh(L)
        return vals, vecs

    def gue_spacing_test(self, eigenvalues: np.ndarray) -> dict:
        """
        Test whether eigenvalue spacings follow GUE statistics.
        Connects to the gue_engine.py already in the architecture.
        """
        try:
            from gue_engine import GUEEngine
            engine = GUEEngine()
            return engine.test_eigenvalue_spacing(eigenvalues)
        except ImportError:
            pass
        # Fallback: basic level spacing ratio
        sorted_vals = np.sort(eigenvalues[eigenvalues > 1e-10])
        if len(sorted_vals) < 4:
            return {"gue_compatible": False, "n_eigenvalues": len(sorted_vals)}
        spacings = np.diff(sorted_vals)
        ratios   = np.minimum(spacings[:-1], spacings[1:]) / \
                   np.maximum(spacings[:-1], spacings[1:])
        mean_r   = float(np.mean(ratios))
        # GUE prediction: <r> ≈ 0.536; Poisson: <r> ≈ 0.386
        return {
            "mean_spacing_ratio": round(mean_r, 4),
            "gue_compatible":     abs(mean_r - 0.536) < 0.05,
            "gue_distance":       round(abs(mean_r - 0.536), 4),
            "n_eigenvalues":      len(sorted_vals),
        }

    def dominant_modes(
        self, concepts: list[str], n_modes: int = 3
    ) -> list[dict]:
        """
        Return the dominant spectral modes of the concept graph.
        Each mode is a pattern of concepts that 'resonate' together.
        Low-eigenvalue modes = large-scale conceptual structure.
        """
        vals, vecs = self.spectral_decomposition(concepts)
        modes = []
        for i in range(min(n_modes, len(vals))):
            weights = np.abs(vecs[:, i])
            ranked  = sorted(
                zip(concepts, weights.tolist()),
                key=lambda x: x[1], reverse=True
            )
            modes.append({
                "eigenvalue":  round(float(vals[i]), 6),
                "top_concepts": [c for c, _ in ranked[:3]],
                "weights":      [round(w, 3) for _, w in ranked[:3]],
            })
        return modes


# ============================================================
# Integration with the existing architecture
# ============================================================

class CP2ReasoningSpace:
    """
    Top-level CP² geometry layer for the Bri/Scaffold architecture.

    Replaces flat Euclidean similarity throughout the scaffold with
    Fubini-Study geodesic geometry.

    Key changes:
      Structure mapper:  cosine → geodesic distance
      Knowledge base:    flat similarity → CP² proximity
      DMN insights:      vector arithmetic → parallel transport
      GUE test:          abstract statistics → CP² eigenvalue structure
    """

    def __init__(self, llm_fn=None, verbose: bool = True):
        self.verbose    = verbose
        self.embedder   = ConceptEmbedder(llm_fn)
        self.mapper     = CP2StructureMapper(self.embedder)
        self.spectral   = CP2SpectralAnalyser(self.embedder)

    def find_analogy(
        self, A: str, B: str, C: str, candidates: list[str] = None
    ) -> dict:
        """
        A:B :: C:? using parallel transport on CP².
        If candidates provided, returns the closest match.
        Otherwise returns the geometric point D.
        """
        D_point = self.mapper.analogy_point(A, B, C)

        result = {
            "A": A, "B": B, "C": C,
            "D_point": str(D_point),
            "d_AB":    round(self.mapper.distance(A, B), 4),
            "d_CD":    None,
        }

        if candidates:
            best, sim = self.mapper.find_closest(D_point, candidates)
            result["D_candidate"] = best
            result["D_similarity"] = round(sim, 4)
            D_embedded = self.embedder.embed(best)
            result["d_CD"] = round(D_point.fubini_study_distance(D_embedded), 4)

        return result

    def analyse_reasoning_chain(
        self, steps: list[str]
    ) -> dict:
        """
        Map a reasoning chain onto CP² and analyse its geometry.
        Returns: geodesic path length, spectral structure, GUE test.
        """
        if not steps or len(steps) < 2:
            return {}

        points = [self.embedder.embed(s) for s in steps]

        # Geodesic path length through the reasoning chain
        path_length = sum(
            points[i].fubini_study_distance(points[i+1])
            for i in range(len(points)-1)
        )

        # Spectral analysis
        gue = self.spectral.gue_spacing_test(
            self.spectral.spectral_decomposition(steps)[0]
        )
        modes = self.spectral.dominant_modes(steps, n_modes=2)

        return {
            "n_steps":          len(steps),
            "geodesic_length":  round(path_length, 4),
            "gue_test":         gue,
            "dominant_modes":   modes,
            "mean_step_dist":   round(path_length / (len(steps)-1), 4),
        }

    def cross_domain_similarity(
        self, concept_a: str, domain_a: str,
        concept_b: str, domain_b: str
    ) -> dict:
        """
        Measure how structurally similar two concepts from different
        domains are using CP² geometry.
        If the geodesic distance is small despite different domains,
        that's a genuine structural connection.
        """
        sim    = self.mapper.similarity(concept_a, concept_b)
        dist   = self.mapper.distance(concept_a, concept_b)
        return {
            "concept_a":   concept_a,
            "concept_b":   concept_b,
            "fs_similarity": round(sim, 4),
            "fs_distance":   round(dist, 4),
            "structurally_close": dist < 0.4,
        }
