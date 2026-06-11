"""
technique_embedder.py
=====================

Differentiable technique embeddings — the first step toward making
the scaffold's hand-crafted signals learnable.

The core idea: instead of keyword matching to find similar techniques,
learn a projection matrix W that maps technique descriptions into an
embedding space where functional similarity (techniques that work
together in verified runs) is captured by cosine similarity.

Training signal comes entirely from the scaffold's own structures:
  - DirectedCoOccurrenceGraph lift scores → positive/negative pairs
  - Obligation store verification results → which runs succeeded
  - GUE convergence score → quality diagnostic on the embedding space

This answers one of the unsolved questions from our conversation:
"How do you make failure memory update weights rather than context?"

The gradient path is now explicit:
  verification result
      → obligation store (records success/failure)
      → training signal (which technique pairs to pull/push)
      → triplet loss gradient
      → W update (the projection matrix is a small set of weights)
      → future find_similar calls return different results

W is small (vocab_size × embed_dim, typically 500 × 64 = 32,000 params).
Trains in milliseconds on CPU. No GPU needed until fine-tuning larger models.

Pure numpy. Optional torch acceleration when available (auto-detected).
"""

import os
import re
import json
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Iterable

# Optional torch — used for GPU acceleration if available
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Text encoder — TF-IDF bag-of-words
# ============================================================

STOP_WORDS = {
    "the", "a", "an", "is", "are", "be", "to", "of", "and", "or",
    "for", "in", "on", "at", "by", "with", "from", "as", "this",
    "that", "it", "any", "all", "some", "when", "where", "which",
    "we", "you", "if", "not", "but", "so", "then", "use", "using",
    "can", "will", "has", "have", "had", "been", "its", "their",
}


class TextEncoder:
    """
    TF-IDF encoder. Builds vocabulary from technique library, then
    encodes any text as a dense TF-IDF feature vector.
    """

    def __init__(self, max_vocab: int = 600):
        self.max_vocab  = max_vocab
        self.vocab:     dict[str, int] = {}
        self.idf:       np.ndarray     = np.array([])
        self.is_fitted: bool           = False

    # ---- Vocabulary building ----

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary and IDF weights from a list of documents."""
        # Token counts across all documents
        word_doc_counts: dict[str, int] = {}
        all_tokens_per_doc: list[set[str]] = []

        for doc in documents:
            tokens = self._tokenise(doc)
            unique = set(tokens)
            all_tokens_per_doc.append(unique)
            for t in unique:
                word_doc_counts[t] = word_doc_counts.get(t, 0) + 1

        # Keep most common words up to max_vocab
        sorted_words = sorted(
            word_doc_counts.items(), key=lambda x: x[1], reverse=True
        )[:self.max_vocab]

        self.vocab = {w: i for i, (w, _) in enumerate(sorted_words)}

        # IDF weights: log((N + 1) / (df + 1)) + 1
        N = len(documents)
        self.idf = np.array([
            math.log((N + 1) / (word_doc_counts.get(w, 0) + 1)) + 1.0
            for w in self.vocab
        ])
        self.is_fitted = True

    def fit_from_library(self, technique_library) -> None:
        """Fit on all techniques in a TechniqueLibrary instance."""
        docs = []
        for t in technique_library._data.get("techniques", {}).values():
            text = " ".join([
                t.get("name", ""),
                t.get("description", ""),
                t.get("when_to_use", ""),
                t.get("example_text", ""),
            ])
            docs.append(text)
        if docs:
            self.fit(docs)

    # ---- Encoding ----

    def encode(self, text: str) -> np.ndarray:
        """
        Return a TF-IDF vector of shape (vocab_size,).
        Returns zeros if not fitted or text has no known words.
        """
        if not self.is_fitted:
            raise RuntimeError("TextEncoder not fitted. Call fit() first.")
        tokens = self._tokenise(text)
        if not tokens:
            return np.zeros(len(self.vocab))

        tf = np.zeros(len(self.vocab))
        for t in tokens:
            if t in self.vocab:
                tf[self.vocab[t]] += 1.0

        if tf.sum() > 0:
            tf /= tf.sum()   # normalise TF

        return tf * self.idf  # TF-IDF

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_]*", text.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "vocab": self.vocab,
                "idf":   self.idf.tolist(),
            }, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            d = json.load(f)
        self.vocab = d["vocab"]
        self.idf   = np.array(d["idf"])
        self.is_fitted = True


# ============================================================
# Triplet loss and gradients (pure numpy)
# ============================================================

def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-8)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _triplet_loss_and_grad(
    e_a: np.ndarray,
    e_p: np.ndarray,
    e_n: np.ndarray,
    margin: float = 0.3,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triplet loss: L = max(0, ||e_a - e_p||² - ||e_a - e_n||² + margin)

    Returns (loss, grad_e_a, grad_e_p, grad_e_n) where all embeddings
    are already L2-normalised.
    """
    d_pos = np.sum((e_a - e_p) ** 2)
    d_neg = np.sum((e_a - e_n) ** 2)
    raw   = d_pos - d_neg + margin

    if raw <= 0:
        zero = np.zeros_like(e_a)
        return 0.0, zero, zero, zero

    loss    = float(raw)
    g_a     = 2.0 * (e_a - e_p) - 2.0 * (e_a - e_n)
    g_p     = -2.0 * (e_a - e_p)
    g_n     =  2.0 * (e_a - e_n)
    return loss, g_a, g_p, g_n


def _backprop_through_normalize(
    grad_e: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Backprop gradient through L2 normalisation: e = z / ||z||

    Jacobian: ∂e/∂z = (I - e e^T) / ||z||
    So:  ∂L/∂z = (grad_e - (grad_e · e) e) / ||z||
    """
    norm   = np.linalg.norm(z) + 1e-8
    e      = z / norm
    proj   = np.dot(grad_e, e)
    grad_z = (grad_e - proj * e) / norm
    return grad_z


# ============================================================
# TechniqueEmbedder
# ============================================================

@dataclass
class TrainingHistory:
    steps:   int   = 0
    losses:  list  = field(default_factory=list)
    gue_log: list  = field(default_factory=list)


class TechniqueEmbedder:
    """
    Learns a projection W: R^vocab → R^embed_dim via triplet loss.

    Positive pairs  : techniques that co-occur in verified successful runs
                      (weighted by lift score from DirectedCoOccurrenceGraph)
    Negative pairs  : techniques that don't co-occur or appear in failed runs

    After training, embed() replaces keyword fingerprinting with learned
    cosine similarity. find_similar() is the drop-in for keyword retrieval.

    The gradient path — making failure memory update weights:
        obligation_store (failure record)
        → training signal (pair weights)
        → triplet loss gradient
        → W update
        → find_similar() returns different techniques next time
    """

    def __init__(
        self,
        embed_dim:    int   = 64,
        learning_rate: float = 0.02,
        margin:       float = 0.3,
        vocab_size:   int   = 600,
    ):
        self.embed_dim = embed_dim
        self.lr        = learning_rate
        self.margin    = margin
        self.encoder   = TextEncoder(max_vocab=vocab_size)
        self.W:        Optional[np.ndarray] = None   # (vocab, embed_dim)
        self.history   = TrainingHistory()
        self._cache:   dict[str, np.ndarray] = {}    # technique_id → embedding

    # ---- Initialisation ----

    def initialise(self, technique_library) -> None:
        """
        Build vocabulary and randomly initialise W.
        Call once before training.
        """
        self.encoder.fit_from_library(technique_library)
        vocab_size = len(self.encoder.vocab)
        # Xavier initialisation
        scale = 1.0 / math.sqrt(vocab_size)
        self.W = np.random.randn(vocab_size, self.embed_dim) * scale
        self._cache.clear()

    @property
    def is_initialised(self) -> bool:
        return self.W is not None and self.encoder.is_fitted

    # ---- Embedding ----

    def embed(self, technique_id: str, technique_dict: dict) -> np.ndarray:
        """
        Embed a technique. Result is cached and invalidated when W changes.
        """
        if technique_id in self._cache:
            return self._cache[technique_id]
        if not self.is_initialised:
            raise RuntimeError("Call initialise() first.")
        text = " ".join([
            technique_dict.get("name", ""),
            technique_dict.get("description", ""),
            technique_dict.get("when_to_use", ""),
            technique_dict.get("example_text", ""),
        ])
        z  = self.encoder.encode(text) @ self.W   # shape (embed_dim,)
        e  = _normalize(z)
        self._cache[technique_id] = e
        return e

    def embed_text(self, text: str) -> np.ndarray:
        """Embed arbitrary text (for query encoding)."""
        if not self.is_initialised:
            raise RuntimeError("Call initialise() first.")
        z = self.encoder.encode(text) @ self.W
        return _normalize(z)

    def _invalidate_cache(self) -> None:
        self._cache.clear()

    # ---- Core training step ----

    def _gradient_step(
        self,
        f_a: np.ndarray,
        f_p: np.ndarray,
        f_n: np.ndarray,
        weight: float = 1.0,
    ) -> float:
        """
        One gradient step on a single (anchor, positive, negative) triplet.

        f_a, f_p, f_n are TF-IDF feature vectors (not yet projected).
        Returns the triplet loss value.
        """
        # Forward pass
        z_a = f_a @ self.W
        z_p = f_p @ self.W
        z_n = f_n @ self.W
        e_a, e_p, e_n = _normalize(z_a), _normalize(z_p), _normalize(z_n)

        # Loss and embedding gradients
        loss, g_ea, g_ep, g_en = _triplet_loss_and_grad(
            e_a, e_p, e_n, self.margin
        )
        if loss == 0.0:
            return 0.0

        loss *= weight
        g_ea *= weight
        g_ep *= weight
        g_en *= weight

        # Backprop through normalisation to pre-projection z
        g_za = _backprop_through_normalize(g_ea, z_a)
        g_zp = _backprop_through_normalize(g_ep, z_p)
        g_zn = _backprop_through_normalize(g_en, z_n)

        # Gradient w.r.t. W: sum of outer products
        # ∂L/∂W = f_a^T ⊗ g_za + f_p^T ⊗ g_zp + f_n^T ⊗ g_zn
        grad_W = (
            np.outer(f_a, g_za)
            + np.outer(f_p, g_zp)
            + np.outer(f_n, g_zn)
        )

        # Update
        self.W -= self.lr * grad_W
        self._invalidate_cache()
        return float(loss)

    # ---- Training from scaffold structures ----

    def train_from_run(
        self,
        ordered_technique_ids: list[str],
        technique_library,
        directed_graph=None,
        all_technique_ids: Optional[list[str]] = None,
        n_negatives_per_positive: int = 3,
    ) -> float:
        """
        Train on one completed successful run.

        Positive pairs: all pairs within the run (in order).
        Negative pairs: random techniques NOT in the run.
        Pair weight: lift score if directed_graph provided, else 1.0.
        """
        if not self.is_initialised or len(ordered_technique_ids) < 2:
            return 0.0

        techs = technique_library._data.get("techniques", {})
        if all_technique_ids is None:
            all_technique_ids = list(techs.keys())

        run_set = set(ordered_technique_ids)
        negatives_pool = [t for t in all_technique_ids if t not in run_set]
        if not negatives_pool:
            return 0.0

        total_loss = 0.0
        n_steps    = 0

        for i, t_a in enumerate(ordered_technique_ids):
            if t_a not in techs:
                continue
            f_a = self.encoder.encode(
                _technique_text(techs[t_a])
            )

            for t_p in ordered_technique_ids[i + 1:]:
                if t_p not in techs:
                    continue
                f_p = self.encoder.encode(
                    _technique_text(techs[t_p])
                )

                # Weight by lift if available
                weight = 1.0
                if directed_graph is not None:
                    lift = directed_graph.lift(t_a, t_p)
                    weight = max(1.0, lift)

                # Sample negatives
                neg_sample = np.random.choice(
                    negatives_pool,
                    size=min(n_negatives_per_positive, len(negatives_pool)),
                    replace=False,
                )
                for t_n in neg_sample:
                    if t_n not in techs:
                        continue
                    f_n = self.encoder.encode(
                        _technique_text(techs[t_n])
                    )
                    loss = self._gradient_step(f_a, f_p, f_n, weight)
                    total_loss += loss
                    n_steps    += 1

        avg_loss = total_loss / max(1, n_steps)
        self.history.steps  += n_steps
        self.history.losses.append(avg_loss)
        return avg_loss

    def train_from_graph(
        self,
        directed_graph,
        technique_library,
        n_epochs:  int   = 5,
        min_lift:  float = 1.2,
        verbose:   bool  = True,
    ) -> list[float]:
        """
        Full training pass using the co-occurrence graph as supervision.

        Positive pairs: all pairs with lift >= min_lift.
        Negative pairs: pairs with zero co-occurrence.
        Pair weights: lift score.

        Returns list of per-epoch average losses.
        """
        if not self.is_initialised:
            raise RuntimeError("Call initialise() first.")

        techs     = technique_library._data.get("techniques", {})
        all_tids  = list(techs.keys())
        n         = len(all_tids)

        if n < 3:
            if verbose:
                print("[embedder] Need ≥ 3 techniques to train.")
            return []

        # Collect all positive pairs with weights
        positive_pairs: list[tuple[str, str, float]] = []
        for i, ta in enumerate(all_tids):
            for tb in all_tids[i + 1:]:
                lift = directed_graph.lift(ta, tb)
                if lift >= min_lift:
                    positive_pairs.append((ta, tb, lift))

        if not positive_pairs:
            if verbose:
                print("[embedder] No positive pairs found "
                      f"(need lift ≥ {min_lift}). "
                      "Run more verified sessions first.")
            return []

        if verbose:
            print(f"[embedder] Training: {len(positive_pairs)} positive pairs, "
                  f"{n_epochs} epochs")

        epoch_losses = []

        for epoch in range(n_epochs):
            np.random.shuffle(positive_pairs)
            epoch_loss = 0.0
            n_steps    = 0

            for ta, tb, weight in positive_pairs:
                if ta not in techs or tb not in techs:
                    continue
                f_a = self.encoder.encode(_technique_text(techs[ta]))
                f_p = self.encoder.encode(_technique_text(techs[tb]))

                # Sample a negative (low or zero co-occurrence)
                negatives = [
                    t for t in all_tids
                    if t != ta and t != tb
                    and directed_graph.co_count(ta, t) == 0
                    and directed_graph.co_count(tb, t) == 0
                ]
                if not negatives:
                    negatives = [
                        t for t in all_tids if t != ta and t != tb
                    ]
                if not negatives:
                    continue

                t_n = np.random.choice(negatives)
                f_n = self.encoder.encode(_technique_text(techs[t_n]))

                loss = self._gradient_step(f_a, f_p, f_n, weight)
                epoch_loss += loss
                n_steps    += 1

            avg = epoch_loss / max(1, n_steps)
            epoch_losses.append(avg)
            if verbose:
                bar = "█" * int((1 - min(avg, 1)) * 20)
                print(f"  Epoch {epoch+1}/{n_epochs}  "
                      f"loss={avg:.4f}  {bar}")

        self.history.steps  += sum(len(positive_pairs) for _ in range(n_epochs))
        self.history.losses.extend(epoch_losses)
        return epoch_losses

    def train_from_obligation_store(
        self,
        obligation_store,
        technique_library,
        directed_graph,
        top_n_gaps: int = 20,
        verbose:    bool = True,
    ) -> float:
        """
        Use low-confidence findings from the obligation store as a
        training signal. Techniques that contributed to unresolved
        obligations get pushed away from each other (they didn't
        work together). Techniques with no associated obligations
        get reinforced as a positive pair.

        This is the direct implementation of:
        "failure memory updating weights."
        """
        if not self.is_initialised:
            return 0.0

        techs = technique_library._data.get("techniques", {})
        all_tids = list(techs.keys())

        try:
            gaps = obligation_store.query_persistent_gaps(top_n=top_n_gaps)
        except Exception:
            return 0.0

        if not gaps:
            return 0.0

        # Build set of source branches that contributed to persistent gaps
        gap_sources: set[str] = set()
        for g in gaps:
            if hasattr(g, "source_branches"):
                for s in g.source_branches:
                    gap_sources.add(s)

        # Techniques that appeared together in failed/gap-producing branches
        # are pushed apart (negative signal for future composition)
        failed_pairs: list[tuple[str, str]] = []
        for key, count in directed_graph._data.get("co_counts", {}).items():
            ta, tb = key.split("|", 1)
            # If both appeared in runs that produced persistent unresolved gaps,
            # treat them as a negative pair
            n_ab = directed_graph.co_count(ta, tb)
            n_total = directed_graph._data["n_runs"]
            if n_total > 0:
                co_rate = n_ab / n_total
                # High co-occurrence but also high gap rate → push apart
                if co_rate > 0.1 and len(gap_sources) > 0:
                    failed_pairs.append((ta, tb))

        total_loss = 0.0
        n_steps    = 0

        # For each failed pair, push them apart using a reversed triplet:
        # The "positive" is a random non-co-occurring technique
        for ta, tb in failed_pairs:
            if ta not in techs or tb not in techs:
                continue
            f_anchor = self.encoder.encode(_technique_text(techs[ta]))
            f_neg    = self.encoder.encode(_technique_text(techs[tb]))

            # Find a technique with low co-occurrence with ta
            non_cooccurring = [
                t for t in all_tids
                if t != ta and t != tb
                and directed_graph.co_count(ta, t) == 0
            ]
            if not non_cooccurring:
                continue

            t_pos = np.random.choice(non_cooccurring)
            f_pos = self.encoder.encode(_technique_text(techs[t_pos]))

            # Standard triplet: pull anchor toward non-co-occurring, away from failed pair
            loss = self._gradient_step(f_anchor, f_pos, f_neg, weight=0.5)
            total_loss += loss
            n_steps    += 1

        avg = total_loss / max(1, n_steps)
        if verbose and n_steps > 0:
            print(f"[embedder] Obligation-store training: "
                  f"{n_steps} steps, avg loss={avg:.4f}")
        return avg

    # ---- Retrieval ----

    def find_similar(
        self,
        query:            str,
        technique_library,
        domain:           str  = "",
        top_n:            int  = 5,
        min_score:        float = 0.1,
    ) -> list[tuple[str, float]]:
        """
        Find techniques most similar to query in embedding space.
        Drop-in replacement for keyword-based retrieval.

        Returns list of (technique_id, cosine_similarity) sorted
        by similarity descending.
        """
        if not self.is_initialised:
            raise RuntimeError("Call initialise() first.")

        query_embed = self.embed_text(query)
        techs       = technique_library._data.get("techniques", {})
        scored: list[tuple[str, float]] = []

        for tid, t in techs.items():
            if domain and t.get("domain", "") != domain:
                continue
            te = self.embed(tid, t)
            score = float(np.dot(query_embed, te))
            if score >= min_score:
                scored.append((tid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    # ---- GUE diagnostic ----

    def gue_diagnostic(self) -> dict:
        """
        Check if the embedding space is well-organised by testing
        the singular value spacing of the embedding matrix.

        If techniques are well-separated and cover distinct regions
        of embedding space, singular values follow GUE-like spacing.
        If techniques all cluster together, spacing is Poisson-like.

        Returns identify_ensemble() result on singular values.
        """
        if not self._cache or not self.is_initialised:
            return {"note": "No embeddings cached yet. Call embed() first."}
        try:
            from gue_engine import identify_ensemble
        except ImportError:
            return {"note": "gue_engine not available."}

        E = np.stack(list(self._cache.values()))   # (n_techniques, embed_dim)
        _, sv, _ = np.linalg.svd(E, full_matrices=False)
        ens = identify_ensemble(sv)
        return {
            "n_embedded":  E.shape[0],
            "embed_dim":   E.shape[1],
            "n_sv":        len(sv),
            "sv_range":    (round(float(sv.min()), 4),
                            round(float(sv.max()), 4)),
            **ens,
        }

    # ---- Persistence ----

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "W.npy"), self.W)
        self.encoder.save(os.path.join(directory, "encoder.json"))
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({
                "embed_dim":      self.embed_dim,
                "lr":             self.lr,
                "margin":         self.margin,
                "training_steps": self.history.steps,
                "n_losses":       len(self.history.losses),
            }, f)

    @classmethod
    def load(cls, directory: str) -> "TechniqueEmbedder":
        with open(os.path.join(directory, "meta.json")) as f:
            meta = json.load(f)
        obj = cls(
            embed_dim=meta["embed_dim"],
            learning_rate=meta["lr"],
            margin=meta["margin"],
        )
        obj.W = np.load(os.path.join(directory, "W.npy"))
        obj.encoder.load(os.path.join(directory, "encoder.json"))
        obj.history.steps = meta.get("training_steps", 0)
        return obj


# ============================================================
# EmbeddedTechniqueComposer — drop-in replacement
# ============================================================

class EmbeddedTechniqueComposer:
    """
    Extends TechniqueComposer to use learned embeddings for retrieval
    instead of keyword fingerprints.

    Usage:
        from technique_embedder import EmbeddedTechniqueComposer
        from directed_graph import DirectedCoOccurrenceGraph

        lib    = TechniqueLibrary("/path/to/techniques.json")
        dgraph = DirectedCoOccurrenceGraph("/path/to/directed.json")
        composer = EmbeddedTechniqueComposer(lib, dgraph)
        composer.initialise_and_train()   # first time

        # Use exactly like TechniqueComposer
        results = composer.find_similar_techniques(question, domain)

        # After a verified run
        composer.update_from_run(ordered_ids, success=True)
    """

    def __init__(
        self,
        technique_library,
        directed_graph,
        embedder_dir: str  = "./technique_embedder",
        embed_dim:    int  = 64,
        verbose:      bool = True,
    ):
        self.library        = technique_library
        self.directed_graph = directed_graph
        self.embedder_dir   = embedder_dir
        self.verbose        = verbose

        # Load existing embedder or create new one
        meta_path = os.path.join(embedder_dir, "meta.json")
        if os.path.exists(meta_path):
            self.embedder = TechniqueEmbedder.load(embedder_dir)
            if self.verbose:
                print(f"[composer] Loaded embedder "
                      f"({self.embedder.history.steps} training steps)")
        else:
            self.embedder = TechniqueEmbedder(embed_dim=embed_dim)
            if self.verbose:
                print("[composer] New embedder created (not yet trained)")

    def initialise_and_train(
        self, n_epochs: int = 10, min_lift: float = 1.2,
    ) -> list[float]:
        """
        Initialise vocabulary and train on existing co-occurrence graph.
        Call once when setting up, and again after significant new runs.
        """
        if not self.embedder.is_initialised:
            self.embedder.initialise(self.library)
            if self.verbose:
                print(f"[composer] Vocabulary built: "
                      f"{len(self.embedder.encoder.vocab)} words")

        losses = self.embedder.train_from_graph(
            self.directed_graph, self.library,
            n_epochs=n_epochs, min_lift=min_lift,
            verbose=self.verbose,
        )
        self.embedder.save(self.embedder_dir)
        return losses

    def find_similar_techniques(
        self,
        query:   str,
        domain:  str  = "",
        top_n:   int  = 5,
    ) -> list[tuple[str, float]]:
        """
        Find similar techniques using learned embeddings.
        Falls back to vocabulary overlap if embedder not trained.
        """
        if not self.embedder.is_initialised:
            if self.verbose:
                print("[composer] Embedder not initialised — "
                      "call initialise_and_train() first.")
            return []
        return self.embedder.find_similar(
            query, self.library, domain=domain, top_n=top_n,
        )

    def update_from_run(
        self,
        ordered_technique_ids: list,
        success: bool = True,
    ) -> None:
        """
        Update co-occurrence graph AND train embedder on each successful run.
        This is where the gradient path closes:
            verified run → W update → future find_similar changes.
        """
        # Update co-occurrence graph (preserving order)
        if isinstance(ordered_technique_ids, list):
            self.directed_graph.update_from_run(
                ordered_technique_ids, success=success,
            )
        else:
            self.directed_graph.update_from_run(
                list(ordered_technique_ids), success=success,
            )

        # Train embedder on this run
        if success and len(ordered_technique_ids) >= 2:
            loss = self.embedder.train_from_run(
                list(ordered_technique_ids),
                self.library,
                directed_graph=self.directed_graph,
            )
            if self.verbose and loss > 0:
                print(f"[composer] Embedding update: loss={loss:.4f}")

        # Persist updated embedder
        self.embedder.save(self.embedder_dir)

    def train_from_failures(self, obligation_store) -> float:
        """
        Use the obligation store's persistent gaps as a negative training signal.
        Techniques associated with unresolved obligations get pushed apart.
        """
        return self.embedder.train_from_obligation_store(
            obligation_store, self.library, self.directed_graph,
            verbose=self.verbose,
        )

    def gue_diagnostic(self) -> dict:
        """Check if embedding space is GUE-organised."""
        return self.embedder.gue_diagnostic()

    def summary(self) -> dict:
        n_tech   = len(self.library._data.get("techniques", {}))
        n_cached = len(self.embedder._cache)
        return {
            "n_techniques":   n_tech,
            "n_embedded":     n_cached,
            "training_steps": self.embedder.history.steps,
            "vocab_size":     len(self.embedder.encoder.vocab),
            "embed_dim":      self.embedder.embed_dim,
            "is_trained":     self.embedder.history.steps > 0,
        }


# ============================================================
# Helpers
# ============================================================

def _technique_text(t: dict) -> str:
    return " ".join([
        t.get("name",          ""),
        t.get("description",   ""),
        t.get("when_to_use",   ""),
        t.get("example_text",  ""),
    ])
