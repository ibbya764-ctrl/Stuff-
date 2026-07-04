from __future__ import annotations

"""Data pipeline for BHDCMoralLM: a byte-level capability corpus + the curated
moral corpus.

Deliberately dependency-light (no HF datasets/tokenizers): byte-level tokenizer
(0-255 + reserved specials) so any UTF-8 text is trainable, a capability corpus
that downloads+caches public-domain text (with an offline fallback so it always
runs), and a loader for the committed moral JSONL.
"""

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

BYTE_VOCAB = 256
PAD, BOS, EOS = 256, 257, 258        # reserved specials above the byte range
DEFAULT_VOCAB = 320                  # headroom; ids 259-319 unused

CACHE_DIR = Path("harness/data/corpus")
CAPABILITY_SOURCES = [
    # public-domain / permissively-hosted plain text
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
]
_OFFLINE_FALLBACK = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles.\n"
) * 400


class ByteTokenizer:
    vocab_size = DEFAULT_VOCAB

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8", errors="ignore"))

    def decode(self, ids: List[int]) -> str:
        return bytes(b for b in ids if b < BYTE_VOCAB).decode("utf-8", errors="ignore")


@dataclass
class CapabilityCorpus:
    """Byte-level LM corpus with a train/val split and random-window batching."""
    train: torch.Tensor
    val: torch.Tensor
    tokenizer: ByteTokenizer

    @classmethod
    def load(cls, sources: Optional[List[str]] = None, val_frac: float = 0.1,
             max_bytes: Optional[int] = None) -> "CapabilityCorpus":
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        text_parts: List[str] = []
        for url in (sources or CAPABILITY_SOURCES):
            fn = CACHE_DIR / (url.rsplit("/", 1)[-1] or "corpus.txt")
            if not fn.exists():
                try:
                    urllib.request.urlretrieve(url, fn)
                except Exception:
                    continue
            if fn.exists():
                text_parts.append(fn.read_text(encoding="utf-8", errors="ignore"))
        text = "\n".join(text_parts) if text_parts else _OFFLINE_FALLBACK
        if max_bytes:
            text = text[:max_bytes]
        tok = ByteTokenizer()
        ids = torch.tensor(tok.encode(text), dtype=torch.long)
        n_val = max(1, int(len(ids) * val_frac))
        return cls(train=ids[:-n_val], val=ids[-n_val:], tokenizer=tok)

    def get_batch(self, split: str, batch: int, seqlen: int,
                  device: str = "cpu", generator=None) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train if split == "train" else self.val
        hi = len(data) - seqlen - 1
        ix = torch.randint(0, max(1, hi), (batch,), generator=generator)
        x = torch.stack([data[i:i + seqlen] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + seqlen] for i in ix])
        return x.to(device), y.to(device)


@dataclass
class MoralExample:
    prompt: str
    response: str
    labels: Dict[str, float]
    category: str
    split: str

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.response}"


class MoralCorpus:
    """The curated human-anchored conscience corpus (data/moral/moral_corpus.jsonl)."""

    def __init__(self, path: str = "data/moral/moral_corpus.jsonl"):
        self.examples: List[MoralExample] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self.examples.append(MoralExample(
                prompt=d["prompt"], response=d["response"], labels=d["labels"],
                category=d["category"], split=d.get("split", "train"),
            ))

    def split(self, which: str) -> List[MoralExample]:
        return [e for e in self.examples if e.split == which]

    def categories(self) -> List[str]:
        return sorted({e.category for e in self.examples})
