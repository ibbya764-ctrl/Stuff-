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

# The capability corpus, by design rather than accident. All public-domain
# (Project Gutenberg) or MIT-licensed (code). Categories are deliberate:
#   narrative     -- long-range prose structure, dialogue, plot, register;
#   reasoning     -- empirical argument and deduction in natural language;
#   philosophy    -- sustained abstract argumentation (incl. ethics vocabulary,
#                    as CAPABILITY text -- the moral corpus is separate and
#                    human-labelled; these books just teach the language);
#   civic         -- expository/legal argument;
#   satire        -- irony (a hard case: surface-harmful, intent-benign);
#   code          -- formal syntax diversity for the byte model.
_GB = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
CAPABILITY_MANIFEST = [
    # (name, category, url)
    ("shakespeare_complete", "narrative",  _GB.format(id=100)),
    ("pride_and_prejudice",  "narrative",  _GB.format(id=1342)),
    ("moby_dick",            "narrative",  _GB.format(id=2701)),
    ("sherlock_holmes",      "reasoning",  _GB.format(id=1661)),
    ("huckleberry_finn",     "narrative",  _GB.format(id=76)),
    ("grimms_fairy_tales",   "narrative",  _GB.format(id=2591)),
    ("alice_in_wonderland",  "narrative",  _GB.format(id=11)),
    ("war_and_peace",        "narrative",  _GB.format(id=2600)),
    ("origin_of_species",    "reasoning",  _GB.format(id=1228)),
    ("einstein_relativity",  "reasoning",  _GB.format(id=30155)),
    ("faraday_candle",       "reasoning",  _GB.format(id=14474)),
    ("plato_republic",       "philosophy", _GB.format(id=1497)),
    ("marcus_meditations",   "philosophy", _GB.format(id=2680)),
    ("nicomachean_ethics",   "philosophy", _GB.format(id=8438)),
    ("mill_on_liberty",      "philosophy", _GB.format(id=34901)),
    ("federalist_papers",    "civic",      _GB.format(id=1404)),
    ("modest_proposal",      "satire",     _GB.format(id=1080)),
    ("mingpt_model_py",      "code",
     "https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/model.py"),
    ("nanogpt_train_py",     "code",
     "https://raw.githubusercontent.com/karpathy/nanoGPT/master/train.py"),
]
CAPABILITY_SOURCES = [u for _, _, u in CAPABILITY_MANIFEST]  # back-compat

_OFFLINE_FALLBACK = (
    "To be, or not to be, that is the question:\n"
    "Whether 'tis nobler in the mind to suffer\n"
    "The slings and arrows of outrageous fortune,\n"
    "Or to take arms against a sea of troubles.\n"
) * 400


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Cut the Project Gutenberg header/footer so the model trains on the book,
    not on licence boilerplate repeated 17 times."""
    start = text.find("*** START OF")
    if start != -1:
        nl = text.find("\n", start)
        text = text[nl + 1:] if nl != -1 else text
    end = text.find("*** END OF")
    if end != -1:
        text = text[:end]
    return text.strip()


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

    composition: Optional[Dict[str, int]] = None   # bytes per source (report)

    @classmethod
    def load(cls, sources: Optional[List[str]] = None, val_frac: float = 0.1,
             max_bytes: Optional[int] = None) -> "CapabilityCorpus":
        """Download+cache each source, strip Gutenberg boilerplate, and split
        train/val PER SOURCE (each source's tail goes to val) so validation
        covers every category instead of only whichever file was concatenated
        last. ``max_bytes`` caps each split proportionally for quick runs."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        manifest = ([(u.rsplit("/", 1)[-1], "unknown", u) for u in sources]
                    if sources is not None else CAPABILITY_MANIFEST)
        train_parts: List[torch.Tensor] = []
        val_parts: List[torch.Tensor] = []
        composition: Dict[str, int] = {}
        tok = ByteTokenizer()
        for name, _cat, url in manifest:
            fn = CACHE_DIR / f"{name}.txt"
            if not fn.exists():
                try:
                    urllib.request.urlretrieve(url, fn)
                except Exception:
                    continue
            if not fn.exists():
                continue
            text = fn.read_text(encoding="utf-8", errors="ignore")
            if "gutenberg" in url:
                text = _strip_gutenberg_boilerplate(text)
            ids = torch.tensor(tok.encode(text), dtype=torch.long)
            if len(ids) < 64:
                continue
            composition[name] = len(ids)
            n_val = max(1, int(len(ids) * val_frac))
            train_parts.append(ids[:-n_val])
            val_parts.append(ids[-n_val:])
        if not train_parts:
            ids = torch.tensor(tok.encode(_OFFLINE_FALLBACK), dtype=torch.long)
            n_val = max(1, int(len(ids) * val_frac))
            train_parts, val_parts = [ids[:-n_val]], [ids[-n_val:]]
            composition = {"offline_fallback": len(ids)}
        train = torch.cat(train_parts)
        val = torch.cat(val_parts)
        if max_bytes:
            train = train[:max_bytes]
            val = val[:max(64, int(max_bytes * val_frac))]
        return cls(train=train, val=val, tokenizer=tok, composition=composition)

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
