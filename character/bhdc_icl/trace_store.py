from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .types import TraceEvent


class JsonlTraceStore:
    """Append-only JSONL trace store.

    This is deliberately boring infrastructure. BHDC v18 needs provenance,
    rework derivatives, renewal survival and audit history before it can make
    claims about character; this store preserves those measurements.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: TraceEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.asdict(), ensure_ascii=False) + "\n")

    def iter_events(self, limit: Optional[int] = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        count = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                    count += 1
                    if limit is not None and count >= limit:
                        break

    def tail(self, n: int = 10) -> list[dict]:
        rows = list(self.iter_events())
        return rows[-n:]
