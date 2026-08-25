from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    source: str
    text: str
    chunk_index: int
    page: int | None = None
    overlap_from_previous: bool = False
    overlap_prefix: str = ""

    @property
    def citation(self) -> str:
        location = f"p.{self.page}" if self.page is not None else f"chunk {self.chunk_index}"
        return f"{self.source} · {location}"


@dataclass(frozen=True, slots=True)
class Claim:
    subject: str
    relation: str
    object: str
    source: str
    chunk_index: int
    page: int | None = None
    evidence: str = ""
    confidence: float | None = None
    synthetic: bool = False
    overlap_from_previous: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["confidence"] is not None:
            data["confidence"] = round(float(data["confidence"]), 4)
        return data

    @property
    def citation(self) -> str:
        location = f"p.{self.page}" if self.page is not None else f"chunk {self.chunk_index}"
        return f"{self.source} · {location}"
