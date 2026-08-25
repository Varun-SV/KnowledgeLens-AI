from __future__ import annotations

from typing import Any

MAX_ENTITY_LABEL_CHARS = 240
MAX_RELATION_CHARS = 160
MAX_SOURCE_ID_CHARS = 512
MAX_EVIDENCE_CHARS = 500
MAX_PROVENANCE_STATUS_CHARS = 64


def is_bounded_text(value: Any, max_chars: int, *, allow_empty: bool = False) -> bool:
    """Return whether a text value is present (unless allowed) and within its character budget."""
    if not isinstance(value, str) or len(value) > max_chars:
        return False
    if allow_empty:
        return True
    return bool(value.strip())
