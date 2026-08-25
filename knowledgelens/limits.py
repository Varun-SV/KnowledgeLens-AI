from __future__ import annotations

from typing import Any

MAX_ENTITY_LABEL_CHARS = 240
MAX_RELATION_CHARS = 160
MAX_SOURCE_ID_CHARS = 512
MAX_EVIDENCE_CHARS = 500
MAX_PROVENANCE_STATUS_CHARS = 64
MAX_EXTRACTION_FOCUS_CHARS = 2_000
MAX_MODEL_NAME_CHARS = 200
MAX_CHAT_QUERY_CHARS = 4_000
MAX_API_KEY_CHARS = 8_192
MAX_REQUEST_HEADERS_BYTES = 16 * 1024
MAX_CLAIMS_PER_CHUNK = 200
MAX_GRAPH_NODES = 10_000
MAX_GRAPH_EDGES = 25_000
MAX_VISUALIZATION_NODES = 1_500
MAX_VISUALIZATION_EDGES = 5_000


def is_bounded_text(value: Any, max_chars: int, *, allow_empty: bool = False) -> bool:
    """Return whether a text value is present (unless allowed) and within its character budget."""
    if not isinstance(value, str) or len(value) > max_chars:
        return False
    if allow_empty:
        return True
    return bool(value.strip())
