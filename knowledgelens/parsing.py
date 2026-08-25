from __future__ import annotations

import json
import math
import unicodedata
from collections.abc import Iterable
from typing import Any

from .models import Claim, DocumentChunk

_IDENTIFIER_PUNCTUATION = frozenset("+#._-/:@")
_TRAILING_DECORATIVE_PUNCTUATION = "._-/@:"
_LEADING_DECORATIVE_PUNCTUATION = "_-/ :"
_MASTER_CONCEPT_PREFIXES = (
    "the central concept is ",
    "the central concept: ",
    "central concept is ",
    "central concept: ",
    "the main concept is ",
    "main concept is ",
    "master concept is ",
    "master concept: ",
    "the topic is ",
    "topic is ",
    "topic: ",
    "concept is ",
    "concept: ",
)


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned

    first_newline = cleaned.find("\n")
    if first_newline == -1:
        return cleaned

    header = cleaned[:first_newline].strip().casefold()
    if header not in {"```", "```json"}:
        return cleaned

    cleaned = cleaned[first_newline + 1 :]
    trimmed = cleaned.rstrip()
    if trimmed.endswith("```"):
        trimmed = trimmed[:-3]
    return trimmed.strip()


def canonicalize_label(display: str) -> str:
    """Create a Unicode-aware key without erasing technology/identifier punctuation."""
    folded = unicodedata.normalize("NFKC", display).casefold()
    chars = [char if char.isalnum() or char in _IDENTIFIER_PUNCTUATION else " " for char in folded]

    tokens: list[str] = []
    for raw_token in "".join(chars).split():
        # Periods, slashes, colons, etc. are useful *inside* identifiers (node.js,
        # C++/CLI, namespace::type) but are commonly decorative at the end of prose.
        token = raw_token.rstrip(_TRAILING_DECORATIVE_PUNCTUATION)
        token = token.lstrip(_LEADING_DECORATIVE_PUNCTUATION)
        if token and any(char.isalnum() for char in token):
            tokens.append(token)
    return " ".join(tokens)


def normalize_entity(raw: Any) -> tuple[str, str] | None:
    if raw is None:
        return None

    display = " ".join(str(raw).strip().split())
    if not display:
        return None

    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    if display.casefold() in stopwords:
        return None

    canonical = canonicalize_label(display)
    if not canonical:
        return None
    return canonical, display


def normalize_relation(raw: Any) -> tuple[str, str] | None:
    """Normalize a predicate without applying entity-only stopword rejection."""
    if raw is None:
        return None
    display = " ".join(str(raw).strip().split())
    if not display:
        return None
    canonical = canonicalize_label(display)
    if not canonical:
        return None
    return canonical, display


def _strip_master_concept_prefix(concept: str) -> str:
    folded = concept.casefold()
    for prefix in _MASTER_CONCEPT_PREFIXES:
        if folded.startswith(prefix):
            stripped = concept[len(prefix) :].strip()
            if stripped:
                return stripped
    return concept


def parse_master_concept_response(text: str) -> str:
    """Return a concise concept line, tolerating common fences and explanatory prefixes."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :].rstrip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].rstrip()

    for raw_line in cleaned.splitlines():
        concept = raw_line.strip().strip("\"'`").strip()
        if concept:
            return _strip_master_concept_prefix(concept)
    return ""


def _coerce_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    if 0 <= numeric <= 1:
        return numeric
    if 1 < numeric <= 100:
        return numeric / 100
    return None


def _claim_from_mapping(item: dict[str, Any], chunk: DocumentChunk) -> Claim | None:
    subject = item.get("subject") or item.get("s")
    relation = item.get("relation") or item.get("relationship") or item.get("predicate") or item.get("r")
    obj = item.get("object") or item.get("o")

    ns = normalize_entity(subject)
    nr = normalize_relation(relation)
    no = normalize_entity(obj)
    if not (ns and nr and no):
        return None

    evidence = " ".join(str(item.get("evidence") or item.get("quote") or "").split())
    if not evidence:
        return None
    if len(evidence) > 500:
        evidence = evidence[:497] + "..."

    return Claim(
        subject=ns[1],
        relation=nr[1],
        object=no[1],
        source=chunk.source,
        chunk_index=chunk.chunk_index,
        page=chunk.page,
        evidence=evidence,
        confidence=_coerce_confidence(item.get("confidence")),
    )


def _iter_json_items(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("claims", "triples", "relationships", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
                return
        if {"subject", "object"}.issubset(payload.keys()):
            yield payload
    elif isinstance(payload, list):
        yield from (item for item in payload if isinstance(item, dict))


def parse_claims(text: str, chunk: DocumentChunk) -> list[Claim]:
    """Parse structured JSON first, then fall back to the legacy pipe format only if JSON decoding fails."""
    cleaned = _strip_markdown_fence(text)
    claims: list[Claim] = []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None
    else:
        for item in _iter_json_items(payload):
            claim = _claim_from_mapping(item, chunk)
            if claim:
                claims.append(claim)
        return claims

    for raw_line in cleaned.splitlines():
        line = raw_line.strip().lstrip("0123456789.-•*# ").strip()
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 3:
            continue
        subject, relation, obj = parts[:3]
        evidence = parts[3] if len(parts) >= 4 else ""
        confidence = parts[4] if len(parts) >= 5 else None
        claim = _claim_from_mapping(
            {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "evidence": evidence,
                "confidence": confidence,
            },
            chunk,
        )
        if claim:
            claims.append(claim)

    return claims
