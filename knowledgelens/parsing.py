from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from .models import Claim, DocumentChunk

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def normalize_entity(raw: Any) -> tuple[str, str] | None:
    if raw is None:
        return None
    display = " ".join(str(raw).strip().split())
    if len(display) < 2:
        return None

    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
    if display.lower() in stopwords:
        return None

    canonical = re.sub(r"[^a-z0-9]+", " ", display.lower()).strip()
    canonical = " ".join(canonical.split())
    if not canonical:
        return None
    return canonical, display


def _coerce_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric > 1 and numeric <= 100:
        numeric /= 100
    return min(1.0, max(0.0, numeric))


def _claim_from_mapping(item: dict[str, Any], chunk: DocumentChunk) -> Claim | None:
    subject = item.get("subject") or item.get("s")
    relation = item.get("relation") or item.get("relationship") or item.get("predicate") or item.get("r")
    obj = item.get("object") or item.get("o")

    ns = normalize_entity(subject)
    nr = normalize_entity(relation)
    no = normalize_entity(obj)
    if not (ns and nr and no):
        return None

    evidence = " ".join(str(item.get("evidence") or item.get("quote") or "").split())
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
                for item in value:
                    if isinstance(item, dict):
                        yield item
                return
        if {"subject", "object"}.issubset(payload.keys()):
            yield payload
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item


def parse_claims(text: str, chunk: DocumentChunk) -> list[Claim]:
    """Parse structured JSON first, then fall back to the legacy pipe format."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    claims: list[Claim] = []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = None

    if payload is not None:
        for item in _iter_json_items(payload):
            claim = _claim_from_mapping(item, chunk)
            if claim:
                claims.append(claim)
        if claims:
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
