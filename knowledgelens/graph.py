from __future__ import annotations

import hashlib
import math
import unicodedata
from collections import defaultdict
from numbers import Real
from typing import Any

import networkx as nx

from .models import Claim
from .parsing import normalize_entity


def canonical_key(label: str) -> str:
    normalized = normalize_entity(label)
    return normalized[0] if normalized else label.strip().casefold()


def _identity_text(value: str) -> str:
    """Normalize Unicode/case/whitespace while preserving identifier punctuation."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _valid_confidence(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and 0 <= numeric <= 1


def _valid_claim_shape(data: dict[str, Any]) -> bool:
    relation = data.get("relation")
    evidence = data.get("evidence")
    chunk_index = data.get("chunk_index")
    page = data.get("page")
    synthetic = data.get("synthetic")

    if not isinstance(relation, str) or not relation.strip():
        return False
    if not isinstance(evidence, str) or not evidence.strip():
        return False
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        return False
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        return False
    if not isinstance(synthetic, bool) or not _valid_confidence(data.get("confidence")):
        return False
    return True


def is_auditable_claim_data(data: dict[str, Any]) -> bool:
    """Return whether an edge has complete source-backed provenance safe for grounded use."""
    if not _valid_claim_shape(data):
        return False
    if data["synthetic"] or data.get("provenance_status") == "legacy-aggregated":
        return False
    source = data.get("source")
    return isinstance(source, str) and bool(source.strip())


def create_graph(master_concept: str) -> tuple[nx.MultiDiGraph, dict[str, str]]:
    graph = nx.MultiDiGraph()
    graph.add_node(master_concept, label=master_concept, type="master")
    return graph, {canonical_key(master_concept): master_concept}


def get_or_create_node(
    graph: nx.MultiDiGraph,
    node_map: dict[str, str],
    label: str,
    node_type: str = "entity",
) -> str | None:
    normalized = normalize_entity(label)
    if not normalized:
        return None
    canonical, display = normalized

    existing = node_map.get(canonical)
    if existing:
        if graph.nodes[existing].get("type") != "master" and node_type == "master":
            graph.nodes[existing]["type"] = "master"
        return existing

    graph.add_node(display, label=display, type=node_type)
    node_map[canonical] = display
    return display


def _claim_key(claim: Claim, subject: str, obj: str) -> str:
    """Build semantic/provenance identity after entity resolution.

    Chunk index is intentionally not part of identity: forced overlap may present the
    same exact source evidence to adjacent chunks. Page remains part of the key so an
    identical excerpt repeated on distinct PDF pages keeps independent provenance.
    """
    raw = "\x1f".join(
        [
            _identity_text(subject),
            _identity_text(claim.relation),
            _identity_text(obj),
            _identity_text(claim.source),
            str(claim.page),
            _identity_text(claim.evidence),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _claim_edge_data(claim: Claim) -> dict[str, Any] | None:
    data: dict[str, Any] = {
        "relation": claim.relation,
        "source": claim.source,
        "page": claim.page,
        "chunk_index": claim.chunk_index,
        "evidence": claim.evidence,
        "confidence": claim.confidence,
        "synthetic": claim.synthetic,
    }
    if not _valid_claim_shape(data):
        return None
    if not claim.synthetic and not is_auditable_claim_data(data):
        return None
    return data


def add_claims(
    graph: nx.MultiDiGraph,
    node_map: dict[str, str],
    claims: list[Claim],
) -> int:
    added = 0
    for claim in claims:
        edge_data = _claim_edge_data(claim)
        if edge_data is None:
            continue

        subject = get_or_create_node(graph, node_map, claim.subject)
        obj = get_or_create_node(graph, node_map, claim.object)
        if not subject or not obj:
            continue

        key = _claim_key(claim, subject, obj)
        if graph.has_edge(subject, obj, key=key):
            continue
        graph.add_edge(subject, obj, key=key, **edge_data)
        added += 1
    return added


def add_master_links(
    graph: nx.MultiDiGraph,
    node_map: dict[str, str],
    master: str,
    links: list[tuple[str, str]],
) -> int:
    claims: list[Claim] = []
    for node, relation in links:
        if node == master:
            continue
        claims.append(
            Claim(
                subject=master,
                relation=relation or "relates to",
                object=node,
                source="KnowledgeLens",
                chunk_index=0,
                evidence="Synthetic overview link generated from graph importance.",
                confidence=None,
                synthetic=True,
            )
        )
    return add_claims(graph, node_map, claims)


def top_entities(graph: nx.MultiDiGraph, limit: int = 20) -> list[str]:
    entities = [node for node in graph.nodes if graph.nodes[node].get("type") != "master"]
    return sorted(entities, key=lambda node: graph.degree(node), reverse=True)[:limit]


def graph_to_export(graph: nx.MultiDiGraph) -> dict[str, Any]:
    masters = [node for node in graph.nodes if graph.nodes[node].get("type") == "master"]
    claims: list[dict[str, Any]] = []
    for subject, obj, key, data in graph.edges(keys=True, data=True):
        claims.append(
            {
                "id": key,
                "subject": subject,
                "relation": data.get("relation", "related to"),
                "object": obj,
                "source": data.get("source", ""),
                "legacy_sources": list(data.get("legacy_sources", [])),
                "page": data.get("page"),
                "chunk_index": data.get("chunk_index"),
                "evidence": data.get("evidence", ""),
                "confidence": data.get("confidence"),
                "synthetic": bool(data.get("synthetic", False)),
                "provenance_status": data.get("provenance_status"),
            }
        )

    # `sources` is a claim-count ledger only for claims whose primary source is
    # actually known. Legacy v1 migration preserved candidate source identities
    # but not relation↔source pairings, so candidates are exposed separately and
    # never receive fabricated per-claim counts.
    source_counts: dict[str, int] = defaultdict(int)
    legacy_source_candidates: set[str] = set()
    for claim in claims:
        if claim["synthetic"]:
            continue
        if claim["source"]:
            source_counts[claim["source"]] += 1
            continue
        if claim.get("provenance_status") == "legacy-aggregated":
            legacy_source_candidates.update(
                str(source) for source in claim.get("legacy_sources", []) if source
            )

    auditable_claims = sum(1 for claim in claims if is_auditable_claim_data(claim))
    legacy_claims = sum(
        1
        for claim in claims
        if not claim["synthetic"] and claim.get("provenance_status") == "legacy-aggregated"
    )
    topology_edges = sum(1 for claim in claims if claim["synthetic"])
    source_identities = set(source_counts) | legacy_source_candidates

    return {
        "schema_version": 2,
        "master_concept": masters[0] if masters else None,
        "stats": {
            "nodes": graph.number_of_nodes(),
            "claims": auditable_claims,
            "legacy_claims": legacy_claims,
            "topology_edges": topology_edges,
            "edges_total": len(claims),
            "sources": len(source_identities),
        },
        "entities": [
            {
                "name": node,
                "type": graph.nodes[node].get("type", "entity"),
                "connections": graph.degree(node),
            }
            for node in graph.nodes
        ],
        "claims": claims,
        "sources": dict(sorted(source_counts.items())),
        "legacy_source_candidates": sorted(legacy_source_candidates),
    }
