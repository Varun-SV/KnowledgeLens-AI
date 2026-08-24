from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import networkx as nx

from .models import Claim
from .parsing import normalize_entity


def canonical_key(label: str) -> str:
    normalized = normalize_entity(label)
    return normalized[0] if normalized else label.strip().lower()


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


def _claim_key(claim: Claim) -> str:
    raw = "\x1f".join(
        [
            claim.subject,
            claim.relation,
            claim.object,
            claim.source,
            str(claim.page),
            str(claim.chunk_index),
            claim.evidence,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add_claims(
    graph: nx.MultiDiGraph,
    node_map: dict[str, str],
    claims: list[Claim],
) -> int:
    added = 0
    for claim in claims:
        subject = get_or_create_node(graph, node_map, claim.subject)
        obj = get_or_create_node(graph, node_map, claim.object)
        if not subject or not obj:
            continue

        key = _claim_key(claim)
        if graph.has_edge(subject, obj, key=key):
            continue
        graph.add_edge(
            subject,
            obj,
            key=key,
            relation=claim.relation,
            source=claim.source,
            page=claim.page,
            chunk_index=claim.chunk_index,
            evidence=claim.evidence,
            confidence=claim.confidence,
            synthetic=claim.synthetic,
        )
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
                "page": data.get("page"),
                "chunk_index": data.get("chunk_index"),
                "evidence": data.get("evidence", ""),
                "confidence": data.get("confidence"),
                "synthetic": bool(data.get("synthetic", False)),
            }
        )

    source_counts: dict[str, int] = defaultdict(int)
    for claim in claims:
        if claim["source"]:
            source_counts[claim["source"]] += 1

    return {
        "schema_version": 2,
        "master_concept": masters[0] if masters else None,
        "stats": {
            "nodes": graph.number_of_nodes(),
            "claims": graph.number_of_edges(),
            "sources": len(source_counts),
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
    }
