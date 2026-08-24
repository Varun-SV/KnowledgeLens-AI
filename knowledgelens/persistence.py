from __future__ import annotations

import json
from typing import Any

import networkx as nx

from .graph import canonical_key

STATE_SCHEMA_VERSION = 2


def reconstruct_node_map(graph: nx.MultiDiGraph) -> dict[str, str]:
    return {canonical_key(str(node)): str(node) for node in graph.nodes}


def migrate_legacy_graph(graph: nx.Graph) -> nx.MultiDiGraph:
    if isinstance(graph, nx.MultiDiGraph):
        return graph

    migrated = nx.MultiDiGraph()
    migrated.add_nodes_from(graph.nodes(data=True))

    for subject, obj, data in graph.edges(data=True):
        relations = data.get("relations") or [data.get("relation") or "related to"]
        if isinstance(relations, str):
            relations = [relations]

        raw_sources = data.get("sources")
        if not raw_sources:
            source = data.get("source")
            raw_sources = [source] if source else []
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        legacy_sources = sorted({str(source) for source in raw_sources if source})

        source_note = ", ".join(legacy_sources) if legacy_sources else "unknown"
        evidence = (
            "Migrated from KnowledgeLens graph state v1. The legacy schema did not preserve "
            f"which relation was supported by which source; candidate sources: {source_note}."
        )

        # Create one claim per relation, not a relation×source Cartesian product.
        for relation in relations:
            migrated.add_edge(
                subject,
                obj,
                relation=relation,
                source="",
                legacy_sources=legacy_sources,
                page=None,
                chunk_index=0,
                evidence=evidence,
                confidence=None,
                synthetic=False,
                provenance_status="legacy-aggregated",
            )

    return migrated


def _node_link_graph(graph_data: dict[str, Any]) -> nx.Graph:
    # Pin the field name for new state while accepting files written by NetworkX <=3.5.
    if "edges" in graph_data:
        return nx.node_link_graph(graph_data, edges="edges")
    if "links" in graph_data:
        return nx.node_link_graph(graph_data, edges="links")

    copy = dict(graph_data)
    copy["edges"] = []
    return nx.node_link_graph(copy, edges="edges")


def serialize_graph_state(
    graph: nx.MultiDiGraph,
    master_concept: str | None,
    node_canonical_map: dict[str, str],
) -> str:
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "master_concept": master_concept,
        "node_canonical_map": node_canonical_map,
        "graph_data": nx.node_link_data(graph, edges="edges"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def deserialize_graph_state(raw: bytes | str) -> tuple[nx.MultiDiGraph, str | None, dict[str, str]]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    state = json.loads(raw)
    graph_data = state.get("graph_data")
    if not isinstance(graph_data, dict):
        raise ValueError("Graph state does not contain valid graph_data.")

    loaded = migrate_legacy_graph(_node_link_graph(graph_data))
    master = state.get("master_concept")
    node_map = state.get("node_canonical_map")
    if not isinstance(node_map, dict):
        node_map = reconstruct_node_map(loaded)
    return loaded, master, node_map
