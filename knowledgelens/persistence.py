from __future__ import annotations

import json
from typing import Any

import networkx as nx

from .graph import canonical_key

STATE_SCHEMA_VERSION = 2


def reconstruct_node_map(graph: nx.MultiDiGraph) -> dict[str, str]:
    return {canonical_key(str(node)): str(node) for node in graph.nodes}


def migrate_legacy_graph(graph: nx.Graph, master_concept: str | None = None) -> nx.MultiDiGraph:
    if isinstance(graph, nx.MultiDiGraph):
        return graph

    migrated = nx.MultiDiGraph()
    migrated.add_nodes_from(graph.nodes(data=True))

    legacy_masters = {node for node, data in graph.nodes(data=True) if data.get("type") == "master"}
    if master_concept in graph:
        legacy_masters.add(master_concept)

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

        legacy_synthetic = subject in legacy_masters and not legacy_sources
        if legacy_synthetic:
            evidence = (
                "Migrated from a source-less KnowledgeLens v1 master overview link. "
                "This topology edge is synthetic and must not be used as document evidence."
            )
            provenance_status = "legacy-synthetic"
        else:
            source_note = ", ".join(legacy_sources) if legacy_sources else "unknown"
            evidence = (
                "Migrated from KnowledgeLens graph state v1. The legacy schema did not preserve "
                f"which relation was supported by which source; candidate sources: {source_note}."
            )
            provenance_status = "legacy-aggregated"

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
                synthetic=legacy_synthetic,
                provenance_status=provenance_status,
            )

    return migrated


def _node_link_graph(graph_data: dict[str, Any]) -> nx.Graph:
    # Both supported KnowledgeLens state generations are directed. Accepting an
    # undirected payload would force migration to invent an arbitrary edge direction
    # from serialization order, corrupting the meaning of claims and exports.
    if graph_data.get("directed") is not True:
        raise ValueError("Graph state must describe a directed graph; undirected states are not supported.")

    # Pin the field name for new state while accepting files written by NetworkX <=3.5.
    if "edges" in graph_data:
        return nx.node_link_graph(graph_data, edges="edges")
    if "links" in graph_data:
        return nx.node_link_graph(graph_data, edges="links")
    raise ValueError("Graph state is missing a supported edge collection ('edges' or legacy 'links').")


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

    schema_version = state.get("schema_version", 1)
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("Graph state contains an invalid schema_version.")
    if schema_version > STATE_SCHEMA_VERSION:
        raise ValueError(
            f"Graph state schema v{schema_version} is newer than this KnowledgeLens build supports "
            f"(v{STATE_SCHEMA_VERSION})."
        )

    graph_data = state.get("graph_data")
    if not isinstance(graph_data, dict):
        raise ValueError("Graph state does not contain valid graph_data.")

    master = state.get("master_concept")
    loaded = migrate_legacy_graph(_node_link_graph(graph_data), master_concept=master)
    node_map = state.get("node_canonical_map")
    if not isinstance(node_map, dict):
        node_map = reconstruct_node_map(loaded)
    return loaded, master, node_map
