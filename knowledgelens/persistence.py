from __future__ import annotations

import json
import math
from numbers import Real
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


def _validate_graph_data_shape(graph_data: dict[str, Any], schema_version: int) -> None:
    if graph_data.get("directed") is not True:
        raise ValueError("Graph state must describe a directed graph; undirected states are not supported.")

    multigraph = graph_data.get("multigraph")
    if schema_version == 1 and multigraph is not False:
        raise ValueError("KnowledgeLens graph state v1 must contain a directed non-multigraph graph.")
    if schema_version == 2 and multigraph is not True:
        raise ValueError("KnowledgeLens graph state v2 must contain a directed MultiDiGraph.")


def _node_link_graph(graph_data: dict[str, Any], schema_version: int) -> nx.Graph:
    _validate_graph_data_shape(graph_data, schema_version)

    # Pin the field name for new state while accepting files written by NetworkX <=3.5.
    if "edges" in graph_data:
        return nx.node_link_graph(graph_data, edges="edges")
    if "links" in graph_data:
        return nx.node_link_graph(graph_data, edges="links")
    raise ValueError("Graph state is missing a supported edge collection ('edges' or legacy 'links').")


def _validate_node_ids(graph: nx.Graph) -> None:
    if any(not isinstance(node, str) or not node.strip() for node in graph.nodes):
        raise ValueError("Graph state node identifiers must be non-empty strings.")


def _validate_v2_graph(graph: nx.Graph, master_concept: str | None) -> None:
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("KnowledgeLens graph state v2 must deserialize to a MultiDiGraph.")

    _validate_node_ids(graph)

    if master_concept is not None:
        if not isinstance(master_concept, str) or not master_concept.strip():
            raise ValueError("Graph state master_concept must be a non-empty string or null.")
        if master_concept not in graph:
            raise ValueError("Graph state master_concept does not exist in graph_data.")
        if graph.nodes[master_concept].get("type") != "master":
            raise ValueError("Graph state master_concept node is not marked as type 'master'.")

    for _subject, _obj, _key, data in graph.edges(keys=True, data=True):
        relation = data.get("relation")
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("Graph state v2 contains an edge without a valid relation.")

        synthetic = data.get("synthetic")
        if not isinstance(synthetic, bool):
            raise ValueError("Graph state v2 edges must declare a boolean synthetic flag.")

        evidence = data.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("Graph state v2 contains an edge without supporting/provenance evidence.")

        chunk_index = data.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
            raise ValueError("Graph state v2 edge chunk_index must be a non-negative integer.")

        page = data.get("page")
        if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
            raise ValueError("Graph state v2 edge page must be null or a positive integer.")

        confidence = data.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, Real):
                raise ValueError("Graph state v2 edge confidence must be null or a number from 0 to 1.")
            numeric_confidence = float(confidence)
            if not math.isfinite(numeric_confidence) or not 0 <= numeric_confidence <= 1:
                raise ValueError("Graph state v2 edge confidence must be null or a finite number from 0 to 1.")

        provenance_status = data.get("provenance_status")
        if provenance_status is not None and not isinstance(provenance_status, str):
            raise ValueError("Graph state v2 edge provenance_status must be a string or null.")

        legacy_sources = data.get("legacy_sources", [])
        if not isinstance(legacy_sources, list) or any(
            not isinstance(source, str) or not source.strip() for source in legacy_sources
        ):
            raise ValueError("Graph state v2 legacy_sources must be a list of non-empty strings.")

        if not synthetic and provenance_status != "legacy-aggregated":
            source = data.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Source-backed graph state v2 edges must contain a non-empty source.")


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
    if not isinstance(state, dict):
        raise ValueError("Graph state root must be a JSON object.")

    schema_version = state.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
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
    loaded_raw = _node_link_graph(graph_data, schema_version)
    if schema_version == 2:
        _validate_v2_graph(loaded_raw, master)
        loaded = loaded_raw
    else:
        _validate_node_ids(loaded_raw)
        loaded = migrate_legacy_graph(loaded_raw, master_concept=master)

    node_map = state.get("node_canonical_map")
    if not isinstance(node_map, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or value not in loaded
        for key, value in node_map.items()
    ):
        node_map = reconstruct_node_map(loaded)
    return loaded, master, node_map
