from __future__ import annotations

import json
import math
from numbers import Real
from typing import Any

import networkx as nx

from .graph import canonical_key
from .limits import (
    MAX_ENTITY_LABEL_CHARS,
    MAX_EVIDENCE_CHARS,
    MAX_GRAPH_EDGES,
    MAX_GRAPH_NODES,
    MAX_PROVENANCE_STATUS_CHARS,
    MAX_RELATION_CHARS,
    MAX_SOURCE_ID_CHARS,
    is_bounded_text,
)

STATE_SCHEMA_VERSION = 2
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_STATE_NODES = MAX_GRAPH_NODES
MAX_STATE_EDGES = MAX_GRAPH_EDGES
_NODE_LINK_META = "_knowledgelens_node_link_fields"
_NODE_LINK_FIELDS = {
    "source": "__kl_from",
    "target": "__kl_to",
    "name": "__kl_id",
    "key": "__kl_key",
}


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


def _validate_serialized_complexity(graph_data: dict[str, Any]) -> None:
    """Reject oversized graph collections before NetworkX materializes the graph."""
    nodes = graph_data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("Graph state must contain a node collection.")

    edge_field = "edges" if "edges" in graph_data else "links" if "links" in graph_data else None
    if edge_field is None:
        raise ValueError("Graph state is missing a supported edge collection ('edges' or legacy 'links').")
    edges = graph_data.get(edge_field)
    if not isinstance(edges, list):
        raise ValueError("Graph state edge collection must be a list.")

    if len(nodes) > MAX_STATE_NODES:
        raise ValueError(f"Graph state exceeds the {MAX_STATE_NODES:,}-node safety limit.")
    if len(edges) > MAX_STATE_EDGES:
        raise ValueError(f"Graph state exceeds the {MAX_STATE_EDGES:,}-edge safety limit.")


def _validate_graph_complexity(graph: nx.Graph) -> None:
    if graph.number_of_nodes() > MAX_STATE_NODES:
        raise ValueError(f"Graph state exceeds the {MAX_STATE_NODES:,}-node safety limit.")
    if graph.number_of_edges() > MAX_STATE_EDGES:
        raise ValueError(f"Graph state exceeds the {MAX_STATE_EDGES:,}-edge safety limit.")


def _validate_state_size(raw: bytes | str) -> None:
    if isinstance(raw, bytes):
        size = len(raw)
    else:
        if len(raw) > MAX_STATE_BYTES:
            raise ValueError(f"Graph state exceeds the {MAX_STATE_BYTES // (1024 * 1024)} MiB safety limit.")
        size = len(raw.encode("utf-8"))
    if size > MAX_STATE_BYTES:
        raise ValueError(f"Graph state exceeds the {MAX_STATE_BYTES // (1024 * 1024)} MiB safety limit.")


def _node_link_kwargs(graph_data: dict[str, Any], edge_field: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Return sanitized node-link data and the field mapping needed to deserialize it."""
    payload = dict(graph_data)
    field_metadata = payload.pop(_NODE_LINK_META, None)
    if field_metadata is None:
        return payload, {"edges": edge_field}

    if field_metadata != _NODE_LINK_FIELDS:
        raise ValueError("Graph state declares an unsupported node-link field mapping.")

    return payload, {"edges": edge_field, **_NODE_LINK_FIELDS}


def _validate_serialized_identities(
    graph_data: dict[str, Any],
    schema_version: int,
    edge_field: str,
    node_link_kwargs: dict[str, str],
) -> None:
    """Reject structural identities that NetworkX would otherwise silently coalesce."""
    node_id_field = node_link_kwargs.get("name", "id")
    source_field = node_link_kwargs.get("source", "source")
    target_field = node_link_kwargs.get("target", "target")
    key_field = node_link_kwargs.get("key", "key")

    node_ids: set[str] = set()
    for node in graph_data["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("Graph state nodes must be JSON objects.")
        node_id = node.get(node_id_field)
        if not is_bounded_text(node_id, MAX_ENTITY_LABEL_CHARS):
            raise ValueError(
                f"Graph state node identifiers must be non-empty strings up to {MAX_ENTITY_LABEL_CHARS} characters."
            )
        if node_id in node_ids:
            raise ValueError(f"Graph state contains duplicate serialized node identifier: {node_id!r}.")
        node_ids.add(node_id)

    seen_edges: set[tuple[str, ...]] = set()
    for edge in graph_data[edge_field]:
        if not isinstance(edge, dict):
            raise ValueError("Graph state edges must be JSON objects.")
        subject = edge.get(source_field)
        obj = edge.get(target_field)
        if subject not in node_ids or obj not in node_ids:
            raise ValueError("Graph state edge endpoints must reference declared serialized nodes.")

        if schema_version == 2:
            key = edge.get(key_field)
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError("Graph state v2 edges must contain a string or integer structural key.")
            identity = (str(subject), str(obj), str(key))
        else:
            identity = (str(subject), str(obj))

        if identity in seen_edges:
            raise ValueError("Graph state contains duplicate serialized edge identity that would lose data.")
        seen_edges.add(identity)


def _node_link_graph(graph_data: dict[str, Any], schema_version: int) -> nx.Graph:
    _validate_graph_data_shape(graph_data, schema_version)
    _validate_serialized_complexity(graph_data)

    edge_field = "edges" if "edges" in graph_data else "links"
    payload, kwargs = _node_link_kwargs(graph_data, edge_field)
    _validate_serialized_identities(payload, schema_version, edge_field, kwargs)
    return nx.node_link_graph(payload, **kwargs)


def _validate_node_ids(graph: nx.Graph) -> None:
    if any(not is_bounded_text(node, MAX_ENTITY_LABEL_CHARS) for node in graph.nodes):
        raise ValueError(
            f"Graph state node identifiers must be non-empty strings up to {MAX_ENTITY_LABEL_CHARS} characters."
        )


def _validate_master(graph: nx.Graph, master_concept: str | None) -> None:
    master_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == "master"]
    if graph.number_of_nodes() == 0:
        if master_concept is not None or master_nodes:
            raise ValueError("An empty graph state cannot declare a master concept.")
        return

    if not is_bounded_text(master_concept, MAX_ENTITY_LABEL_CHARS):
        raise ValueError(
            f"A non-empty graph state must declare a master_concept up to {MAX_ENTITY_LABEL_CHARS} characters."
        )
    if master_concept not in graph:
        raise ValueError("Graph state master_concept does not exist in graph_data.")
    if len(master_nodes) != 1 or master_nodes[0] != master_concept:
        raise ValueError("Graph state must contain exactly one master node matching master_concept.")


def _validate_v2_graph(graph: nx.Graph, master_concept: str | None) -> None:
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("KnowledgeLens graph state v2 must deserialize to a MultiDiGraph.")

    _validate_graph_complexity(graph)
    _validate_node_ids(graph)
    _validate_master(graph, master_concept)

    for _subject, _obj, _key, data in graph.edges(keys=True, data=True):
        relation = data.get("relation")
        if not is_bounded_text(relation, MAX_RELATION_CHARS):
            raise ValueError(
                f"Graph state v2 relation values must be non-empty and at most {MAX_RELATION_CHARS} characters."
            )

        synthetic = data.get("synthetic")
        if not isinstance(synthetic, bool):
            raise ValueError("Graph state v2 edges must declare a boolean synthetic flag.")

        evidence = data.get("evidence")
        if not is_bounded_text(evidence, MAX_EVIDENCE_CHARS):
            raise ValueError(
                f"Graph state v2 evidence must be non-empty and at most {MAX_EVIDENCE_CHARS} characters."
            )

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
        if provenance_status is not None and not is_bounded_text(
            provenance_status, MAX_PROVENANCE_STATUS_CHARS
        ):
            raise ValueError(
                "Graph state v2 edge provenance_status must be a non-empty bounded string or null."
            )

        legacy_sources = data.get("legacy_sources", [])
        if not isinstance(legacy_sources, list) or any(
            not is_bounded_text(source, MAX_SOURCE_ID_CHARS) for source in legacy_sources
        ):
            raise ValueError(
                f"Graph state v2 legacy_sources must contain source identifiers up to {MAX_SOURCE_ID_CHARS} characters."
            )

        source = data.get("source", "")
        if not is_bounded_text(source, MAX_SOURCE_ID_CHARS, allow_empty=True):
            raise ValueError(
                f"Graph state v2 source identifiers must be at most {MAX_SOURCE_ID_CHARS} characters."
            )
        if not synthetic and provenance_status != "legacy-aggregated" and not source.strip():
            raise ValueError("Source-backed graph state v2 edges must contain a non-empty source.")


def _serialized_graph_data(graph: nx.MultiDiGraph) -> dict[str, Any]:
    graph_data = nx.node_link_data(graph, edges="edges", **_NODE_LINK_FIELDS)
    graph_data[_NODE_LINK_META] = dict(_NODE_LINK_FIELDS)
    return graph_data


def serialize_graph_state(
    graph: nx.MultiDiGraph,
    master_concept: str | None,
    node_canonical_map: dict[str, str],
) -> str:
    """Serialize only states that this build can subsequently deserialize."""
    _validate_v2_graph(graph, master_concept)
    graph_data = _serialized_graph_data(graph)
    _validate_serialized_complexity(graph_data)
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "master_concept": master_concept,
        "node_canonical_map": node_canonical_map,
        "graph_data": graph_data,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    _validate_state_size(raw)
    return raw


def deserialize_graph_state(raw: bytes | str) -> tuple[nx.MultiDiGraph, str | None, dict[str, str]]:
    _validate_state_size(raw)
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
        _validate_master(loaded_raw, master)
        loaded = migrate_legacy_graph(loaded_raw, master_concept=master)
        _validate_v2_graph(loaded, master)

    node_map = state.get("node_canonical_map")
    if not isinstance(node_map, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or value not in loaded
        for key, value in node_map.items()
    ):
        node_map = reconstruct_node_map(loaded)
    return loaded, master, node_map
