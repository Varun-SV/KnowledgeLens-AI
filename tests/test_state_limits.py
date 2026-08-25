import json

import networkx as nx
import pytest

import knowledgelens.persistence as persistence


def _small_valid_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="master")
    graph.add_node("B", type="entity")
    graph.add_edge(
        "A",
        "B",
        key="claim",
        relation="supports",
        source="doc.md",
        legacy_sources=[],
        chunk_index=1,
        page=None,
        evidence="A supports B",
        confidence=0.9,
        synthetic=False,
        provenance_status=None,
    )
    return graph


def test_oversized_state_is_rejected_before_json_decode(monkeypatch):
    monkeypatch.setattr(persistence, "MAX_STATE_BYTES", 8)

    def must_not_decode(_raw):
        raise AssertionError("json.loads must not run for oversized state")

    monkeypatch.setattr(persistence.json, "loads", must_not_decode)
    with pytest.raises(ValueError, match="safety limit"):
        persistence.deserialize_graph_state(b"x" * 9)


def test_node_complexity_is_rejected_before_networkx_materialization(monkeypatch):
    monkeypatch.setattr(persistence, "MAX_STATE_NODES", 1)

    def must_not_materialize(*_args, **_kwargs):
        raise AssertionError("NetworkX must not materialize an oversized graph")

    monkeypatch.setattr(persistence.nx, "node_link_graph", must_not_materialize)
    raw = json.dumps(
        {
            "schema_version": 2,
            "master_concept": "A",
            "graph_data": {
                "directed": True,
                "multigraph": True,
                "nodes": [{"id": "A", "type": "master"}, {"id": "B", "type": "entity"}],
                "edges": [],
            },
        }
    )
    with pytest.raises(ValueError, match="node safety limit"):
        persistence.deserialize_graph_state(raw)


def test_edge_complexity_is_rejected_before_networkx_materialization(monkeypatch):
    monkeypatch.setattr(persistence, "MAX_STATE_EDGES", 0)

    def must_not_materialize(*_args, **_kwargs):
        raise AssertionError("NetworkX must not materialize an oversized graph")

    monkeypatch.setattr(persistence.nx, "node_link_graph", must_not_materialize)
    raw = json.dumps(
        {
            "schema_version": 2,
            "master_concept": "A",
            "graph_data": {
                "directed": True,
                "multigraph": True,
                "nodes": [{"id": "A", "type": "master"}, {"id": "B", "type": "entity"}],
                "edges": [{"__kl_from": "A", "__kl_to": "B", "__kl_key": "x"}],
                "_knowledgelens_node_link_fields": {
                    "source": "__kl_from",
                    "target": "__kl_to",
                    "name": "__kl_id",
                    "key": "__kl_key"
                }
            },
        }
    )
    with pytest.raises(ValueError, match="edge safety limit"):
        persistence.deserialize_graph_state(raw)


def test_serializer_rejects_graph_complexity_before_node_link_materialization(monkeypatch):
    graph = _small_valid_graph()
    monkeypatch.setattr(persistence, "MAX_STATE_NODES", 1)

    def must_not_serialize(*_args, **_kwargs):
        raise AssertionError("node_link_data must not run for an oversized graph")

    monkeypatch.setattr(persistence.nx, "node_link_data", must_not_serialize)
    with pytest.raises(ValueError, match="node safety limit"):
        persistence.serialize_graph_state(graph, "A", {"a": "A", "b": "B"})


def test_serializer_never_emits_state_above_loader_byte_limit(monkeypatch):
    graph = _small_valid_graph()
    monkeypatch.setattr(persistence, "MAX_STATE_BYTES", 64)

    with pytest.raises(ValueError, match="MiB safety limit"):
        persistence.serialize_graph_state(graph, "A", {"a": "A", "b": "B"})
