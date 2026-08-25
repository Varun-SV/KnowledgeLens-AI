import json

import networkx as nx
import pytest

from knowledgelens.persistence import deserialize_graph_state, migrate_legacy_graph, serialize_graph_state


def test_v1_migration_does_not_fabricate_relation_source_pairs():
    old = nx.DiGraph()
    old.add_node("A", type="master")
    old.add_node("B", type="entity")
    old.add_edge("A", "B", relations=["r1", "r2"], sources={"s1", "s2"})
    migrated = migrate_legacy_graph(old)
    assert migrated.number_of_edges() == 2
    for _, _, data in migrated.edges(data=True):
        assert data["source"] == ""
        assert data["legacy_sources"] == ["s1", "s2"]
        assert data["provenance_status"] == "legacy-aggregated"
        assert data["synthetic"] is False


def test_v1_source_less_master_links_remain_synthetic():
    old = nx.DiGraph()
    old.add_node("Knowledge Base", type="master")
    old.add_node("Topic", type="entity")
    old.add_edge("Knowledge Base", "Topic", relations=["includes"], sources=set())

    migrated = migrate_legacy_graph(old, master_concept="Knowledge Base")
    data = next(iter(migrated.edges(data=True)))[2]

    assert data["synthetic"] is True
    assert data["provenance_status"] == "legacy-synthetic"
    assert data["legacy_sources"] == []


def test_serialization_pins_edges_field():
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="master")
    raw = serialize_graph_state(graph, "A", {"a": "A"})
    payload = json.loads(raw)
    assert "edges" in payload["graph_data"]
    assert "links" not in payload["graph_data"]


def test_loader_accepts_legacy_links_field():
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="master")
    graph.add_node("B")
    graph.add_edge("A", "B", relation="r", source="s", chunk_index=1, synthetic=False)
    data = nx.node_link_data(graph, edges="links")
    raw = json.dumps({"schema_version": 2, "master_concept": "A", "graph_data": data})
    loaded, master, _ = deserialize_graph_state(raw)
    assert master == "A"
    assert loaded.number_of_edges() == 1


def test_loader_rejects_missing_edge_collection():
    raw = json.dumps(
        {
            "schema_version": 2,
            "master_concept": "A",
            "graph_data": {"directed": True, "multigraph": True, "nodes": [{"id": "A"}]},
        }
    )
    with pytest.raises(ValueError, match="edge collection"):
        deserialize_graph_state(raw)


def test_loader_rejects_undirected_graph_state():
    raw = json.dumps(
        {
            "schema_version": 1,
            "master_concept": "A",
            "graph_data": {
                "directed": False,
                "multigraph": False,
                "nodes": [{"id": "A"}, {"id": "B"}],
                "links": [{"source": "A", "target": "B", "relations": ["supports"], "sources": ["doc.md"]}],
            },
        }
    )
    with pytest.raises(ValueError, match="directed graph"):
        deserialize_graph_state(raw)


def test_loader_rejects_newer_schema_versions():
    raw = json.dumps(
        {
            "schema_version": 999,
            "master_concept": "A",
            "graph_data": {"directed": True, "multigraph": True, "nodes": [{"id": "A"}], "edges": []},
        }
    )
    with pytest.raises(ValueError, match="newer"):
        deserialize_graph_state(raw)
