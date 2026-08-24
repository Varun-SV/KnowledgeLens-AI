import json

import networkx as nx

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
