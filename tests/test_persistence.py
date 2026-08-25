import json

import networkx as nx
import pytest

from knowledgelens.persistence import deserialize_graph_state, migrate_legacy_graph, serialize_graph_state

_NODE_LINK_FIELDS = {
    "source": "__kl_from",
    "target": "__kl_to",
    "name": "__kl_id",
    "key": "__kl_key",
}
_NODE_LINK_META = "_knowledgelens_node_link_fields"


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


def _valid_v2_graph() -> nx.MultiDiGraph:
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
        evidence="The source supports this relation.",
        confidence=0.9,
        synthetic=False,
        provenance_status=None,
    )
    return graph


def test_serialization_pins_edges_and_namespaces_node_link_structure():
    graph = _valid_v2_graph()
    raw = serialize_graph_state(graph, "A", {"a": "A", "b": "B"})
    payload = json.loads(raw)
    graph_data = payload["graph_data"]

    assert "edges" in graph_data
    assert "links" not in graph_data
    assert graph_data[_NODE_LINK_META] == _NODE_LINK_FIELDS

    edge = graph_data["edges"][0]
    assert edge["__kl_from"] == "A"
    assert edge["__kl_to"] == "B"
    assert edge["__kl_key"] == "claim"
    assert edge["source"] == "doc.md"


def test_serialization_round_trip_preserves_claim_provenance_source():
    graph = _valid_v2_graph()
    raw = serialize_graph_state(graph, "A", {"a": "A", "b": "B"})
    loaded, master, _ = deserialize_graph_state(raw)

    assert master == "A"
    data = next(iter(loaded.edges(data=True)))[2]
    assert data["source"] == "doc.md"
    assert data["evidence"] == "The source supports this relation."


def test_loader_accepts_legacy_links_collection_with_namespaced_v2_fields():
    graph = _valid_v2_graph()
    data = nx.node_link_data(graph, edges="links", **_NODE_LINK_FIELDS)
    data[_NODE_LINK_META] = dict(_NODE_LINK_FIELDS)
    raw = json.dumps({"schema_version": 2, "master_concept": "A", "graph_data": data})

    loaded, master, _ = deserialize_graph_state(raw)

    assert master == "A"
    assert loaded.number_of_edges() == 1
    assert next(iter(loaded.edges(data=True)))[2]["source"] == "doc.md"


def test_loader_fails_closed_for_early_v2_default_fields_that_lost_provenance_source():
    graph = _valid_v2_graph()
    # NetworkX's default structural endpoint field is also named `source`, so this
    # legacy encoding cannot preserve the claim's provenance `source` attribute.
    data = nx.node_link_data(graph, edges="links")
    raw = json.dumps({"schema_version": 2, "master_concept": "A", "graph_data": data})

    with pytest.raises(ValueError, match="non-empty source"):
        deserialize_graph_state(raw)


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


def test_loader_rejects_legacy_state_with_mismatched_or_multiple_master_nodes():
    mismatched = nx.DiGraph()
    mismatched.add_node("A", type="entity")
    mismatched.add_node("B", type="master")
    mismatched.add_edge("A", "B", relations=["supports"], sources=["doc.md"])

    multiple = nx.DiGraph()
    multiple.add_node("A", type="master")
    multiple.add_node("B", type="master")
    multiple.add_edge("A", "B", relations=["supports"], sources=["doc.md"])

    with pytest.raises(ValueError, match="exactly one master"):
        deserialize_graph_state(
            json.dumps({"master_concept": "A", "graph_data": nx.node_link_data(mismatched, edges="links")})
        )
    with pytest.raises(ValueError, match="exactly one master"):
        deserialize_graph_state(
            json.dumps({"master_concept": "A", "graph_data": nx.node_link_data(multiple, edges="links")})
        )


def test_loader_rejects_wrong_multigraph_shape_for_each_schema():
    v1_multigraph = json.dumps(
        {
            "schema_version": 1,
            "master_concept": "A",
            "graph_data": {"directed": True, "multigraph": True, "nodes": [{"id": "A"}], "links": []},
        }
    )
    v2_simple = json.dumps(
        {
            "schema_version": 2,
            "master_concept": "A",
            "graph_data": {"directed": True, "multigraph": False, "nodes": [{"id": "A"}], "edges": []},
        }
    )
    with pytest.raises(ValueError, match="v1.*non-multigraph"):
        deserialize_graph_state(v1_multigraph)
    with pytest.raises(ValueError, match="v2.*MultiDiGraph"):
        deserialize_graph_state(v2_simple)


def test_loader_rejects_v2_edges_that_bypass_source_backed_invariants():
    raw = json.dumps(
        {
            "schema_version": 2,
            "master_concept": "A",
            "graph_data": {
                "directed": True,
                "multigraph": True,
                "nodes": [{"id": "A", "type": "master"}, {"id": "B"}],
                "edges": [
                    {
                        "source": "A",
                        "target": "B",
                        "key": "bad",
                        "relation": "supports",
                        "chunk_index": 1,
                        "page": None,
                        "confidence": 0.9,
                        "synthetic": False,
                    }
                ],
            },
        }
    )
    with pytest.raises(ValueError, match="evidence"):
        deserialize_graph_state(raw)


def test_loader_rejects_non_string_node_ids_in_v2():
    raw = json.dumps(
        {
            "schema_version": 2,
            "master_concept": None,
            "graph_data": {"directed": True, "multigraph": True, "nodes": [{"id": 7}], "edges": []},
        }
    )
    with pytest.raises(ValueError, match="node identifiers"):
        deserialize_graph_state(raw)


def test_loader_rejects_missing_or_multiple_v2_master_nodes():
    missing_master = nx.MultiDiGraph()
    missing_master.add_node("A", type="entity")
    missing_data = nx.node_link_data(missing_master, edges="edges", **_NODE_LINK_FIELDS)
    missing_data[_NODE_LINK_META] = dict(_NODE_LINK_FIELDS)

    multiple_masters = nx.MultiDiGraph()
    multiple_masters.add_node("A", type="master")
    multiple_masters.add_node("B", type="master")
    multiple_data = nx.node_link_data(multiple_masters, edges="edges", **_NODE_LINK_FIELDS)
    multiple_data[_NODE_LINK_META] = dict(_NODE_LINK_FIELDS)

    with pytest.raises(ValueError, match="exactly one master"):
        deserialize_graph_state(
            json.dumps({"schema_version": 2, "master_concept": "A", "graph_data": missing_data})
        )
    with pytest.raises(ValueError, match="exactly one master"):
        deserialize_graph_state(
            json.dumps({"schema_version": 2, "master_concept": "A", "graph_data": multiple_data})
        )


def test_migrated_v1_state_can_be_reserialized_and_loaded_as_v2():
    graph = nx.DiGraph()
    graph.add_node("A", type="master")
    graph.add_node("B", type="entity")
    graph.add_edge("A", "B", relations=["supports"], sources=["doc.md"])
    v1 = json.dumps(
        {
            "schema_version": 1,
            "master_concept": "A",
            "graph_data": nx.node_link_data(graph, edges="links"),
        }
    )

    migrated, master, node_map = deserialize_graph_state(v1)
    v2 = serialize_graph_state(migrated, master, node_map)
    reloaded, reloaded_master, _ = deserialize_graph_state(v2)

    assert reloaded_master == "A"
    assert reloaded.number_of_edges() == 1
    data = next(iter(reloaded.edges(data=True)))[2]
    assert data["provenance_status"] == "legacy-aggregated"


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
