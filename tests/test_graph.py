import networkx as nx

from knowledgelens.graph import add_claims, add_master_links, create_graph, graph_to_export
from knowledgelens.models import Claim


def test_master_node_cannot_be_downgraded():
    graph, node_map = create_graph("Machine Learning")
    add_claims(
        graph,
        node_map,
        [
            Claim(
                subject="Machine Learning",
                relation="uses",
                object="Data",
                source="paper.pdf",
                chunk_index=1,
            )
        ],
    )
    assert graph.nodes["Machine Learning"]["type"] == "master"


def test_multiple_claims_keep_independent_provenance():
    graph, node_map = create_graph("System")
    add_claims(
        graph,
        node_map,
        [
            Claim("Cache", "reduces", "Latency", "perf.pdf", 1, page=2),
            Claim("Cache", "increases", "Staleness", "risk.md", 2),
        ],
    )
    export = graph_to_export(graph)
    assert {c["source"] for c in export["claims"]} == {"perf.pdf", "risk.md"}


def test_programming_language_entities_with_significant_punctuation_do_not_merge():
    graph, node_map = create_graph("Languages")
    add_claims(
        graph,
        node_map,
        [
            Claim("C", "differs from", "C++", "languages.md", 1),
            Claim("C#", "targets", ".NET", "languages.md", 2),
        ],
    )

    assert {"C", "C++", "C#", ".NET"}.issubset(graph.nodes)
    assert node_map["c"] == "C"
    assert node_map["c++"] == "C++"
    assert node_map["c#"] == "C#"
    assert node_map[".net"] == ".NET"


def test_synthetic_master_links_do_not_count_as_sources():
    graph, node_map = create_graph("System")
    add_claims(graph, node_map, [Claim("Cache", "reduces", "Latency", "perf.pdf", 1)])
    add_master_links(graph, node_map, "System", [("Cache", "includes")])
    export = graph_to_export(graph)
    assert export["stats"]["sources"] == 1
    assert export["sources"] == {"perf.pdf": 1}


def test_migrated_legacy_sources_remain_visible_in_source_totals():
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="master")
    graph.add_node("B", type="entity")
    graph.add_edge(
        "A",
        "B",
        key="legacy",
        relation="relates to",
        source="",
        legacy_sources=["paper.pdf", "notes.md"],
        page=None,
        chunk_index=0,
        evidence="legacy aggregated provenance",
        confidence=None,
        synthetic=False,
        provenance_status="legacy-aggregated",
    )

    export = graph_to_export(graph)

    assert export["stats"]["sources"] == 2
    assert export["sources"] == {"notes.md": 1, "paper.pdf": 1}
