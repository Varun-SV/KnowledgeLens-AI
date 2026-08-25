import math

import networkx as nx

from knowledgelens.graph import add_claims, add_master_links, create_graph, graph_to_export
from knowledgelens.models import Claim


def _claim(subject, relation, obj, source, chunk_index, *, page=None, confidence=None):
    return Claim(
        subject,
        relation,
        obj,
        source,
        chunk_index,
        page=page,
        evidence=f"{subject} {relation} {obj}",
        confidence=confidence,
    )


def test_master_node_cannot_be_downgraded():
    graph, node_map = create_graph("Machine Learning")
    add_claims(graph, node_map, [_claim("Machine Learning", "uses", "Data", "paper.pdf", 1)])
    assert graph.nodes["Machine Learning"]["type"] == "master"


def test_multiple_claims_keep_independent_provenance():
    graph, node_map = create_graph("System")
    add_claims(
        graph,
        node_map,
        [
            _claim("Cache", "reduces", "Latency", "perf.pdf", 1, page=2),
            _claim("Cache", "increases", "Staleness", "risk.md", 2),
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
            _claim("C", "differs from", "C++", "languages.md", 1),
            _claim("C#", "targets", ".NET", "languages.md", 2),
        ],
    )

    assert {"C", "C++", "C#", ".NET"}.issubset(graph.nodes)
    assert node_map["c"] == "C"
    assert node_map["c++"] == "C++"
    assert node_map["c#"] == "C#"
    assert node_map[".net"] == ".NET"


def test_graph_admission_rejects_malformed_direct_claims():
    graph, node_map = create_graph("System")
    malformed = [
        Claim("A", "supports", "B", "doc.md", 1, evidence=""),
        Claim("C", "supports", "D", "", 1, evidence="C supports D"),
        Claim("E", "", "F", "doc.md", 1, evidence="E supports F"),
        Claim("G", "supports", "H", "doc.md", 1, evidence="G supports H", confidence=True),
        Claim("I", "supports", "J", "doc.md", 1, evidence="I supports J", confidence=math.nan),
    ]

    assert add_claims(graph, node_map, malformed) == 0
    assert graph.number_of_edges() == 0
    assert graph_to_export(graph)["stats"]["claims"] == 0


def test_synthetic_master_links_do_not_count_as_sources():
    graph, node_map = create_graph("System")
    add_claims(graph, node_map, [_claim("Cache", "reduces", "Latency", "perf.pdf", 1)])
    add_master_links(graph, node_map, "System", [("Cache", "includes")])
    export = graph_to_export(graph)
    assert export["stats"]["sources"] == 1
    assert export["sources"] == {"perf.pdf": 1}


def test_synthetic_master_links_do_not_inflate_claim_totals():
    graph, node_map = create_graph("System")
    add_claims(graph, node_map, [_claim("Cache", "reduces", "Latency", "perf.pdf", 1)])
    add_master_links(graph, node_map, "System", [("Cache", "includes"), ("Latency", "covers")])

    export = graph_to_export(graph)

    assert graph.number_of_edges() == 3
    assert export["stats"]["claims"] == 1
    assert export["stats"]["legacy_claims"] == 0
    assert export["stats"]["topology_edges"] == 2
    assert export["stats"]["edges_total"] == 3


def test_migrated_legacy_sources_remain_visible_but_not_counted_as_auditable_claims():
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

    assert export["stats"]["claims"] == 0
    assert export["stats"]["legacy_claims"] == 1
    assert export["stats"]["sources"] == 2
    assert export["sources"] == {"notes.md": 1, "paper.pdf": 1}
