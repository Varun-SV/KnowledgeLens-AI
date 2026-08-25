import math

import networkx as nx
import pytest

import knowledgelens.graph as graph_module
from knowledgelens.graph import GraphCapacityError, add_claims, add_master_links, create_graph, graph_to_export
from knowledgelens.limits import MAX_ENTITY_LABEL_CHARS, MAX_EVIDENCE_CHARS, MAX_RELATION_CHARS, MAX_SOURCE_ID_CHARS
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


def test_master_concept_is_bounded_at_graph_creation():
    with pytest.raises(ValueError, match="Master concept"):
        create_graph("M" * (MAX_ENTITY_LABEL_CHARS + 1))


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


def test_claim_identity_deduplicates_case_and_nfkc_variants_after_node_resolution():
    graph, node_map = create_graph("System")
    claims = [
        Claim("Cache", "reduces", "Latency", "doc.md", 1, evidence="Cache reduces Latency"),
        Claim("ＣＡＣＨＥ", "REDUCES", "latency", "doc.md", 1, evidence="Ｃａｃｈｅ reduces Latency"),
    ]

    assert add_claims(graph, node_map, claims) == 1
    assert graph.number_of_edges() == 1
    assert {"Cache", "Latency"}.issubset(graph.nodes)


def test_exact_same_source_claim_deduplicates_across_overlap_chunk_indices():
    graph, node_map = create_graph("System")
    claims = [
        Claim("Cache", "reduces", "Latency", "doc.md", 7, evidence="Cache reduces Latency"),
        Claim("Cache", "reduces", "Latency", "doc.md", 8, evidence="Cache reduces Latency"),
    ]

    assert add_claims(graph, node_map, claims) == 1
    assert graph.number_of_edges() == 1
    data = next(iter(graph.edges(data=True)))[2]
    assert data["chunk_index"] == 7


def test_same_claim_on_distinct_pdf_pages_keeps_independent_provenance():
    graph, node_map = create_graph("System")
    claims = [
        Claim("Cache", "reduces", "Latency", "doc.pdf", 7, page=2, evidence="Cache reduces Latency"),
        Claim("Cache", "reduces", "Latency", "doc.pdf", 8, page=3, evidence="Cache reduces Latency"),
    ]

    assert add_claims(graph, node_map, claims) == 2
    assert graph.number_of_edges() == 2


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


def test_graph_admission_rejects_oversized_direct_claim_fields():
    graph, node_map = create_graph("System")
    oversized = [
        Claim("S" * (MAX_ENTITY_LABEL_CHARS + 1), "supports", "B", "doc.md", 1, evidence="evidence"),
        Claim("A", "r" * (MAX_RELATION_CHARS + 1), "B", "doc.md", 1, evidence="evidence"),
        Claim("A", "supports", "B", "s" * (MAX_SOURCE_ID_CHARS + 1), 1, evidence="evidence"),
        Claim("A", "supports", "B", "doc.md", 1, evidence="e" * (MAX_EVIDENCE_CHARS + 1)),
    ]

    assert add_claims(graph, node_map, oversized) == 0
    assert graph.number_of_edges() == 0


def test_node_capacity_fails_before_partial_graph_mutation(monkeypatch):
    monkeypatch.setattr(graph_module, "MAX_GRAPH_NODES", 2)
    graph, node_map = create_graph("System")

    assert add_claims(graph, node_map, [_claim("A", "supports", "System", "doc.md", 1)]) == 1
    with pytest.raises(GraphCapacityError, match="nodes"):
        add_claims(graph, node_map, [_claim("B", "supports", "System", "doc.md", 2)])

    assert set(graph.nodes) == {"System", "A"}
    assert "b" not in node_map
    assert graph.number_of_edges() == 1


def test_edge_capacity_fails_before_adding_new_endpoint(monkeypatch):
    monkeypatch.setattr(graph_module, "MAX_GRAPH_EDGES", 1)
    graph, node_map = create_graph("System")

    assert add_claims(graph, node_map, [_claim("A", "supports", "System", "doc.md", 1)]) == 1
    with pytest.raises(GraphCapacityError, match="edges"):
        add_claims(graph, node_map, [_claim("A", "supports", "B", "doc.md", 2)])

    assert "B" not in graph
    assert "b" not in node_map
    assert graph.number_of_edges() == 1


def test_synthetic_master_links_do_not_count_as_sources():
    graph, node_map = create_graph("System")
    add_claims(graph, node_map, [_claim("Cache", "reduces", "Latency", "perf.pdf", 1)])
    add_master_links(graph, node_map, "System", [("Cache", "includes")])
    export = graph_to_export(graph)
    assert export["stats"]["sources"] == 1
    assert export["sources"] == {"perf.pdf": 1}
    assert export["legacy_source_candidates"] == []


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


def test_migrated_legacy_sources_remain_visible_without_fabricated_claim_counts():
    graph = nx.MultiDiGraph()
    graph.add_node("A", type="master")
    graph.add_node("B", type="entity")
    for key, relation in (("legacy-1", "supports"), ("legacy-2", "depends on")):
        graph.add_edge(
            "A",
            "B",
            key=key,
            relation=relation,
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
    assert export["stats"]["legacy_claims"] == 2
    assert export["stats"]["sources"] == 2
    assert export["sources"] == {}
    assert export["legacy_source_candidates"] == ["notes.md", "paper.pdf"]
