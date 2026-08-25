import networkx as nx

import knowledgelens.retrieval as retrieval
from knowledgelens.retrieval import relevant_nodes, retrieve_graph_context, score_node


def _edge(graph, a, b, key, relation, evidence="evidence"):
    graph.add_edge(
        a,
        b,
        key=key,
        relation=relation,
        source="doc.md",
        page=None,
        chunk_index=1,
        evidence=evidence,
        confidence=0.9,
        synthetic=False,
    )


def test_retrieval_includes_source_citation():
    graph = nx.MultiDiGraph()
    graph.add_node("Cache")
    graph.add_node("Latency")
    _edge(graph, "Cache", "Latency", "1", "reduces")
    context = retrieve_graph_context(graph, "How does Cache affect Latency?")
    assert "doc.md" in context
    assert "Cache --[reduces]--> Latency" in context


def test_retrieval_preserves_integer_and_string_multiedge_keys_as_distinct_claims():
    graph = nx.MultiDiGraph()
    graph.add_node("Cache")
    graph.add_node("Latency")
    _edge(graph, "Cache", "Latency", 1, "reduces", evidence="Cache reduces latency.")
    _edge(graph, "Cache", "Latency", "1", "improves", evidence="Cache improves response time.")

    context = retrieve_graph_context(graph, "Explain Cache")

    assert "Cache --[reduces]--> Latency" in context
    assert "Cache --[improves]--> Latency" in context
    assert "Cache reduces latency." in context
    assert "Cache improves response time." in context


def test_short_entity_does_not_match_inside_unrelated_words():
    assert score_node("explain the details", "AI") < 3.0
    assert score_node("explain AI details", "AI") >= 3.0


def test_stopwords_do_not_create_overlap_matches_by_themselves():
    assert score_node("summarize the evidence", "The API") < 0.6
    assert score_node("summarize the evidence", "The Cache") < 0.6
    assert score_node("explain the API", "The API") >= 3.0


def test_retrieval_scoring_preserves_identifier_punctuation():
    query = "Compare C++ performance with Rust"
    assert score_node(query, "C++") > score_node(query, "C")
    assert score_node(query, "C++") > score_node(query, "C#")

    graph = nx.MultiDiGraph()
    for node in ("C", "C++", "C#", "Rust"):
        graph.add_node(node)
    _edge(graph, "C++", "Rust", "cpp-rust", "compared with")
    _edge(graph, "C", "Memory", "c-memory", "manages")
    _edge(graph, "C#", ".NET", "cs-dotnet", "targets")

    assert relevant_nodes(graph, "Explain C++")[:1] == ["C++"]


def test_large_graph_retrieval_bounds_expensive_fuzzy_matching(monkeypatch):
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(f"Entity {index}" for index in range(10_000))
    calls: list[tuple[str, str]] = []

    class CountingMatcher:
        def __init__(self, _isjunk, left, right):
            calls.append((left, right))
            assert len(left) <= retrieval._MAX_FUZZY_QUERY_CHARS
            assert len(right) <= retrieval._MAX_FUZZY_NODE_CHARS

        def ratio(self):
            return 0.0

    monkeypatch.setattr(retrieval, "SequenceMatcher", CountingMatcher)

    relevant_nodes(graph, "x" * 4_000)

    assert len(calls) <= retrieval._MAX_FUZZY_CANDIDATES


def test_mixed_direction_path_is_retrieved():
    graph = nx.MultiDiGraph()
    for node in ("A", "X", "Y", "Z", "B"):
        graph.add_node(node)
    _edge(graph, "X", "A", "xa", "points to")
    _edge(graph, "X", "Y", "xy", "connects")
    _edge(graph, "Z", "Y", "zy", "supports")
    _edge(graph, "Z", "B", "zb", "points to")
    context = retrieve_graph_context(graph, "Compare A and B")
    assert "Graph path:" in context
    assert "X --[points to]--> A" in context
    assert "Z --[points to]--> B" in context


def test_synthetic_overview_links_are_not_grounded_evidence():
    graph = nx.MultiDiGraph()
    graph.add_node("Knowledge Base", type="master")
    graph.add_node("Topic")
    graph.add_edge(
        "Knowledge Base",
        "Topic",
        key="synthetic",
        relation="includes",
        source="KnowledgeLens",
        page=None,
        chunk_index=0,
        evidence="Synthetic overview link generated from graph importance.",
        confidence=None,
        synthetic=True,
    )

    context = retrieve_graph_context(graph, "Knowledge Base")

    assert "KnowledgeLens" not in context
    assert "includes" not in context
    assert "No specific source-backed graph connections" in context


def test_legacy_aggregated_relations_are_not_grounded_evidence():
    graph = nx.MultiDiGraph()
    graph.add_node("Legacy Topic", type="master")
    graph.add_node("Old Fact")
    graph.add_edge(
        "Legacy Topic",
        "Old Fact",
        key="legacy",
        relation="claims",
        source="",
        legacy_sources=["old.pdf"],
        page=None,
        chunk_index=0,
        evidence="Migration note only; original evidence was not preserved.",
        confidence=None,
        synthetic=False,
        provenance_status="legacy-aggregated",
    )

    context = retrieve_graph_context(graph, "Legacy Topic")

    assert "old.pdf" not in context
    assert "--[claims]-->" not in context
    assert "No specific source-backed graph connections" in context


def test_master_query_routes_through_topology_to_evidence_bearing_neighbors():
    graph = nx.MultiDiGraph()
    graph.add_node("Knowledge Base", type="master")
    graph.add_node("Cache")
    graph.add_node("Latency")
    graph.add_edge(
        "Knowledge Base",
        "Cache",
        key="synthetic",
        relation="includes",
        source="KnowledgeLens",
        chunk_index=0,
        synthetic=True,
    )
    _edge(graph, "Cache", "Latency", "real", "reduces")

    seeds = relevant_nodes(graph, "Summarize Knowledge Base")
    context = retrieve_graph_context(graph, "Summarize Knowledge Base")

    assert seeds[0] == "Cache"
    assert "Cache --[reduces]--> Latency" in context
    assert "KnowledgeLens" not in context
    assert "--[includes]-->" not in context


def test_master_expansion_reserves_seed_for_other_explicit_entity():
    graph = nx.MultiDiGraph()
    graph.add_node("Knowledge Base", type="master")
    for index, node in enumerate(("Cache", "Latency", "Auth", "Queue", "Storage")):
        graph.add_edge(
            "Knowledge Base",
            node,
            key=f"synthetic-{index}",
            relation="includes",
            source="KnowledgeLens",
            chunk_index=0,
            synthetic=True,
        )
        _edge(graph, node, f"Leaf {index}", f"real-{index}", "supports")
    _edge(graph, "Remote Entity", "Remote Leaf", "remote", "connects")

    seeds = relevant_nodes(graph, "Compare Knowledge Base and Remote Entity", limit=5)

    assert "Remote Entity" in seeds
    assert len(seeds) == 5


def test_master_is_removed_before_direct_match_limit_is_applied():
    graph = nx.MultiDiGraph()
    graph.add_node("Knowledge Base", type="master")
    direct = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    for index, node in enumerate(direct):
        _edge(graph, node, f"{node} Leaf", f"direct-{index}", "supports")
    graph.add_edge(
        "Knowledge Base",
        "Master Neighbor",
        key="synthetic-master",
        relation="includes",
        source="KnowledgeLens",
        chunk_index=0,
        synthetic=True,
    )
    _edge(graph, "Master Neighbor", "Master Leaf", "master-real", "supports")

    query = "Knowledge Base Knowledge Base compare Alpha Beta Gamma Delta Epsilon"
    seeds = relevant_nodes(graph, query, limit=5)

    assert set(seeds) == set(direct)


def test_generic_query_falls_back_to_evidence_bearing_node():
    graph = nx.MultiDiGraph()
    graph.add_node("Knowledge Base", type="master")
    graph.add_node("Topic")
    graph.add_edge(
        "Knowledge Base",
        "Topic",
        key="synthetic",
        relation="includes",
        source="KnowledgeLens",
        chunk_index=0,
        synthetic=True,
    )
    _edge(graph, "Cache", "Latency", "real", "reduces")

    seeds = relevant_nodes(graph, "summarize the evidence")
    context = retrieve_graph_context(graph, "summarize the evidence")

    assert seeds[0] in {"Cache", "Latency"}
    assert "Cache --[reduces]--> Latency" in context
    assert "KnowledgeLens" not in context


def test_context_budget_prioritizes_selected_entity_path_before_dense_neighborhood():
    graph = nx.MultiDiGraph()
    _edge(graph, "Alpha", "Bridge", "ab", "connects")
    _edge(graph, "Bridge", "Beta", "bb", "supports")
    for index in range(12):
        _edge(
            graph,
            "Alpha",
            f"Noise {index}",
            f"noise-{index}",
            "mentions",
            evidence="x" * 180,
        )

    context = retrieve_graph_context(graph, "Compare Alpha and Beta", max_chars=500)

    assert "Graph path: Alpha -- Bridge -- Beta" in context
    assert "Alpha --[connects]--> Bridge" in context
    assert "Bridge --[supports]--> Beta" in context
    assert len(context) <= 500
    assert not context.endswith("...")


def test_path_header_is_omitted_when_no_supporting_claim_fits_budget():
    graph = nx.MultiDiGraph()
    _edge(graph, "Alpha", "Beta", "ab", "connects", evidence="supported by a source")

    header = "Graph path: Alpha -- Beta"
    context = retrieve_graph_context(graph, "Compare Alpha and Beta", max_chars=len(header))

    assert header not in context
    assert "No specific source-backed graph connections" in context
