import networkx as nx

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


def test_short_entity_does_not_match_inside_unrelated_words():
    assert score_node("explain the details", "AI") < 3.0
    assert score_node("explain AI details", "AI") >= 3.0


def test_mixed_direction_path_is_retrieved():
    graph = nx.MultiDiGraph()
    for node in ("A", "X", "Y", "Z", "B"):
        graph.add_node(node)
    _edge(graph, "X", "A", "xa", "points to")
    _edge(graph, "X", "Y", "xy", "connects")
    _edge(graph, "Z", "Y", "zy", "supports")
    _edge(graph, "Z", "B", "zb", "points to")
    context = retrieve_graph_context(graph, "Compare A and B")
    assert "[graph path]" in context
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

    assert "[graph path] Alpha -- Bridge -- Beta" in context
    assert "Alpha --[connects]--> Bridge" in context
    assert "Bridge --[supports]--> Beta" in context
    assert len(context) <= 500
    assert not context.endswith("...")
