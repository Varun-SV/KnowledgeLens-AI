from knowledgelens.graph import add_claims, create_graph, graph_to_export
from knowledgelens.models import Claim


def test_master_node_cannot_be_downgraded_by_extracted_claim():
    graph, node_map = create_graph("Machine Learning")
    add_claims(
        graph,
        node_map,
        [
            Claim(
                subject="Machine Learning",
                relation="uses",
                object="Optimization",
                source="paper.pdf",
                page=1,
                chunk_index=1,
            )
        ],
    )
    assert graph.nodes["Machine Learning"]["type"] == "master"


def test_parallel_claims_keep_per_source_provenance():
    graph, node_map = create_graph("System Design")
    claims = [
        Claim("Cache", "reduces", "Latency", "a.pdf", 1, page=2, evidence="cache lowers latency"),
        Claim("Cache", "increases", "Staleness", "b.pdf", 4, page=8, evidence="cache can become stale"),
    ]
    assert add_claims(graph, node_map, claims) == 2
    assert graph.number_of_edges("Cache", "Latency") == 1
    assert graph.number_of_edges("Cache", "Staleness") == 1

    exported = graph_to_export(graph)
    assert {item["source"] for item in exported["claims"]} == {"a.pdf", "b.pdf"}
