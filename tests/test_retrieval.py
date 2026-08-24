from knowledgelens.graph import add_claims, create_graph
from knowledgelens.models import Claim
from knowledgelens.retrieval import retrieve_graph_context


def test_retrieval_includes_evidence_and_source_location():
    graph, node_map = create_graph("Transformers")
    add_claims(
        graph,
        node_map,
        [
            Claim(
                subject="Attention",
                relation="enables",
                object="Parallel Training",
                source="paper.pdf",
                chunk_index=2,
                page=4,
                evidence="Attention removes recurrent dependencies.",
                confidence=0.9,
            )
        ],
    )
    context = retrieve_graph_context(graph, "How does Attention enable parallel training?")
    assert "paper.pdf · p.4" in context
    assert "Attention --[enables]--> Parallel Training" in context
    assert "Attention removes recurrent dependencies" in context
