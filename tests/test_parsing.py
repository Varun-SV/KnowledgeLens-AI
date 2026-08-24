from knowledgelens.models import DocumentChunk
from knowledgelens.parsing import parse_claims


def test_parse_structured_claims_preserves_provenance():
    chunk = DocumentChunk(source="paper.pdf", text="", chunk_index=7, page=3)
    claims = parse_claims(
        '[{"subject":"Attention","relation":"enables","object":"Parallelism","evidence":"Layers can run in parallel.","confidence":0.93}]',
        chunk,
    )
    assert len(claims) == 1
    claim = claims[0]
    assert claim.source == "paper.pdf"
    assert claim.page == 3
    assert claim.chunk_index == 7
    assert claim.confidence == 0.93


def test_parse_legacy_pipe_format_for_compatibility():
    chunk = DocumentChunk(source="notes.txt", text="", chunk_index=1)
    claims = parse_claims("Transformer | uses | Attention | supported here | 90", chunk)
    assert len(claims) == 1
    assert claims[0].relation == "uses"
    assert claims[0].confidence == 0.9
