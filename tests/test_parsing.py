from knowledgelens.models import DocumentChunk
from knowledgelens.parsing import normalize_entity, parse_claims, parse_master_concept_response


def test_parse_json_claim_with_provenance():
    chunk = DocumentChunk(source="paper.pdf", text="x", chunk_index=3, page=7)
    claims = parse_claims(
        '[{"subject":"Attention","relation":"enables","object":"Parallel Training","evidence":"Supported","confidence":0.9}]',
        chunk,
    )
    assert len(claims) == 1
    claim = claims[0]
    assert claim.source == "paper.pdf"
    assert claim.page == 7
    assert claim.confidence == 0.9


def test_parse_legacy_pipe_format():
    chunk = DocumentChunk(source="notes.md", text="x", chunk_index=2)
    claims = parse_claims("Cache | reduces | Latency | measured in tests | 95", chunk)
    assert claims[0].confidence == 0.95
    assert claims[0].evidence == "measured in tests"


def test_unicode_entity_normalization_is_preserved():
    assert normalize_entity("人工知能")[0] == "人工知能"
    assert normalize_entity("Привет Мир")[0] == "привет мир"
    assert normalize_entity("الذكاء الاصطناعي")[0] == "الذكاء الاصطناعي"


def test_identifier_significant_punctuation_remains_distinct():
    assert normalize_entity("C")[0] == "c"
    assert normalize_entity("C++")[0] == "c++"
    assert normalize_entity("C#")[0] == "c#"
    assert len({normalize_entity(value)[0] for value in ("C", "C++", "C#")}) == 3
    assert normalize_entity(".NET")[0] == ".net"
    assert normalize_entity("Node.js")[0] == "node.js"
    assert normalize_entity("Attention.")[0] == "attention"


def test_master_concept_parser_accepts_markdown_fences_and_blank_lines():
    assert parse_master_concept_response("```\nMachine Learning\n```") == "Machine Learning"
    assert parse_master_concept_response("```text\n\nDistributed Systems\n```") == "Distributed Systems"
    assert parse_master_concept_response("\n`Knowledge Graphs`\n") == "Knowledge Graphs"


def test_master_concept_parser_strips_common_explanatory_prefixes():
    assert parse_master_concept_response("The central concept is Machine Learning") == "Machine Learning"
    assert parse_master_concept_response("Topic: Knowledge Graphs") == "Knowledge Graphs"
    assert parse_master_concept_response("Master concept: Distributed Systems") == "Distributed Systems"
    assert parse_master_concept_response("```text\nTopic: Retrieval Augmented Generation\n```") == "Retrieval Augmented Generation"


def test_markdown_fence_stripping_handles_large_whitespace_without_regex():
    chunk = DocumentChunk(source="x.md", text="x", chunk_index=1)
    payload = '```json\n{"subject":"AI","relation":"uses","object":"GPU"}\n```' + (" " * 100_000)
    claims = parse_claims(payload, chunk)
    assert len(claims) == 1
