from knowledgelens.limits import MAX_ENTITY_LABEL_CHARS, MAX_RELATION_CHARS, MAX_SOURCE_ID_CHARS
from knowledgelens.models import DocumentChunk
from knowledgelens.parsing import normalize_entity, parse_claims, parse_master_concept_response


def test_parse_json_claim_with_provenance():
    chunk = DocumentChunk(
        source="paper.pdf",
        text="Attention enables Parallel Training. Supported by the architecture.",
        chunk_index=3,
        page=7,
    )
    claims = parse_claims(
        '[{"subject":"Attention","relation":"enables","object":"Parallel Training","evidence":"Supported by the architecture.","confidence":0.9}]',
        chunk,
    )
    assert len(claims) == 1
    claim = claims[0]
    assert claim.source == "paper.pdf"
    assert claim.page == 7
    assert claim.confidence == 0.9


def test_parse_legacy_pipe_format():
    chunk = DocumentChunk(source="notes.md", text="Cache reduces latency; measured in tests.", chunk_index=2)
    claims = parse_claims("Cache | reduces | Latency | measured in tests | 95", chunk)
    assert claims[0].confidence == 0.95
    assert claims[0].evidence == "measured in tests"


def test_claims_without_supporting_evidence_are_rejected():
    chunk = DocumentChunk(source="notes.md", text="Cache reduces latency.", chunk_index=2)
    json_claims = parse_claims('[{"subject":"Cache","relation":"reduces","object":"Latency"}]', chunk)
    pipe_claims = parse_claims("Cache | reduces | Latency", chunk)
    assert json_claims == []
    assert pipe_claims == []


def test_invented_evidence_not_present_in_source_is_rejected():
    chunk = DocumentChunk(source="notes.md", text="Cache reduces latency under load.", chunk_index=2)
    claims = parse_claims(
        '[{"subject":"Cache","relation":"reduces","object":"Latency","evidence":"Cache eliminates all latency."}]',
        chunk,
    )
    assert claims == []


def test_verbatim_evidence_matching_tolerates_case_and_whitespace_only():
    chunk = DocumentChunk(source="notes.md", text="Cache reduces\n  latency under LOAD.", chunk_index=2)
    claims = parse_claims(
        '[{"subject":"Cache","relation":"reduces","object":"Latency","evidence":"cache reduces latency under load."}]',
        chunk,
    )
    assert len(claims) == 1


def test_relation_stopwords_are_not_rejected_as_entity_stopwords():
    chunk = DocumentChunk(source="topology.md", text="The source says so for every topology relation.", chunk_index=1)
    for relation in ("in", "on", "to", "for"):
        claims = parse_claims(
            f'[{{"subject":"Service","relation":"{relation}","object":"Region","evidence":"source says so"}}]',
            chunk,
        )
        assert len(claims) == 1
        assert claims[0].relation == relation


def test_successfully_decoded_non_claim_json_is_not_reinterpreted_as_pipe_output():
    chunk = DocumentChunk(source="notes.md", text="A relates to B.", chunk_index=1)
    claims = parse_claims('{"message":"A | relates | B | looks like evidence"}', chunk)
    assert claims == []


def test_invalid_confidence_values_are_treated_as_unknown():
    chunk = DocumentChunk(source="notes.md", text="The source says so about Alpha and Beta.", chunk_index=1)
    values = ("101", "-1", '"Infinity"', '"NaN"')
    for value in values:
        claims = parse_claims(
            f'[{{"subject":"Alpha","relation":"supports","object":"Beta","evidence":"source says so","confidence":{value}}}]',
            chunk,
        )
        assert len(claims) == 1
        assert claims[0].confidence is None


def test_boolean_confidence_values_are_treated_as_unknown():
    chunk = DocumentChunk(source="notes.md", text="Alpha supports Beta according to the source.", chunk_index=1)
    for value in ("true", "false"):
        claims = parse_claims(
            f'[{{"subject":"Alpha","relation":"supports","object":"Beta","evidence":"Alpha supports Beta","confidence":{value}}}]',
            chunk,
        )
        assert len(claims) == 1
        assert claims[0].confidence is None


def test_model_controlled_labels_and_sources_are_bounded():
    chunk = DocumentChunk(source="doc.md", text="Alpha supports Beta.", chunk_index=1)
    oversized_entity = "A" * (MAX_ENTITY_LABEL_CHARS + 1)
    oversized_relation = "r" * (MAX_RELATION_CHARS + 1)

    assert parse_claims(
        f'[{{"subject":"{oversized_entity}","relation":"supports","object":"Beta","evidence":"Alpha supports Beta"}}]',
        chunk,
    ) == []
    assert parse_claims(
        f'[{{"subject":"Alpha","relation":"{oversized_relation}","object":"Beta","evidence":"Alpha supports Beta"}}]',
        chunk,
    ) == []

    oversized_source = DocumentChunk(
        source="s" * (MAX_SOURCE_ID_CHARS + 1),
        text="Alpha supports Beta.",
        chunk_index=1,
    )
    assert parse_claims(
        '[{"subject":"Alpha","relation":"supports","object":"Beta","evidence":"Alpha supports Beta"}]',
        oversized_source,
    ) == []


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


def test_master_concept_parser_rejects_oversized_model_label():
    assert parse_master_concept_response("M" * (MAX_ENTITY_LABEL_CHARS + 1)) == ""


def test_markdown_fence_stripping_handles_large_whitespace_without_regex():
    chunk = DocumentChunk(source="x.md", text="GPU accelerates AI workloads.", chunk_index=1)
    payload = (
        '```json\n{"subject":"AI","relation":"uses","object":"GPU","evidence":"GPU accelerates AI workloads"}\n```'
        + (" " * 100_000)
    )
    claims = parse_claims(payload, chunk)
    assert len(claims) == 1
