from pathlib import Path


def test_extraction_prompt_requires_verbatim_source_evidence():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert '"evidence":"short verbatim source excerpt"' in app_source
    assert "Evidence MUST be a short verbatim excerpt copied from the supplied source text" in app_source
    assert "never paraphrase or invent evidence" in app_source


def test_state_upload_checks_streamlit_size_before_getvalue():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    helper = app_source[app_source.index("def _state_upload_payload") : app_source.index("def _state_payload_fingerprint")]

    assert "size > MAX_STATE_BYTES" in helper
    assert helper.index("size > MAX_STATE_BYTES") < helper.index("uploaded_state.getvalue()")


def test_fatal_ingestion_error_is_distinct_from_no_text_error():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    fatal_index = app_source.index("if ingest_result.fatal_error:")
    no_chunks_index = app_source.index("if not chunks:", fatal_index)
    assert fatal_index < no_chunks_index
    assert "st.error(ingest_result.fatal_error)" in app_source[fatal_index:no_chunks_index]


def test_graph_tooltips_use_html_escape_helper():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert 'title=safe_tooltip_text(f"{node}\\n{degree} connected graph edges")' in app_source
    assert "title=safe_tooltip_text(title)" in app_source
