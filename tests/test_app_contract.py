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


def test_manual_master_limit_is_wired_before_graph_creation():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert "max_chars=MAX_ENTITY_LABEL_CHARS" in app_source
    validation = app_source.index("master_error = manual_master_concept_error(auto_detect, manual_master)")
    graph_creation = app_source.index("graph, node_map = create_graph(master)")
    assert validation < graph_creation


def test_state_export_failure_is_rendered_instead_of_crashing_page():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    export_start = app_source.index("state_export = serialize_graph_state(")
    warning = app_source.index("Graph state export unavailable:", export_start)
    download = app_source.index('"Graph state",', export_start)
    assert export_start < warning < download


def test_graph_chat_uses_hardened_data_only_prompt_boundary():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    helper_call = "system_prompt, user_prompt = grounded_chat_messages(context, user_query)"
    assert helper_call in app_source
    chat_start = app_source.index(helper_call)
    call_start = app_source.index("answer = call_llm_api(", chat_start)
    call_end = app_source.index("temperature,", call_start)
    call = app_source[call_start:call_end]
    assert "system_prompt" in call
    assert "user_prompt" in call
    assert "Graph context:" not in call
