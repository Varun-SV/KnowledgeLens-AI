from pathlib import Path


def test_extraction_prompt_requires_verbatim_source_evidence():
    runtime_source = Path("knowledgelens/runtime.py").read_text(encoding="utf-8")
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert '"evidence":"short verbatim source excerpt"' in runtime_source
    assert "Evidence MUST be a short verbatim excerpt copied from source_text" in runtime_source
    assert "never paraphrase or invent evidence" in runtime_source
    assert "extraction_messages(chunk.citation, chunk.text, custom_focus)" in app_source


def test_source_text_and_master_detection_use_data_only_prompt_helpers():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    assert "master_detection_messages(excerpts)" in app_source
    assert "extraction_messages(chunk.citation, chunk.text, custom_focus)" in app_source
    assert 'f"Source: {chunk.citation}\\n\\n{chunk.text}"' not in app_source


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


def test_visualization_budget_is_checked_before_pyvis_network_creation():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    helper = app_source[app_source.index("def visualize_graph") : app_source.index("def _state_upload_payload")]

    guard = helper.index("render_error = visualization_limit_error")
    network = helper.index("network = Network(")
    assert guard < network
    assert "st.warning(render_error)" in helper[guard:network]
    assert "return" in helper[guard:network]


def test_parallel_edges_use_shared_unique_curve_helper():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    assert "parallel_edge_smooth" in app_source
    assert "smooth=parallel_edge_smooth(edge_index, edge_totals[pair])" in app_source
    assert "def _parallel_edge_smooth" not in app_source


def test_manual_master_and_extraction_focus_limits_are_wired_before_graph_creation():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    assert "max_chars=MAX_ENTITY_LABEL_CHARS" in app_source
    assert "max_chars=MAX_EXTRACTION_FOCUS_CHARS" in app_source
    master_validation = app_source.index("master_error = manual_master_concept_error(auto_detect, manual_master)")
    focus_validation = app_source.index("focus_error = extraction_focus_error(custom_focus)")
    graph_creation = app_source.index("graph, node_map = create_graph(master)")
    assert master_validation < graph_creation
    assert focus_validation < graph_creation


def test_api_key_limit_is_wired_into_password_widget_and_shared_preflight():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    runtime_source = Path("knowledgelens/runtime.py").read_text(encoding="utf-8")

    api_widget = app_source[app_source.index('api_key = st.text_input(') : app_source.index('default_model =', app_source.index('api_key = st.text_input('))]
    assert "max_chars=MAX_API_KEY_CHARS" in api_widget
    assert "len(api_key) > MAX_API_KEY_CHARS" in runtime_source
    assert "request_configuration_error(provider, api_key, model_name)" in app_source


def test_graph_capacity_failure_stops_build_before_session_commit():
    app_source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")

    loop_start = app_source.index("for index, chunk in enumerate(chunks, start=1):")
    loop_end = app_source.index("if graph.number_of_nodes() > 1:", loop_start)
    loop = app_source[loop_start:loop_end]
    request_handler = loop.index("except Exception as exc:")
    capacity_handler = loop.index("except GraphCapacityError as exc:")
    session_commit = app_source.index("st.session_state.kg_graph = graph", loop_end)

    assert request_handler < capacity_handler
    assert capacity_handler < session_commit
    capacity_end = loop.index("progress.progress(index / len(chunks))", capacity_handler)
    assert "st.stop()" in loop[capacity_handler:capacity_end]


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
