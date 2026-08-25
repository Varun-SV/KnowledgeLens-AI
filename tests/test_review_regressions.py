import json
from pathlib import Path

import pytest

from knowledgelens.graph import add_claims, create_graph
from knowledgelens.models import Claim, DocumentChunk
from knowledgelens.parsing import parse_claims
from knowledgelens.persistence import deserialize_graph_state
from knowledgelens.resilience import RequestFailureCircuit

_NODE_LINK_FIELDS = {
    "source": "__kl_from",
    "target": "__kl_to",
    "name": "__kl_id",
    "key": "__kl_key",
}
_NODE_LINK_META = "_knowledgelens_node_link_fields"


def _valid_edge(relation: str) -> dict:
    return {
        "__kl_from": "Alpha",
        "__kl_to": "Beta",
        "__kl_key": "same-key",
        "relation": relation,
        "source": "doc.md",
        "legacy_sources": [],
        "page": None,
        "chunk_index": 1,
        "evidence": "Alpha supports Beta.",
        "confidence": 0.9,
        "synthetic": False,
        "provenance_status": None,
    }


def test_repeated_evidence_in_overlap_and_new_suffix_is_preserved():
    evidence = "Cache reduces latency."
    chunk = DocumentChunk(
        source="notes.md",
        text=f"{evidence} New material appears here. {evidence}",
        chunk_index=8,
        overlap_from_previous=True,
        overlap_prefix=evidence,
    )
    parsed = parse_claims(
        '[{"subject":"Cache","relation":"reduces","object":"Latency","evidence":"Cache reduces latency."}]',
        chunk,
    )

    assert len(parsed) == 1
    assert parsed[0].overlap_from_previous is False

    graph, node_map = create_graph("System")
    previous = Claim(
        subject="Cache",
        relation="reduces",
        object="Latency",
        source="notes.md",
        chunk_index=7,
        evidence=evidence,
        confidence=0.9,
    )
    assert add_claims(graph, node_map, [previous]) == 1
    assert add_claims(graph, node_map, parsed) == 1
    assert graph.number_of_edges("Cache", "Latency") == 2


def test_loader_rejects_duplicate_serialized_node_ids_before_networkx_coalesces_them():
    graph_data = {
        "directed": True,
        "multigraph": True,
        "nodes": [
            {"__kl_id": "Alpha", "type": "master"},
            {"__kl_id": "Alpha", "type": "entity"},
        ],
        "edges": [],
        _NODE_LINK_META: dict(_NODE_LINK_FIELDS),
    }
    raw = json.dumps({"schema_version": 2, "master_concept": "Alpha", "graph_data": graph_data})

    with pytest.raises(ValueError, match="duplicate serialized node"):
        deserialize_graph_state(raw)


def test_loader_rejects_duplicate_serialized_multiedge_identity_before_data_loss():
    graph_data = {
        "directed": True,
        "multigraph": True,
        "nodes": [
            {"__kl_id": "Alpha", "type": "master"},
            {"__kl_id": "Beta", "type": "entity"},
        ],
        "edges": [_valid_edge("supports"), _valid_edge("contradicts")],
        _NODE_LINK_META: dict(_NODE_LINK_FIELDS),
    }
    raw = json.dumps({"schema_version": 2, "master_concept": "Alpha", "graph_data": graph_data})

    with pytest.raises(ValueError, match="duplicate serialized edge"):
        deserialize_graph_state(raw)


def test_request_failure_circuit_opens_and_success_resets_streak():
    circuit = RequestFailureCircuit(limit=3)

    assert circuit.record_failure() is False
    assert circuit.record_failure() is False
    circuit.record_success()
    assert circuit.consecutive_failures == 0
    assert circuit.record_failure() is False
    assert circuit.record_failure() is False
    assert circuit.record_failure() is True
    assert circuit.consecutive_failures == 3


def test_app_wires_failure_circuit_without_treating_graph_admission_as_request_failure():
    source = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    loop = source[source.index("for index, chunk in enumerate(chunks, start=1):") : source.index("if graph.number_of_nodes() > 1:")]

    request_call = loop.index("claims = extract_chunk_claims")
    request_failure = loop.index("circuit_open = request_circuit.record_failure()")
    success_reset = loop.index("request_circuit.record_success()")
    graph_admission = loop.index("add_claims(graph, node_map, claims)")
    assert request_call < request_failure < success_reset < graph_admission
    assert "if circuit_open:" in loop
    assert "st.stop()" in loop


def test_streamlit_rejects_oversized_files_at_widget_and_server_boundaries():
    app = Path("KnowledgeLens_AI.py").read_text(encoding="utf-8")
    config = Path(".streamlit/config.toml").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    state_widget = app[app.index('uploaded_state = st.file_uploader(') : app.index("try:", app.index('uploaded_state = st.file_uploader('))]
    document_widget = app[app.index('uploaded_files = st.file_uploader(') : app.index('if st.button("Build evidence graph"')]

    assert "max_upload_size=MAX_STATE_BYTES // (1024 * 1024)" in state_widget
    assert "max_upload_size=DEFAULT_INGESTION_LIMITS.max_upload_bytes // (1024 * 1024)" in document_widget
    assert "maxUploadSize = 24" in config
    assert "maxUploadSize = 200" not in config
    assert '"streamlit>=1.54,<2"' in pyproject
    assert "streamlit>=1.54,<2" in requirements
