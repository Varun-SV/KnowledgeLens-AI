from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from knowledgelens.graph import add_claims, add_master_links, create_graph, graph_to_export, top_entities
from knowledgelens.http_client import PinnedRequestError, post_json_pinned
from knowledgelens.ingestion import prepare_chunks
from knowledgelens.models import DocumentChunk
from knowledgelens.parsing import parse_claims, parse_master_concept_response
from knowledgelens.persistence import deserialize_graph_state, serialize_graph_state
from knowledgelens.retrieval import retrieve_graph_context
from knowledgelens.runtime import no_claims_build_error, provider_state_key, request_configuration_error
from knowledgelens.security import EndpointPolicyError, env_flag, resolve_endpoint, validate_endpoint

APP_VERSION = "0.2.0"


def configured_endpoint(provider: str) -> str:
    """Resolve a user-selected provider to an operator-controlled endpoint."""
    if provider == "Ollama / local":
        return "http://localhost:11434"
    if provider == "llama.cpp / local":
        return "http://localhost:8080"
    if provider == "OpenAI":
        return "https://api.openai.com"

    configured = os.getenv("KNOWLEDGELENS_CUSTOM_ENDPOINT", "").strip()
    if provider == "Configured endpoint" and configured:
        return configured
    raise EndpointPolicyError("No endpoint is configured for this provider.")


def call_llm_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> str:
    endpoint = resolve_endpoint(base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": temperature,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    try:
        status_code, response_body = post_json_pinned(endpoint, payload, headers)
    except PinnedRequestError as exc:
        raise RuntimeError(f"Could not reach the LLM endpoint: {exc}") from exc

    if 300 <= status_code < 400:
        raise RuntimeError("The LLM endpoint returned a redirect. Redirects are blocked for credential safety.")
    if status_code >= 400:
        detail = response_body.decode("utf-8", errors="replace")[:400]
        raise RuntimeError(f"LLM returned HTTP {status_code}: {detail}")

    try:
        response_payload = json.loads(response_body)
        return str(response_payload["choices"][0]["message"]["content"])
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The endpoint returned an unexpected OpenAI-compatible response.") from exc


def detect_master_concept(
    base_url: str,
    api_key: str,
    model: str,
    chunks: list[DocumentChunk],
    temperature: float,
) -> str:
    samples: list[str] = []
    per_source: dict[str, int] = {}
    for chunk in chunks:
        if per_source.get(chunk.source, 0) >= 2:
            continue
        per_source[chunk.source] = per_source.get(chunk.source, 0) + 1
        samples.append(f"--- {chunk.citation} ---\n{chunk.text[:1200]}")
        if sum(map(len, samples)) > 7000:
            break

    response = call_llm_api(
        base_url,
        api_key,
        model,
        (
            "Identify the single central concept shared by the supplied document excerpts. "
            "Return only a precise 2-5 word noun phrase. No explanation, punctuation, or prefix."
        ),
        "\n\n".join(samples),
        temperature,
    )
    concept = parse_master_concept_response(response)
    if not 2 <= len(concept) <= 80:
        return "Knowledge Base"
    return concept


def extract_chunk_claims(
    base_url: str,
    api_key: str,
    model: str,
    chunk: DocumentChunk,
    temperature: float,
    custom_focus: str,
):
    system_prompt = """You extract auditable knowledge-graph claims from source text.
Return ONLY valid JSON, ideally an array. Each item must use this schema:
{"subject":"...","relation":"...","object":"...","evidence":"short source-supported evidence","confidence":0.0}
Rules:
- Extract only claims supported by the supplied text.
- Keep entities specific and reusable across documents.
- Use concise relation phrases.
- Evidence should be a short paraphrase or brief excerpt, never invented.
- Confidence must be 0 to 1.
- Do not emit markdown fences or commentary.
"""
    if custom_focus.strip():
        system_prompt += f"\nExtraction focus supplied by the user: {custom_focus.strip()}\n"

    response = call_llm_api(
        base_url,
        api_key,
        model,
        system_prompt,
        f"Source: {chunk.citation}\n\n{chunk.text}",
        temperature,
    )
    return parse_claims(response, chunk)


def generate_master_relations(
    base_url: str,
    api_key: str,
    model: str,
    master: str,
    nodes: list[str],
    temperature: float,
) -> list[tuple[str, str]]:
    if not nodes:
        return []
    response = call_llm_api(
        base_url,
        api_key,
        model,
        "Return only JSON object mapping each supplied entity to a short relation from the master concept.",
        f"Master concept: {master}\nEntities: {json.dumps(nodes, ensure_ascii=False)}",
        temperature,
    )
    try:
        payload = json.loads(response.strip().strip("`"))
    except json.JSONDecodeError:
        payload = {}
    links = []
    for node in nodes:
        relation = payload.get(node, "relates to") if isinstance(payload, dict) else "relates to"
        links.append((node, " ".join(str(relation).split())[:80] or "relates to"))
    return links


def _parallel_edge_smooth(index: int, total: int) -> dict[str, object]:
    if total <= 1:
        return {"enabled": True, "type": "continuous"}
    rank = index // 2
    return {
        "enabled": True,
        "type": "curvedCW" if index % 2 == 0 else "curvedCCW",
        "roundness": min(0.12 + rank * 0.08, 0.48),
    }


def visualize_graph(graph: nx.MultiDiGraph, height: int = 760) -> None:
    if graph.number_of_nodes() == 0:
        st.info("The graph is empty.")
        return

    network = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#070B12",
        font_color="#EAF2FF",
        directed=True,
    )
    degrees = dict(graph.degree())
    max_degree = max(max(degrees.values(), default=0), 1)

    for node, data in graph.nodes(data=True):
        degree = degrees.get(node, 0)
        is_master = data.get("type") == "master"
        size = 42 if is_master else 14 + 22 * (degree / max_degree)
        network.add_node(
            node,
            label=str(node),
            title=f"{node}\n{degree} connected graph edges",
            shape="star" if is_master else "dot",
            size=size,
            color="#FFB45C" if is_master else "#68E1FD",
            font={"size": 21 if is_master else 13, "face": "Inter, Arial", "color": "#F5F8FF"},
            borderWidth=2 if is_master else 1,
            borderWidthSelected=4,
        )

    edge_totals: dict[frozenset[object], int] = defaultdict(int)
    for subject, obj in graph.edges():
        edge_totals[frozenset((subject, obj))] += 1
    edge_positions: dict[frozenset[object], int] = defaultdict(int)

    for subject, obj, _key, data in graph.edges(keys=True, data=True):
        pair = frozenset((subject, obj))
        edge_index = edge_positions[pair]
        edge_positions[pair] += 1

        relation = str(data.get("relation", "related to"))
        source = str(data.get("source") or "")
        legacy_sources = [str(item) for item in data.get("legacy_sources", []) if item]
        if not source and legacy_sources:
            source = "legacy candidates: " + ", ".join(legacy_sources)
        source = source or "unknown source"
        page = data.get("page")
        chunk = data.get("chunk_index")
        location = f"p.{page}" if page is not None else f"chunk {chunk}"
        evidence = str(data.get("evidence") or "")
        title = f"{relation}\nSource: {source} · {location}"
        if evidence:
            title += f"\nEvidence: {evidence[:300]}"
        if data.get("synthetic"):
            title += "\nSynthetic overview link — not used as grounded chat evidence."
        elif data.get("provenance_status") == "legacy-aggregated":
            title += "\nLegacy aggregated relation — original source pairing/evidence was not preserved; excluded from grounded chat."

        network.add_edge(
            subject,
            obj,
            label=relation[:18] + ("…" if len(relation) > 18 else ""),
            title=title,
            color="rgba(126, 145, 180, 0.42)",
            arrows="to",
            smooth=_parallel_edge_smooth(edge_index, edge_totals[pair]),
        )

    network.set_options(
        json.dumps(
            {
                "physics": {
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -95,
                        "centralGravity": 0.006,
                        "springLength": 210,
                        "springConstant": 0.045,
                    },
                    "stabilization": {"enabled": True, "iterations": 180},
                },
                "interaction": {
                    "hover": True,
                    "tooltipDelay": 120,
                    "multiselect": True,
                    "navigationButtons": True,
                    "keyboard": True,
                    "zoomView": True,
                    "dragView": True,
                    "dragNodes": True,
                },
            }
        )
    )

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as handle:
            temp_path = handle.name
        network.save_graph(temp_path)
        html_content = Path(temp_path).read_text(encoding="utf-8")
        components.html(html_content, height=height + 35, scrolling=False)
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


def _state_upload_fingerprint(uploaded_state) -> str | None:
    if uploaded_state is None:
        return None
    return hashlib.sha256(uploaded_state.getvalue()).hexdigest()


st.set_page_config(page_title="KnowledgeLens AI", page_icon="◉", layout="wide")
st.markdown(
    """
<style>
:root { --kl-border: rgba(120, 150, 190, .18); }
.block-container { padding-top: 2rem; max-width: 1500px; }
[data-testid="stMetric"] { border: 1px solid var(--kl-border); border-radius: 14px; padding: 12px 14px; }
.kl-kicker { letter-spacing: .16em; text-transform: uppercase; font-size: .72rem; opacity: .62; }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="kl-kicker">Evidence graph workspace</div>', unsafe_allow_html=True)
st.title("◉ KnowledgeLens AI")
st.markdown(
    "Turn documents and code into a **source-backed knowledge graph**—then inspect connections, trace evidence, and ask questions grounded in the graph."
)
st.caption(f"v{APP_VERSION} · OpenAI-compatible endpoints · local or cloud")

if "kg_graph" not in st.session_state:
    st.session_state.kg_graph = nx.MultiDiGraph()
if "node_canonical_map" not in st.session_state:
    st.session_state.node_canonical_map = {}
if "master_concept" not in st.session_state:
    st.session_state.master_concept = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_revision" not in st.session_state:
    st.session_state.graph_revision = 0
if "loaded_state_fingerprint" not in st.session_state:
    st.session_state.loaded_state_fingerprint = None

with st.sidebar:
    st.header("Model connection")
    providers = ["Ollama / local", "llama.cpp / local", "OpenAI"]
    if os.getenv("KNOWLEDGELENS_CUSTOM_ENDPOINT", "").strip():
        providers.append("Configured endpoint")

    default_index = 0 if env_flag("KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS") else providers.index("OpenAI")
    provider = st.selectbox("Provider", providers, index=default_index)
    provider_key = provider_state_key(provider)
    base_url = configured_endpoint(provider)
    st.text_input(
        "Endpoint",
        value=base_url,
        disabled=True,
        help="Custom endpoints are configured by the operator via KNOWLEDGELENS_CUSTOM_ENDPOINT.",
    )
    api_key = st.text_input(
        "API key",
        type="password",
        key=f"api_key_{provider_key}",
        help="Required for OpenAI; optional for local/configured endpoints. Credentials are isolated per provider and never written to graph exports.",
    )
    default_model = "llama3.1" if provider != "OpenAI" else "gpt-4o-mini"
    model_name = st.text_input("Model", value=default_model, key=f"model_{provider_key}")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    try:
        validate_endpoint(base_url)
        st.caption("✓ Endpoint passes the current network policy")
    except EndpointPolicyError as exc:
        st.caption(f"Endpoint policy: {exc}")

    st.divider()
    st.header("Graph extraction")
    auto_detect = st.checkbox("Auto-detect master concept", value=True)
    manual_master = st.text_input("Master concept", disabled=auto_detect)
    master_link_count = st.slider(
        "Overview links",
        3,
        20,
        10,
        help="Synthetic links from the master concept to highly connected entities.",
    )
    custom_focus = st.text_area(
        "Optional extraction focus",
        placeholder="Example: prioritize APIs, dependencies, failure modes, and ownership.",
    )

    st.divider()
    st.header("Persistence")
    uploaded_state = st.file_uploader("Load graph state", type=["json"], key="state_loader")
    current_state_fingerprint = _state_upload_fingerprint(uploaded_state)
    if uploaded_state is None:
        st.session_state.loaded_state_fingerprint = None
    elif current_state_fingerprint != st.session_state.loaded_state_fingerprint:
        try:
            loaded, master, node_map = deserialize_graph_state(uploaded_state.getvalue())
            loaded_stats = graph_to_export(loaded)["stats"]
            st.session_state.kg_graph = loaded
            st.session_state.master_concept = master
            st.session_state.node_canonical_map = node_map
            st.session_state.processing_complete = True
            st.session_state.chat_history = []
            st.session_state.graph_revision += 1
            st.success(
                f"Loaded {loaded.number_of_nodes()} nodes / {loaded_stats['claims']} source-backed claims"
                f" / {loaded_stats['legacy_claims']} legacy ungrounded claims"
                f" / {loaded_stats['topology_edges']} topology edges"
            )
        except Exception as exc:
            st.error(f"Could not load graph state: {exc}")
        finally:
            st.session_state.loaded_state_fingerprint = current_state_fingerprint

    if st.button("Clear workspace", use_container_width=True):
        st.session_state.kg_graph = nx.MultiDiGraph()
        st.session_state.node_canonical_map = {}
        st.session_state.master_concept = None
        st.session_state.processing_complete = False
        st.session_state.chat_history = []
        st.session_state.loaded_state_fingerprint = current_state_fingerprint
        st.session_state.graph_revision += 1
        st.rerun()

supported_extensions = [
    "pdf", "txt", "md", "markdown", "json", "xml", "html", "htm", "css", "js", "ts", "tsx", "jsx",
    "py", "java", "c", "cpp", "h", "cs", "go", "rs", "php", "rb", "kt", "swift", "sh", "yaml", "yml",
    "sql", "r", "m", "tex", "latex", "toml", "ini", "bat", "ps1",
]

uploaded_files = st.file_uploader(
    "Add documents or source files",
    type=supported_extensions,
    accept_multiple_files=True,
    help="Text-based PDFs are supported. Scanned-image OCR is not included yet.",
)

if st.button("Build evidence graph", type="primary", use_container_width=True):
    if not uploaded_files:
        st.error("Add at least one document first.")
        st.stop()
    request_error = request_configuration_error(provider, api_key, model_name)
    if request_error:
        st.error(request_error)
        st.stop()
    try:
        validate_endpoint(base_url)
    except EndpointPolicyError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Reading and chunking sources…"):
        chunks, ingest_warnings = prepare_chunks(uploaded_files)
    for warning in ingest_warnings:
        st.warning(warning)
    if not chunks:
        st.error("No extractable text was found. Scanned PDFs currently require OCR before upload.")
        st.stop()

    try:
        if auto_detect:
            with st.spinner("Finding the central concept…"):
                master = detect_master_concept(base_url, api_key, model_name, chunks, temperature)
        else:
            master = manual_master.strip() or "Knowledge Base"
    except Exception as exc:
        st.warning(f"Master concept detection failed: {exc}")
        master = manual_master.strip() or "Knowledge Base"

    graph, node_map = create_graph(master)
    progress = st.progress(0)
    status = st.empty()
    failures = 0

    for index, chunk in enumerate(chunks, start=1):
        status.caption(f"Extracting claims · {index}/{len(chunks)} · {chunk.citation}")
        try:
            claims = extract_chunk_claims(base_url, api_key, model_name, chunk, temperature, custom_focus)
            add_claims(graph, node_map, claims)
        except Exception as exc:
            failures += 1
            st.warning(f"Skipped {chunk.citation}: {str(exc)[:180]}")
        progress.progress(index / len(chunks))

    if graph.number_of_nodes() > 1:
        candidates = top_entities(graph, master_link_count)
        try:
            links = generate_master_relations(base_url, api_key, model_name, master, candidates, temperature)
        except Exception:
            links = [(node, "relates to") for node in candidates]
        add_master_links(graph, node_map, master, links)

    build_stats = graph_to_export(graph)["stats"]
    status.empty()
    progress.empty()

    build_error = no_claims_build_error(build_stats["claims"], failures)
    if build_error:
        st.error(build_error)
        st.stop()

    # Commit a new graph to session state only after at least one auditable claim
    # survives extraction. A failed rebuild therefore cannot replace a good workspace.
    st.session_state.kg_graph = graph
    st.session_state.node_canonical_map = node_map
    st.session_state.master_concept = master
    st.session_state.processing_complete = True
    st.session_state.chat_history = []
    st.session_state.loaded_state_fingerprint = current_state_fingerprint
    st.session_state.graph_revision += 1
    st.success(
        f"Built {graph.number_of_nodes()} nodes and {build_stats['claims']} source-traceable claims "
        f"plus {build_stats['topology_edges']} topology edges. {failures} chunks failed."
    )
    st.rerun()

if st.session_state.processing_complete or st.session_state.kg_graph.number_of_nodes() > 0:
    graph = st.session_state.kg_graph
    export_data = graph_to_export(graph)
    source_count = export_data["stats"]["sources"]
    claim_count = export_data["stats"]["claims"]
    legacy_claim_count = export_data["stats"]["legacy_claims"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Master concept", st.session_state.master_concept or "—")
    c2.metric("Entities", max(0, graph.number_of_nodes() - 1))
    c3.metric("Claims", claim_count)
    c4.metric("Sources", source_count)

    st.subheader("Explore the evidence graph")
    st.caption(
        f"Hover an edge to see its source and evidence. The graph also contains "
        f"{export_data['stats']['topology_edges']} synthetic topology edges and {legacy_claim_count} legacy relations; "
        "neither is counted as an auditable claim or used for grounded chat."
    )
    visualize_graph(graph)

    st.subheader("Export & continue later")
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "Graph state",
        serialize_graph_state(graph, st.session_state.master_concept, st.session_state.node_canonical_map),
        file_name="knowledgelens_state.json",
        mime="application/json",
        use_container_width=True,
    )
    e2.download_button(
        "Evidence graph JSON",
        json.dumps(export_data, ensure_ascii=False, indent=2),
        file_name="knowledgelens_graph.json",
        mime="application/json",
        use_container_width=True,
    )

    readable_lines = []
    for claim in export_data["claims"]:
        source = claim["source"]
        if claim.get("synthetic"):
            source = "SYNTHETIC TOPOLOGY · non-evidentiary"
        elif claim.get("provenance_status") == "legacy-aggregated":
            candidates = ", ".join(claim.get("legacy_sources", [])) or "unknown"
            source = f"LEGACY AGGREGATED · non-grounded · candidates: {candidates}"
        elif not source and claim.get("legacy_sources"):
            source = "legacy candidates: " + ", ".join(claim["legacy_sources"])
        source = source or "unknown source"
        location = f"p.{claim['page']}" if claim["page"] is not None else f"chunk {claim['chunk_index']}"
        readable_lines.append(
            f"[{source} · {location}] {claim['subject']} --[{claim['relation']}]--> {claim['object']}"
        )
    e3.download_button(
        "Claim ledger",
        "\n".join(readable_lines),
        file_name="knowledgelens_claims.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Ask the graph")
    st.caption("Answers are constrained to retrieved source-backed claims and their citations.")

    user_query = st.chat_input("How are these concepts connected? What evidence supports this claim?")
    if user_query:
        request_error = request_configuration_error(provider, api_key, model_name)
        if request_error:
            st.error(request_error)
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            context = retrieve_graph_context(graph, user_query)
            try:
                with st.spinner("Tracing graph evidence…"):
                    answer = call_llm_api(
                        base_url,
                        api_key,
                        model_name,
                        (
                            "Answer ONLY from the supplied KnowledgeLens graph context. "
                            "Every factual sentence must cite the bracketed source/location supporting it. "
                            "Lines beginning 'Graph path:' are routing metadata, not citations. "
                            "Synthetic overview links and legacy aggregated relations are excluded from grounded context. "
                            "If the graph lacks enough evidence, say that clearly. Do not use outside knowledge."
                        ),
                        f"Graph context:\n{context}\n\nQuestion: {user_query}",
                        temperature,
                    )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                st.error(f"Could not answer: {exc}")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
else:
    st.info("Add files above to build your first evidence graph.")
