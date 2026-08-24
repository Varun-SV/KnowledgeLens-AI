from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import networkx as nx
import requests
import streamlit as st
import streamlit.components.v1 as components
from pypdf import PdfReader
from pyvis.network import Network

from knowledgelens.graph import (
    add_claims,
    add_master_links,
    canonical_key,
    create_graph,
    graph_to_export,
    top_entities,
)
from knowledgelens.models import DocumentChunk
from knowledgelens.parsing import parse_claims
from knowledgelens.retrieval import retrieve_graph_context
from knowledgelens.security import EndpointPolicyError, validate_endpoint

APP_VERSION = "0.2.0"


def extract_sections_from_file(uploaded_file) -> list[tuple[int | None, str]]:
    """Extract source sections while preserving PDF page provenance."""
    uploaded_file.seek(0)
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        sections: list[tuple[int | None, str]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            if text.strip():
                sections.append((page_number, text))
        return sections

    raw = uploaded_file.read()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return [(None, raw.decode(encoding))]
        except UnicodeDecodeError:
            continue
    return []


def _split_oversized(text: str, max_chars: int, overlap: int) -> Iterable[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece:
            yield piece
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def chunk_section(text: str, max_chars: int = 3200, overlap: int = 240) -> list[str]:
    """Paragraph-aware chunking with a hard size ceiling and small overlap."""
    cleaned = text.replace("\r", "\n")
    paragraphs = [" ".join(p.split()) for p in cleaned.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [" ".join(cleaned.split())]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_split_oversized(paragraph, max_chars, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            prefix = current[-overlap:].strip() if current and overlap else ""
            current = f"{prefix}\n\n{paragraph}".strip() if prefix else paragraph
            if len(current) > max_chars:
                chunks.extend(_split_oversized(current, max_chars, overlap))
                current = ""

    if current:
        chunks.append(current.strip())
    return chunks


def prepare_chunks(uploaded_files) -> tuple[list[DocumentChunk], list[str]]:
    chunks: list[DocumentChunk] = []
    warnings: list[str] = []
    global_index = 0

    for uploaded_file in uploaded_files:
        try:
            sections = extract_sections_from_file(uploaded_file)
        except Exception as exc:
            warnings.append(f"{uploaded_file.name}: {exc}")
            continue

        if not sections:
            warnings.append(f"{uploaded_file.name}: no extractable text found")
            continue

        for page, text in sections:
            for piece in chunk_section(text):
                global_index += 1
                chunks.append(
                    DocumentChunk(
                        source=uploaded_file.name,
                        text=piece,
                        chunk_index=global_index,
                        page=page,
                    )
                )
    return chunks, warnings


def call_llm_api(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.1,
) -> str:
    endpoint = validate_endpoint(base_url)
    url = endpoint + "/v1/chat/completions"
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
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=(10, 180),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach the LLM endpoint: {exc}") from exc

    if 300 <= response.status_code < 400:
        raise RuntimeError("The LLM endpoint returned a redirect. Redirects are blocked for credential safety.")
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:400]
        raise RuntimeError(f"LLM returned HTTP {response.status_code}: {detail}") from exc

    try:
        payload = response.json()
        return str(payload["choices"][0]["message"]["content"])
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
    concept = response.strip().strip('"\'`').splitlines()[0].strip()
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


def reconstruct_node_map(graph: nx.MultiDiGraph) -> dict[str, str]:
    return {canonical_key(str(node)): str(node) for node in graph.nodes}


def migrate_legacy_graph(graph: nx.Graph) -> nx.MultiDiGraph:
    if isinstance(graph, nx.MultiDiGraph):
        return graph

    migrated = nx.MultiDiGraph()
    migrated.add_nodes_from(graph.nodes(data=True))
    for subject, obj, data in graph.edges(data=True):
        relations = data.get("relations") or [data.get("relation") or "related to"]
        sources = data.get("sources") or {data.get("source") or "Imported graph"}
        if isinstance(sources, list):
            sources = set(sources)
        for relation in relations:
            for source in sources:
                migrated.add_edge(
                    subject,
                    obj,
                    relation=relation,
                    source=source,
                    page=None,
                    chunk_index=0,
                    evidence="Imported from KnowledgeLens graph state v1.",
                    confidence=None,
                    synthetic=False,
                )
    return migrated


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
            title=f"{node}\n{degree} connected claims",
            shape="star" if is_master else "dot",
            size=size,
            color="#FFB45C" if is_master else "#68E1FD",
            font={"size": 21 if is_master else 13, "face": "Inter, Arial", "color": "#F5F8FF"},
            borderWidth=2 if is_master else 1,
            borderWidthSelected=4,
        )

    for subject, obj, _key, data in graph.edges(keys=True, data=True):
        relation = str(data.get("relation", "related to"))
        source = str(data.get("source", "unknown"))
        page = data.get("page")
        chunk = data.get("chunk_index")
        location = f"p.{page}" if page is not None else f"chunk {chunk}"
        evidence = str(data.get("evidence") or "")
        title = f"{relation}\nSource: {source} · {location}"
        if evidence:
            title += f"\nEvidence: {evidence[:300]}"
        network.add_edge(
            subject,
            obj,
            label=relation[:18] + ("…" if len(relation) > 18 else ""),
            title=title,
            color="rgba(126, 145, 180, 0.42)",
            arrows="to",
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
                "edges": {"smooth": {"type": "continuous"}},
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


def export_state(graph: nx.MultiDiGraph) -> str:
    payload = {
        "schema_version": 2,
        "master_concept": st.session_state.master_concept,
        "node_canonical_map": st.session_state.node_canonical_map,
        "graph_data": nx.node_link_data(graph),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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

with st.sidebar:
    st.header("Model connection")
    provider = st.selectbox("Preset", ["Ollama / local", "llama.cpp / local", "OpenAI-compatible cloud", "Custom"])
    preset_urls = {
        "Ollama / local": "http://localhost:11434",
        "llama.cpp / local": "http://localhost:8080",
        "OpenAI-compatible cloud": "https://api.openai.com",
        "Custom": "https://",
    }
    base_url = st.text_input("Base URL", value=preset_urls[provider], help="KnowledgeLens appends /v1/chat/completions")
    api_key = st.text_input("API key", type="password", help="Kept in this Streamlit session; not written to graph exports.")
    model_name = st.text_input("Model", value="llama3.1")
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
    master_link_count = st.slider("Overview links", 3, 20, 10, help="Synthetic links from the master concept to highly connected entities.")
    custom_focus = st.text_area("Optional extraction focus", placeholder="Example: prioritize APIs, dependencies, failure modes, and ownership.")

    st.divider()
    st.header("Persistence")
    uploaded_state = st.file_uploader("Load graph state", type=["json"], key="state_loader")
    if uploaded_state is not None:
        try:
            state = json.load(uploaded_state)
            loaded = nx.node_link_graph(state["graph_data"])
            loaded = migrate_legacy_graph(loaded)
            st.session_state.kg_graph = loaded
            st.session_state.master_concept = state.get("master_concept")
            st.session_state.node_canonical_map = state.get("node_canonical_map") or reconstruct_node_map(loaded)
            st.session_state.processing_complete = True
            st.session_state.chat_history = []
            st.success(f"Loaded {loaded.number_of_nodes()} nodes / {loaded.number_of_edges()} claims")
        except Exception as exc:
            st.error(f"Could not load graph state: {exc}")

    if st.button("Clear workspace", use_container_width=True):
        st.session_state.kg_graph = nx.MultiDiGraph()
        st.session_state.node_canonical_map = {}
        st.session_state.master_concept = None
        st.session_state.processing_complete = False
        st.session_state.chat_history = []
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
    if not model_name.strip():
        st.error("Enter a model name.")
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
    extracted = 0
    failures = 0

    for index, chunk in enumerate(chunks, start=1):
        status.caption(f"Extracting claims · {index}/{len(chunks)} · {chunk.citation}")
        try:
            claims = extract_chunk_claims(base_url, api_key, model_name, chunk, temperature, custom_focus)
            extracted += add_claims(graph, node_map, claims)
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

    st.session_state.kg_graph = graph
    st.session_state.node_canonical_map = node_map
    st.session_state.master_concept = master
    st.session_state.processing_complete = True
    st.session_state.chat_history = []
    st.session_state.graph_revision += 1
    status.empty()
    progress.empty()

    if graph.number_of_nodes() <= 1:
        st.error("The model did not produce usable claims. Try a stronger instruction-following model or inspect the source text.")
    else:
        st.success(f"Built {graph.number_of_nodes()} nodes and {graph.number_of_edges()} source-traceable claims. {failures} chunks failed.")
    st.rerun()

if st.session_state.processing_complete or st.session_state.kg_graph.number_of_nodes() > 0:
    graph = st.session_state.kg_graph
    export_data = graph_to_export(graph)
    source_count = export_data["stats"]["sources"]

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Master concept", st.session_state.master_concept or "—")
    c2.metric("Entities", max(0, graph.number_of_nodes() - 1))
    c3.metric("Claims", graph.number_of_edges())
    c4.metric("Sources", source_count)

    st.subheader("Explore the evidence graph")
    st.caption("Hover an edge to see its source and evidence. Drag nodes, zoom, and trace how claims connect.")
    visualize_graph(graph)

    st.subheader("Export & continue later")
    e1, e2, e3 = st.columns(3)
    e1.download_button(
        "Graph state",
        export_state(graph),
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
    readable = "\n".join(
        f"[{claim['source']} · {'p.' + str(claim['page']) if claim['page'] else 'chunk ' + str(claim['chunk_index'])}] "
        f"{claim['subject']} --[{claim['relation']}]--> {claim['object']}"
        for claim in export_data["claims"]
    )
    e3.download_button(
        "Claim ledger",
        readable,
        file_name="knowledgelens_claims.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Ask the graph")
    st.caption("Answers are constrained to retrieved graph claims and their source citations.")

    user_query = st.chat_input("How are these concepts connected? What evidence supports this claim?")
    if user_query:
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
