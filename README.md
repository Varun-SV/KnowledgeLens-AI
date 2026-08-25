# ◉ KnowledgeLens AI

**Make knowledge inspectable.** KnowledgeLens AI turns documents and source code into a source-backed knowledge graph, then lets you explore relationships, trace evidence, ask graph-grounded questions, and export the result.

[Open the Streamlit app](https://knowledgelens.streamlit.app/) · [Project website](https://varun-sv.github.io/KnowledgeLens-AI/) · [MIT License](LICENSE)

## Why KnowledgeLens?

Traditional document chat hides most of the retrieval process. KnowledgeLens keeps an explicit graph in the middle:

```text
Documents → source-aware chunks → structured claims → evidence graph → graph retrieval → LLM answer
```

Each extracted claim can preserve its own **source, page/chunk, evidence, and confidence**, so two relationships between the same entities do not have to collapse into one anonymous edge.

## Current capabilities

- **Document + code ingestion:** text-based PDF, TXT, Markdown, JSON, XML, HTML, CSS, JavaScript/TypeScript, Python, Java, C/C++, C#, Go, Rust, PHP, Ruby, Kotlin, Swift, shell, YAML, SQL, R, TeX and other text-like formats.
- **Provider-neutral LLM connection:** built-in Ollama, llama.cpp, and OpenAI presets plus an operator-configured OpenAI-compatible endpoint.
- **Auditable extraction:** structured claim parsing with a compatibility fallback for legacy `SUBJECT | RELATION | OBJECT | VERBATIM_EVIDENCE [| CONFIDENCE]` output. The evidence field is required and must occur in the supplied source chunk.
- **Per-claim provenance:** the graph uses a `MultiDiGraph`, so each source-backed relationship remains independently inspectable.
- **Interactive graph:** drag, pan, zoom and hover relationships to inspect provenance.
- **Graph-aware retrieval:** boundary-aware entity matching, local claim neighborhoods, and short mixed-direction graph paths are surfaced before chat generation.
- **Grounded chat:** the answer prompt is restricted to retrieved graph context and asks for source/location citations.
- **Portable state:** save/reload graph state, export evidence graph JSON, or export a human-readable claim ledger.
- **Legacy state migration:** v1 graph exports are upgraded without inventing relation↔source pairings that the old schema never stored.

> **Not supported yet:** OCR for scanned/image-only PDFs, DOCX/PPTX native parsing, embeddings/vector retrieval, collaborative multi-user persistence, and production authentication. The website intentionally does not claim these features.

## Run locally

KnowledgeLens uses a secure-by-default endpoint policy. Public visitors cannot type an arbitrary server-side request target into the UI. The endpoint selector resolves only to built-in provider URLs or an endpoint configured by the person operating the KnowledgeLens server.

```bash
git clone https://github.com/Varun-SV/KnowledgeLens-AI.git
cd KnowledgeLens-AI
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For a **trusted local** Ollama or llama.cpp deployment:

```bash
# macOS / Linux
export KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS=1

# PowerShell
$env:KNOWLEDGELENS_ALLOW_LOCAL_ENDPOINTS="1"
```

Then start the app:

```bash
streamlit run KnowledgeLens_AI.py
```

Built-in endpoints:

| Provider | Base URL |
| --- | --- |
| Ollama | `http://localhost:11434` |
| llama.cpp server | `http://localhost:8080` |
| OpenAI | `https://api.openai.com` |

KnowledgeLens appends `/v1/chat/completions` itself.

### Custom OpenAI-compatible endpoint

Custom endpoints are configured by the server operator, not supplied by arbitrary visitors:

```bash
export KNOWLEDGELENS_CUSTOM_ENDPOINT="https://llm.example.com"
```

For a private-network endpoint, also explicitly opt in:

```bash
export KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1
```

Plain HTTP is accepted only when **every resolved address is an explicitly allowed local/private destination**. A local/private opt-in never makes public `http://` endpoints acceptable. Redirects are rejected so credentials cannot silently follow a redirect to another host.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
ruff check .
python -m compileall KnowledgeLens_AI.py knowledgelens
```

The core code is split into focused modules:

```text
knowledgelens/
├── graph.py         # MultiDiGraph + provenance-preserving claims
├── ingestion.py     # bounded file extraction + line-preserving chunking
├── models.py        # DocumentChunk / Claim
├── parsing.py       # structured output + compatibility parser
├── persistence.py   # bounded state schema + legacy migration
├── presentation.py  # safe visualization text helpers
├── retrieval.py     # entity scoring, neighborhoods, mixed-direction paths
└── security.py      # endpoint network policy
```

The Streamlit UI/orchestration remains in `KnowledgeLens_AI.py`.

## Website

The static landing page lives in `site/` and is intentionally framework-free: HTML, CSS, SVG, and vanilla JavaScript. GitHub Actions publishes it to GitHub Pages after changes land on `main`.

Its signature interaction is the **provenance lens**: moving the lens over the hero graph reveals relationship labels and the source/evidence behind the nearest claim.

## Automation

- **CI:** supported Python matrix, Ruff, pytest and compile checks.
- **CodeQL:** Python security analysis on pull requests, `main`, and a weekly schedule.
- **Pages:** static `site/` deployment to GitHub Pages.
- **Dependabot:** weekly Python and GitHub Actions dependency updates.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md). Security issues should follow [SECURITY.md](SECURITY.md) rather than being filed publicly.

## License

MIT © Varun S V. See [LICENSE](LICENSE).
