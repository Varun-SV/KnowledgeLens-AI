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
- **Persistent provider profiles:** with PostgreSQL configured, Ollama, llama.cpp, OpenAI, and arbitrary OpenAI-compatible endpoints can be saved/edited from the UI and used immediately without restarting KnowledgeLens.
- **Secure credentials:** provider secrets prefer the OS keychain/credential vault; server deployments can use encrypted PostgreSQL fallback with `KNOWLEDGELENS_MASTER_KEY`. Plaintext credentials are not stored in provider-profile rows.
- **Provider discovery:** bounded, DNS-pinned model discovery is available for OpenAI-compatible providers and Ollama; capability flags stay explicit instead of being guessed from model names.
- **Persistence foundation:** PostgreSQL initializes workspace/document/job/checkpoint/provider/auth tables automatically and the default content store uses streaming SHA-256-addressed local blobs.
- **Compatibility fallback:** if PostgreSQL is not configured or is temporarily unavailable, KnowledgeLens still starts with the prior fixed provider presets and portable NetworkX/JSON state.
- **Auditable extraction:** structured claim parsing with a compatibility fallback for legacy `SUBJECT | RELATION | OBJECT | VERBATIM_EVIDENCE [| CONFIDENCE]` output. The evidence field is required and must occur in the supplied source chunk.
- **Per-claim provenance:** the graph uses a `MultiDiGraph`, so each source-backed relationship remains independently inspectable.
- **Interactive graph:** drag, pan, zoom and hover relationships to inspect provenance.
- **Graph-aware retrieval:** boundary-aware entity matching, local claim neighborhoods, and short mixed-direction graph paths are surfaced before chat generation.
- **Grounded chat:** the answer prompt is restricted to retrieved graph context and asks for source/location citations.
- **Portable state:** save/reload graph state, export evidence graph JSON, or export a human-readable claim ledger.
- **Legacy state migration:** v1 graph exports are upgraded without inventing relation↔source pairings that the old schema never stored.

> **Still planned for the following v0.3 PRs:** Docling OCR/image/table/figure extraction, arbitrarily large resumable document jobs, multimodal model routing, scalable PostgreSQL-backed graph materialization, production OIDC enforcement, and advanced S3/PostgreSQL-blob storage.

## Run locally

KnowledgeLens uses a secure-by-default endpoint policy. Public visitors cannot type arbitrary server-side request targets unless an authenticated/admin policy permits it. Endpoint validation and model discovery both retain the exact validated IP address set to avoid DNS-rebinding gaps.

```bash
git clone https://github.com/Varun-SV/KnowledgeLens-AI.git
cd KnowledgeLens-AI
python -m venv .venv
```

Activate the virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

### PostgreSQL persistence

Create an empty PostgreSQL database/user, then configure only the connection URL; KnowledgeLens initializes its schema automatically on startup:

```bash
# macOS / Linux
export KNOWLEDGELENS_DATABASE_URL="postgresql://knowledgelens:change-me@localhost:5432/knowledgelens"

# PowerShell
$env:KNOWLEDGELENS_DATABASE_URL="postgresql://knowledgelens:change-me@localhost:5432/knowledgelens"
```

For server deployments that cannot use an OS keychain, also configure a strong master key for encrypted secret fallback:

```bash
export KNOWLEDGELENS_MASTER_KEY="replace-this-with-a-long-random-secret"
```

The default blob store is `data/blobs/sha256/...`; override it with `KNOWLEDGELENS_BLOB_ROOT`.

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

With PostgreSQL enabled, open **AI Providers** from the sidebar to add or update endpoints/models/credentials. Changes are picked up on rerun and **do not require a KnowledgeLens server restart**.

Built-in profile defaults:

| Provider | Base URL |
| --- | --- |
| Ollama | `http://localhost:11434` |
| llama.cpp server | `http://localhost:8080` |
| OpenAI | `https://api.openai.com` |

KnowledgeLens appends `/v1/chat/completions` itself for chat/extraction requests.

### Custom OpenAI-compatible endpoints

Custom endpoint profiles are persisted in PostgreSQL. On `local`/`private` deployments the trusted operator can manage them immediately. On `public` deployments profile management remains read-only until the authenticated admin/OIDC layer is completed; this preserves the SSRF trust boundary established in v0.2.

For private-network endpoints, explicitly opt in:

```bash
export KNOWLEDGELENS_ALLOW_PRIVATE_ENDPOINTS=1
```

Plain HTTP is accepted only when **every resolved address is an explicitly allowed local/private destination**. A local/private opt-in never makes public `http://` endpoints acceptable. Redirects are rejected so credentials cannot silently follow a redirect to another host.

The old `KNOWLEDGELENS_CUSTOM_ENDPOINT` environment variable remains as a compatibility fallback when PostgreSQL profiles are unavailable.

## Authentication foundation

PR #2 includes storage/configuration primitives for bootstrap local admins and OIDC settings:

- `KNOWLEDGELENS_AUTH_MODE=disabled|local|oidc`
- `KNOWLEDGELENS_OIDC_ISSUER`
- `KNOWLEDGELENS_OIDC_CLIENT_ID`
- `KNOWLEDGELENS_OIDC_SCOPES`

Password hashing uses scrypt. After configuring PostgreSQL, create the first local administrator from an interactive prompt:

```bash
knowledgelens bootstrap-admin --username admin
```

The password is requested twice through `getpass`; it is never accepted as a command-line argument, so it is not written to normal shell history. The bootstrap command is one-time: once an administrator exists, it refuses to create another first admin.

Full browser-session enforcement, role-aware public administration, and OIDC login complete in the production/Advanced PR rather than being partially exposed here.

## Development

```bash
pip install -e ".[dev]"
python -m pytest
ruff check .
python -m compileall KnowledgeLens_AI.py knowledgelens pages
```

The core code is split into focused modules:

```text
knowledgelens/
├── auth.py              # bootstrap admin + OIDC configuration foundation
├── cli.py               # interactive administration commands
├── blob_store.py        # streaming SHA-256 local content-addressed storage
├── database.py          # PostgreSQL connection + automatic schema initialization
├── graph.py             # MultiDiGraph + provenance-preserving claims
├── http_client.py       # DNS-pinned bounded GET/POST transport
├── ingestion.py         # bounded file extraction + line-preserving chunking
├── models.py            # DocumentChunk / Claim
├── parsing.py           # structured output + compatibility parser
├── persistence.py       # bounded graph-state schema + legacy migration
├── presentation.py      # safe visualization text helpers
├── provider_profiles.py # persistent endpoints/models/capabilities + discovery
├── retrieval.py         # entity scoring, neighborhoods, mixed-direction paths
├── secrets.py           # OS keyring + encrypted PostgreSQL secret backends
├── services.py          # compatibility-safe application service bootstrap
└── security.py          # endpoint network policy
```

The Streamlit workspace/orchestration remains in `KnowledgeLens_AI.py`; provider management is in `pages/1_AI_Providers.py`.

## Website

The static landing page lives in `site/` and is intentionally framework-free: HTML, CSS, SVG, and vanilla JavaScript. GitHub Actions publishes it to GitHub Pages after changes land on `main`.

Its signature interaction is the **provenance lens**: moving the lens over the hero graph reveals relationship labels and the source/evidence behind the nearest claim.

## Automation

- **CI:** supported Python matrix, Ruff, pytest, compile and JavaScript syntax checks.
- **CodeQL:** Python security analysis on pull requests, `main`, and a weekly schedule.
- **Pages:** static `site/` deployment to GitHub Pages.
- **Dependabot:** weekly Python and GitHub Actions dependency updates.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md). Security issues should follow [SECURITY.md](SECURITY.md) rather than being filed publicly.

## License

MIT © Varun S V. See [LICENSE](LICENSE).
