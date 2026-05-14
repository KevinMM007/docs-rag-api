# Docs RAG API

[![CI](https://github.com/KevinMM007/docs-rag-api/actions/workflows/ci.yml/badge.svg)](https://github.com/KevinMM007/docs-rag-api/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL_+_pgvector-16-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](#-tests)
[![Tests](https://img.shields.io/badge/tests-72_passing-brightgreen.svg)](#-tests)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Upload a PDF or Markdown file, then ask questions about it in natural language.** The server chunks the document, embeds each chunk with Google Gemini, stores the vectors in pgvector, and answers questions by retrieving the most relevant chunks and streaming Gemini's answer back token by token over Server-Sent Events — with inline citations to the source document.

---

## ⚡ Try it in 60 seconds

> Live demo: **<https://docs-rag-api.onrender.com/docs>** *(cold-start ~30 s on free tier — the first request wakes the dyno)*

```bash
# 1. Register a user
curl -X POST https://docs-rag-api.onrender.com/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"strong-pw-12"}'

# 2. Log in (returns a JWT)
TOKEN=$(curl -s -X POST https://docs-rag-api.onrender.com/api/v1/auth/login \
  -d "username=you@example.com&password=strong-pw-12" | jq -r .access_token)

# 3. Upload a document
curl -X POST https://docs-rag-api.onrender.com/api/v1/documents/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@your-document.pdf"

# 4. Chat with it (streaming SSE — watch tokens arrive in real time)
curl -N -X POST https://docs-rag-api.onrender.com/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the main conclusion?"}'
```

The chat endpoint streams three event types: `sources` (which chunks were retrieved), `token` (one per Gemini fragment), and either `done` or `error` to terminate.

---

## 🧠 What makes it interesting

| | |
|---|---|
| 🔐 **JWT auth with per-user isolation** | Every document and chunk is scoped to the uploader; cross-user reads are filtered at the SQL `WHERE` clause, not in application code, so a forgotten predicate would have to be a deliberate model change rather than a missed `if`. |
| 🧮 **pgvector + HNSW index** | 768-dimensional embeddings from Gemini's `gemini-embedding-001` (Matryoshka-reduced from 3072), indexed with `vector_cosine_ops` for cosine-distance lookups in single-digit milliseconds even with thousands of chunks. |
| ✂️ **Sentence-aware chunking** | Sliding window with configurable overlap. Prefers to break on `.` `!` `?` `\n\n` boundaries so chunks don't end mid-sentence — embeddings of clean fragments rank noticeably better than embeddings of arbitrary char windows. |
| 💬 **Streaming SSE with grounded answers** | The system prompt forces Gemini to cite filenames in `[brackets]` and refuse to answer when the retrieved context is insufficient. No "Based on the documents..." preamble. |
| 🧪 **Real Postgres in tests, mocked LLM** | Integration tests spin up `pgvector/pgvector:pg16` via testcontainers so the `vector(768)` column path is actually exercised; the Gemini SDK is stubbed at the `_client` boundary so CI never burns the free-tier quota. **96 %** branch coverage. |
| 🐳 **Multi-stage Docker** | 519 MB runtime image (gcc + build deps stripped), non-root user, healthcheck against `/api/v1/ping`. Same image runs locally and on Render. |

---

## 🧱 Stack

| Layer | Technology |
|---|---|
| Web framework | **FastAPI 0.115** + Pydantic v2 |
| ORM | **SQLAlchemy 2** (modern `Mapped[]` syntax) |
| Database | **PostgreSQL 16 + pgvector** (HNSW index, cosine ops) |
| Migrations | **Alembic** |
| LLM (chat + embeddings) | **Google Gemini** (`gemini-2.5-flash`, `gemini-embedding-001`) |
| Document parsing | **PyMuPDF** (PDFs), UTF-8 reader (Markdown) |
| Auth | **JWT** + bcrypt |
| Tests | **Pytest** + **testcontainers-postgres** + Gemini SDK stub |
| Lint | **Ruff** |
| Container | **Docker** (multi-stage, non-root, ~520 MB) |
| CI | **GitHub Actions** (lint + tests + image build) |
| Deploy | **Render** (web service) + **Neon** (Postgres with pgvector, free tier) |

> Cost-conscious by design: Google AI Studio free tier (1500 req/day, no card) + Render free web service + Neon free Postgres = **$0/month total**.

---

## 🚀 Run locally

**Requirements:** Python 3.13, Docker Desktop, a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone + install deps
git clone https://github.com/KevinMM007/docs-rag-api.git
cd docs-rag-api
python -m venv .venv
.venv\Scripts\activate                    # Windows
# source .venv/bin/activate                 # Linux / Mac
pip install -r requirements-dev.txt

# 2. Spin up Postgres + pgvector
docker compose up -d

# 3. Configure environment
copy .env.example .env                     # Windows
# cp .env.example .env                       # Linux / Mac
# Paste your Gemini key into GEMINI_API_KEY=

# 4. Apply migrations
alembic upgrade head

# 5. Run the API
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for the live Swagger UI.

### Alternative: run the Docker image

```bash
docker build -t docs-rag-api:local .
docker run -p 8000:8000 --env-file .env docs-rag-api:local
```

---

## ☁️ Deploy to production

The repo ships with a `render.yaml` Blueprint and a `Dockerfile` ready for any container host. The reference deployment uses Render (web service, free tier) + Neon (Postgres + pgvector, free tier).

### One-time setup

1. **Neon Postgres** ([neon.tech](https://neon.tech)) → create a project, copy the `postgresql://...` connection string. Run `CREATE EXTENSION vector;` once in the Neon SQL editor.
2. **Render** ([render.com](https://render.com)) → New → Blueprint → point at this repo. Render reads `render.yaml` and provisions a Docker web service with the env vars wired up.
3. In the Render dashboard, set the two `sync: false` secrets:
   - `DATABASE_URL` = Neon connection string
   - `GEMINI_API_KEY` = your Google AI Studio key
4. Push to `main`. Render builds the Dockerfile, runs `alembic upgrade head` via `preDeployCommand`, and brings up the service.

Subsequent deploys are automatic on every push to `main`.

---

## 🧪 Tests

```bash
pytest
```

The suite is **72 tests / 96 % branch coverage**, gated at ≥ 90 % via `--cov-fail-under` in `pyproject.toml`.

| Layer | What is exercised |
|---|---|
| Auth | register / login / `/me`, JWT expiry, non-integer subject, deleted user |
| Documents | PDF + Markdown upload, MIME inference from extension, 415 / 413 / 422 / 503 paths, per-user isolation, FK cascade on delete |
| Embeddings | retry exhaustion + recovery, dimension validation, L2 normalisation, API-key guard |
| Retrieval | similarity ranking (identical text → distance 0), per-user filtering, `top_k` cap |
| Chat (SSE) | sources event first, token stream, `done` vs mid-stream `error`, 503 before stream opens |
| Config | `postgres://` → `postgresql+psycopg://` coercion, CORS CSV parsing |

Integration tests run against a real `pgvector/pgvector:pg16` container (testcontainers spins one up per session); the Gemini SDK is monkey-patched at the `_client` boundary so the full wrapper code path runs without touching the network.

---

## 🗂 Repository layout

```
docs-rag-api/
├── app/
│   ├── api/
│   │   ├── deps.py                  # FastAPI deps: get_db, get_current_user
│   │   └── v1/                      # versioned routers
│   │       ├── auth.py              # register / login / me
│   │       ├── documents.py         # upload / list / get / delete / search
│   │       └── chat.py              # POST /chat → SSE stream
│   ├── core/
│   │   ├── config.py                # Pydantic Settings, URL coercion
│   │   ├── database.py              # SQLAlchemy engine + session factory
│   │   └── security.py              # bcrypt + JWT encode/decode
│   ├── crud/                        # thin query helpers (no business logic)
│   ├── models/                      # User, Document, Chunk (vector(768) column)
│   ├── schemas/                     # Pydantic DTOs
│   ├── services/
│   │   ├── parsers.py               # PyMuPDF + Markdown
│   │   ├── chunking.py              # sentence-aware sliding window
│   │   ├── embeddings.py            # Gemini wrapper + retry + L2 normalize
│   │   ├── retrieval.py             # cosine similarity search
│   │   ├── llm.py                   # Gemini chat streaming wrapper
│   │   └── rag.py                   # prompt template + orchestration
│   └── main.py
├── alembic/versions/                # 0001 pgvector+users, 0002 docs+chunks, 0003 embeddings+HNSW
├── tests/                           # 72 tests, 96 % coverage
├── .github/workflows/ci.yml         # lint + tests + Docker build
├── Dockerfile                       # multi-stage, non-root, ~520 MB
├── docker-compose.yml               # local Postgres + pgvector
├── render.yaml                      # Render Blueprint
└── requirements*.txt
```

---

## 📜 License

[MIT](LICENSE)

---

Built by **[Kevin Morales](https://github.com/KevinMM007)** as the third project of a backend portfolio aimed at remote LATAM / USA junior roles.
