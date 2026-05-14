# Docs RAG API

[![Python](https://img.shields.io/badge/python-3.13-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL_+_pgvector-16-4169E1.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen.svg)](#-tests)
[![Tests](https://img.shields.io/badge/tests-72_passing-brightgreen.svg)](#-tests)

> Upload your PDFs and Markdown, then ask questions in natural language. The server chunks the documents, embeds each chunk with Google Gemini, stores the vectors in pgvector, and answers questions by retrieving the most relevant chunks and feeding them to Gemini 2.5 Flash as RAG context — with streaming responses and citations.

> Status: 🚧 in active development (Sesión 5 / 6).

---

## ✨ Planned features

- 🔐 **JWT auth** with per-user document isolation - your uploads are yours
- 📄 **Upload PDFs and Markdown** with PyMuPDF for layout-aware parsing
- ✂️ **Token-aware chunking** with configurable overlap, tuned for retrieval quality
- 🧮 **Vector embeddings via Gemini `text-embedding-004`** stored in pgvector
- 🔍 **Similarity search** using cosine distance over pgvector indexes
- 💬 **Streaming Q&A** with Gemini 2.5 Flash + retrieved-context grounding
- 📚 **Citations** - answers reference the source chunks so users can verify
- 🧪 **Tests with mocked LLM responses** - CI never burns the rate limit
- 🐳 **Production-ready Docker** + GitHub Actions CI/CD + Render deploy

---

## 🧱 Stack

| Layer | Technology |
|---|---|
| Web framework | **FastAPI** + Pydantic v2 |
| ORM | **SQLAlchemy 2** (modern `Mapped[]` syntax) |
| Database | **PostgreSQL 16 + pgvector** |
| Migrations | **Alembic** |
| LLM (chat + embeddings) | **Google Gemini** (`gemini-2.5-flash`, `text-embedding-004`) |
| Document parsing | **PyMuPDF** (PDFs), plain-text reader (Markdown) |
| Auth | **JWT** + bcrypt |
| Tests | **Pytest** + mocked Gemini client |
| Lint | **Ruff** |
| Deploy | **Render** (web service + free Postgres with pgvector) |

> Cost-conscious by design: free tier of Google AI Studio (1500 req/day, no card) + Render free tier (web + Postgres) = ~$0/month total.

---

## 🚀 Quickstart (local)

**Requirements:** Python 3.13, Docker Desktop, a free [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone + install deps
git clone https://github.com/KevinMM007/docs-rag-api.git
cd docs-rag-api
python -m venv .venv
.venv\Scripts\activate                  # Windows
# source .venv/bin/activate              # Linux / Mac
pip install -r requirements-dev.txt

# 2. Spin up Postgres with pgvector (host port 5434 to avoid conflicts)
docker compose up -d

# 3. Configure environment
copy .env.example .env                   # Windows
# cp .env.example .env                    # Linux / Mac
# then paste your Gemini key into GEMINI_API_KEY=

# 4. (Migrations + run the API land in later sessions)
uvicorn app.main:app --reload
```

Open <http://localhost:8000/docs> for the live Swagger UI.

---

## 🧪 Tests

```bash
pytest
```

The test suite runs 72 integration + unit tests against a real Postgres + pgvector container (spun up via testcontainers per session) and a deterministic in-memory stub of the Gemini SDK, so CI never burns the free-tier quota. Coverage is enforced at **≥ 90 %** via `--cov-fail-under` — the suite currently sits at **96 %** with branch coverage enabled.

| Layer | What is exercised |
|---|---|
| Auth | register / login / `/me`, JWT expiry, non-integer subject, deleted user |
| Documents | PDF + Markdown upload, MIME inference from extension, 415 / 413 / 422 / 503 error paths, per-user isolation, FK cascade on delete |
| Embeddings | retry exhaustion + recovery, dimension validation, L2 normalisation, API-key guard |
| Retrieval | similarity ranking (identical text → distance 0), per-user filtering, `top_k` cap |
| Chat (SSE) | sources event first, token stream, `done` vs mid-stream `error`, 503 before stream opens |
| Config | `postgres://` → `postgresql+psycopg://` coercion, CORS CSV parsing |

---

## 🗂 Repository layout

```
docs-rag-api/
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/                  # versioned routers (auth, documents, chat)
│   ├── core/
│   │   ├── config.py            # Pydantic Settings + Gemini config
│   │   ├── database.py
│   │   └── security.py          # JWT helpers (Sesión 2)
│   ├── crud/                    # thin SQLAlchemy query helpers
│   ├── models/                  # User, Document, Chunk (with Vector column)
│   ├── schemas/                 # Pydantic DTOs
│   ├── services/                # parsers, chunking, embeddings, llm, rag
│   └── main.py
├── tests/
├── alembic/                     # migrations (Sesión 2)
├── docker-compose.yml           # Postgres 16 + pgvector
└── requirements*.txt
```

---

## 📜 License

[MIT](LICENSE)

---

Built by **[Kevin Morales](https://github.com/KevinMM007)** as the third project of a backend portfolio aimed at remote LATAM / USA junior roles.
