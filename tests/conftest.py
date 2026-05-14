"""Test fixtures.

Spins up a real Postgres + pgvector container via testcontainers per test
session and runs the full Alembic migration stack against it. Each test gets
a clean ``users`` table (TRUNCATE between tests) and a TestClient with the
``get_db`` dependency overridden to point at the test DB.

A SQLite fallback would force us to mock pgvector and hide bugs that only
manifest against the real backend, so we accept the ~5s startup cost.
"""

from __future__ import annotations

import argparse
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from alembic import command
from alembic.config import Config


@pytest.fixture(scope="session")
def _pg_container() -> Generator[PostgresContainer]:
    # Same image as docker-compose so tests and dev hit identical Postgres
    # + pgvector builds. The extension is preinstalled but not enabled - the
    # first Alembic migration does that.
    with PostgresContainer(
        "pgvector/pgvector:pg16",
        username="test",
        password="test",
        dbname="test",
        driver="psycopg",
    ) as pg:
        yield pg


@pytest.fixture(scope="session")
def db_engine(_pg_container: PostgresContainer) -> Generator[Engine]:
    url = _pg_container.get_connection_url()

    # env.py reads ``-x db_url=...`` first, then falls back to settings.
    # cmd_opts.x is the in-process equivalent of the CLI ``-x`` switch.
    cfg = Config("alembic.ini")
    cfg.cmd_opts = argparse.Namespace(x=[f"db_url={url}"])
    command.upgrade(cfg, "head")

    engine = create_engine(url, pool_pre_ping=True, future=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db(db_engine: Engine) -> Generator[Session]:
    SessionLocal = sessionmaker(
        bind=db_engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    session = SessionLocal()
    # Per-test cleanup: the app commits inside register/login, so a
    # transactional rollback approach would fight FastAPI's own commits.
    # TRUNCATE is fast and resets ``id`` sequences for predictable assertions.
    # CASCADE on the users truncate would clear documents+chunks via FK, but
    # being explicit makes the per-test fixture independent of FK ordering.
    session.execute(
        text("TRUNCATE TABLE chunks, documents, users RESTART IDENTITY CASCADE")
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db: Session) -> Generator[TestClient]:
    # Import inside the fixture so module collection doesn't pull in the app
    # (and its settings) before the test container is up.
    from app.api.deps import get_db
    from app.main import app

    def _override_get_db() -> Generator[Session]:
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def stub_gemini_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the Gemini SDK call site with a deterministic stand-in for
    every test. Two motivations:

    * CI never burns the free-tier rate limit and tests stay offline-safe.
    * Identical input text produces identical vectors, so similarity tests
      can assert exact ordering (a query equal to a chunk gets distance 0).

    Tests that want to exercise the wrapper itself (retry, error handling)
    can re-patch ``embeddings._embed`` from inside the test; pytest's
    monkeypatch composes correctly.
    """
    import hashlib
    import math

    from app.services import embeddings as emb

    def _vector_from_text(text: str) -> list[float]:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        raw = list(digest) * (emb.EMBEDDING_DIM // len(digest) + 1)
        vec = [b / 255.0 for b in raw[: emb.EMBEDDING_DIM]]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def fake_embed(texts: list[str], *, task_type: str) -> list[list[float]]:
        return [_vector_from_text(t) for t in texts]

    monkeypatch.setattr(emb, "_embed", fake_embed)


@pytest.fixture(autouse=True)
def stub_gemini_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the chat-streaming SDK call with a deterministic generator.

    The fake echoes a short canned answer split into a few tokens so SSE
    tests can assert frame-by-frame ordering without flaky timing or burning
    the Gemini free-tier quota.

    Tests that want to exercise error paths can monkey-patch
    ``llm.stream_chat`` from inside the test body.
    """
    from collections.abc import Iterator

    from app.services import llm as llm_mod

    def fake_stream_chat(
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        # A handful of fragments mimics Gemini's typical multi-token chunking
        # so generator-based tests exercise the loop multiple times.
        yield from ("This is ", "a stubbed ", "answer.")

    monkeypatch.setattr(llm_mod, "stream_chat", fake_stream_chat)
