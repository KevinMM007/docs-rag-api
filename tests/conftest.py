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
    session.execute(text("TRUNCATE TABLE users RESTART IDENTITY CASCADE"))
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
