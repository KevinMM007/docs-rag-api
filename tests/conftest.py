"""Test fixtures.

For the smoke tests in this session we just need a FastAPI TestClient.
DB / vector-store fixtures land in Sesión 2 once we actually have models -
pgvector requires a real Postgres (SQLite has no native vector support),
so we'll plug in testcontainers-postgres at that point.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> Generator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client
