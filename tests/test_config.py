"""Unit tests for the Pydantic Settings layer.

These run without the FastAPI TestClient since they're concerned with how
``Settings`` parses raw values - not with the rest of the app.
"""

from __future__ import annotations

from app.core.config import Settings


def test_database_url_keeps_explicit_driver_unchanged() -> None:
    s = Settings(database_url="postgresql+psycopg://u:p@h/db")
    assert s.database_url == "postgresql+psycopg://u:p@h/db"


def test_database_url_rewrites_postgres_scheme() -> None:
    # Render and Heroku hand out ``postgres://...``; the SQLAlchemy psycopg
    # driver requires the explicit ``postgresql+psycopg://`` form.
    s = Settings(database_url="postgres://u:p@h/db")
    assert s.database_url == "postgresql+psycopg://u:p@h/db"


def test_database_url_rewrites_bare_postgresql_scheme() -> None:
    s = Settings(database_url="postgresql://u:p@h/db")
    assert s.database_url == "postgresql+psycopg://u:p@h/db"


def test_cors_origins_list_with_wildcard() -> None:
    s = Settings(cors_origins="*")
    assert s.cors_origins_list == ["*"]


def test_cors_origins_list_with_csv() -> None:
    s = Settings(cors_origins="https://a.example, https://b.example,https://c.example")
    assert s.cors_origins_list == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


def test_cors_origins_list_ignores_blank_entries() -> None:
    s = Settings(cors_origins=" , https://a.example, , ")
    assert s.cors_origins_list == ["https://a.example"]


# The "non-string input is a noop" branch in ``_coerce_postgres_driver`` is
# defensive code: pydantic type-validates the field before our validator runs,
# so the branch is effectively unreachable from the public API. Skipping it
# rather than calling the wrapped descriptor directly keeps tests honest.