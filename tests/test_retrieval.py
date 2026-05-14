"""Integration tests for similarity search.

The Gemini SDK is patched out at the conftest level (autouse fixture) with a
deterministic content-addressed stand-in. Identical input text yields
identical vectors, so a query equal to a stored chunk produces a cosine
distance of zero - which makes ordering assertions deterministic.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _register_and_login(
    client: TestClient,
    email: str = "alice@example.com",
    password: str = "secret-pw-12",
) -> dict[str, str]:
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _upload_markdown(
    client: TestClient,
    headers: dict[str, str],
    *,
    filename: str,
    body: bytes,
) -> dict:
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": (filename, body, "text/markdown")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_upload_persists_embeddings(client: TestClient, db) -> None:
    headers = _register_and_login(client)
    created = _upload_markdown(
        client,
        headers,
        filename="notes.md",
        body=b"# Air quality\n\n" + b"Particulate matter readings " * 80,
    )

    from sqlalchemy import text

    null_count = db.execute(
        text(
            "SELECT COUNT(*) FROM chunks "
            "WHERE document_id = :d AND embedding IS NULL"
        ),
        {"d": created["id"]},
    ).scalar()
    assert null_count == 0


def test_search_returns_most_similar_chunk_first(client: TestClient) -> None:
    headers = _register_and_login(client)
    target = "Air quality measurements in Xalapa from March 2026."
    decoy = "Database schema migration notes for legacy systems."
    _upload_markdown(client, headers, filename="target.md", body=target.encode())
    _upload_markdown(client, headers, filename="decoy.md", body=decoy.encode())

    response = client.post(
        "/api/v1/documents/search",
        headers=headers,
        json={"query": target},
    )
    assert response.status_code == 200, response.text
    hits = response.json()
    assert len(hits) >= 1
    # The chunk whose content equals the query should rank first with distance ~0.
    assert hits[0]["document_filename"] == "target.md"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_search_only_returns_caller_chunks(client: TestClient) -> None:
    alice = _register_and_login(client, "alice@example.com")
    bob = _register_and_login(client, "bob@example.com")
    _upload_markdown(client, alice, filename="alice.md", body=b"alice private content")
    _upload_markdown(client, bob, filename="bob.md", body=b"bob private content")

    response = client.post(
        "/api/v1/documents/search",
        headers=alice,
        json={"query": "bob private content"},
    )
    assert response.status_code == 200
    hits = response.json()
    # Bob's chunk would be the literal match, but alice must not see it.
    assert all(hit["document_filename"] == "alice.md" for hit in hits)


def test_search_respects_top_k_override(client: TestClient) -> None:
    headers = _register_and_login(client)
    # Upload three distinct documents so we have at least three chunks.
    for i in range(3):
        _upload_markdown(
            client,
            headers,
            filename=f"doc{i}.md",
            body=f"unique content number {i}".encode(),
        )

    response = client.post(
        "/api/v1/documents/search?top_k=1",
        headers=headers,
        json={"query": "anything"},
    )
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_search_with_no_documents_returns_empty(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/search",
        headers=headers,
        json={"query": "anything"},
    )
    assert response.status_code == 200
    assert response.json() == []


def test_search_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/search",
        json={"query": "anything"},
    )
    assert response.status_code == 401


def test_search_rejects_empty_query(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/search",
        headers=headers,
        json={"query": ""},
    )
    assert response.status_code == 422
