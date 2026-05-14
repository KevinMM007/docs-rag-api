"""Tests for the RAG orchestrator + the SSE chat endpoint.

Both the embedding SDK and the chat SDK are stubbed via autouse fixtures in
conftest, so these tests run fully offline and deterministically.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.services import llm, retrieval

# ---------------------------------------------------------------------------
# build_prompt: pure unit tests
# ---------------------------------------------------------------------------


def _hit(filename: str, content: str, idx: int = 0, distance: float = 0.1) -> retrieval.SearchHit:
    return retrieval.SearchHit(
        chunk_id=idx + 1,
        document_id=idx + 100,
        document_filename=filename,
        chunk_index=idx,
        content=content,
        distance=distance,
    )


def test_build_prompt_inlines_each_chunk_with_its_source() -> None:
    from app.services.rag import build_prompt

    hits = [
        _hit("a.md", "Apples are red.", idx=0),
        _hit("b.pdf", "Bananas are yellow.", idx=1),
    ]
    prompt = build_prompt("What colour is fruit?", hits)
    assert "[1] Source: a.md" in prompt
    assert "Apples are red." in prompt
    assert "[2] Source: b.pdf" in prompt
    assert "Bananas are yellow." in prompt
    assert "Question: What colour is fruit?" in prompt


def test_build_prompt_handles_empty_hits() -> None:
    from app.services.rag import build_prompt

    prompt = build_prompt("what?", [])
    assert "No relevant context" in prompt
    assert "Question: what?" in prompt


# ---------------------------------------------------------------------------
# Chat endpoint: SSE stream
# ---------------------------------------------------------------------------


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


def _upload_md(client: TestClient, headers: dict, *, filename: str, body: bytes) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": (filename, body, "text/markdown")},
    )
    assert response.status_code == 201, response.text


def _parse_sse_events(raw: str) -> list[dict]:
    """Split a raw SSE response body into JSON payloads."""
    events: list[dict] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # Each block is one or more ``data: <json>`` lines; our server only
        # emits a single data line per block, so a simple prefix strip works.
        data = "\n".join(
            line[len("data: ") :] for line in block.splitlines() if line.startswith("data: ")
        )
        if data:
            events.append(json.loads(data))
    return events


def test_chat_streams_sources_then_tokens_then_done(client: TestClient) -> None:
    headers = _register_and_login(client)
    _upload_md(client, headers, filename="notes.md", body=b"# Notes\n\nApples are red.")

    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"query": "What colour are apples?"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    # At least one token event in between
    assert any(t == "token" for t in types[1:-1])

    # The sources event carries the uploaded document
    sources = events[0]["sources"]
    assert len(sources) >= 1
    assert sources[0]["document_filename"] == "notes.md"

    # Tokens concatenated should equal the stubbed answer in conftest.
    answer = "".join(e["content"] for e in events if e["type"] == "token")
    assert answer == "This is a stubbed answer."


def test_chat_with_no_documents_still_streams(client: TestClient) -> None:
    """No retrieval hits -> empty sources event, but LLM still streams an answer
    (in production, the prompt instructs it to refuse; here the stub just
    emits its canned text)."""
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"query": "What does the doc say?"},
    )
    assert response.status_code == 200
    events = _parse_sse_events(response.text)
    assert events[0]["sources"] == []
    assert events[-1]["type"] == "done"


def test_chat_emits_error_event_when_llm_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = _register_and_login(client)
    _upload_md(client, headers, filename="notes.md", body=b"Some content here.")

    def failing_stream(*_a, **_kw) -> Iterator[str]:
        # Yield one token so the server has opened the SSE response, then blow up.
        yield "Partial "
        raise llm.LLMError("simulated downstream failure")

    monkeypatch.setattr(llm, "stream_chat", failing_stream)

    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"query": "anything"},
    )
    assert response.status_code == 200  # SSE already opened
    events = _parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "error" in types
    # Once an error is emitted, ``done`` must not follow - clients should
    # treat error as terminal.
    assert "done" not in types


def test_chat_requires_auth(client: TestClient) -> None:
    response = client.post("/api/v1/chat", json={"query": "hi"})
    assert response.status_code == 401


def test_chat_rejects_empty_query(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post("/api/v1/chat", headers=headers, json={"query": ""})
    assert response.status_code == 422


def test_chat_returns_503_when_embedding_fails(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An embedding outage must produce a clean 503 with a JSON body, not a
    half-opened SSE stream the browser would have to special-case."""
    from app.services import embeddings

    def boom(_text):
        raise embeddings.EmbeddingError("query embedding failed")

    monkeypatch.setattr(embeddings, "embed_query", boom)

    headers = _register_and_login(client)
    response = client.post("/api/v1/chat", headers=headers, json={"query": "hi"})
    assert response.status_code == 503
    assert "Embedding service unavailable" in response.json()["detail"]


def test_chat_top_k_override_is_passed_through(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {}

    real_search = retrieval.search_similar_chunks

    def spy(db, *, user_id, query_embedding, top_k):
        captured["top_k"] = top_k
        return real_search(
            db, user_id=user_id, query_embedding=query_embedding, top_k=top_k
        )

    monkeypatch.setattr(retrieval, "search_similar_chunks", spy)

    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/chat",
        headers=headers,
        json={"query": "hi", "top_k": 3},
    )
    assert response.status_code == 200
    assert captured["top_k"] == 3
