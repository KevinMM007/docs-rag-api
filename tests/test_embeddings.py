"""Tests for the Gemini embedding wrapper.

We never hit the real API in tests - it would burn the free-tier rate limit
and turn CI non-deterministic. Every call to ``_embed`` is monkey-patched.
"""

from __future__ import annotations

import pytest

from app.services import embeddings


def _fake_vector(seed: int) -> list[float]:
    return [(seed % 17 + i) / 1000.0 for i in range(embeddings.EMBEDDING_DIM)]


def test_embed_documents_returns_vector_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(texts: list[str], *, task_type: str) -> list[list[float]]:
        assert task_type == "RETRIEVAL_DOCUMENT"
        return [_fake_vector(i) for i, _ in enumerate(texts)]

    monkeypatch.setattr(embeddings, "_embed", fake)

    result = embeddings.embed_documents(["a", "b", "c"])
    assert len(result) == 3
    assert all(len(v) == embeddings.EMBEDDING_DIM for v in result)


def test_embed_documents_empty_input_skips_api(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fake(*_a, **_kw):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(embeddings, "_embed", fake)
    assert embeddings.embed_documents([]) == []
    assert called is False


def test_embed_query_uses_query_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake(texts: list[str], *, task_type: str) -> list[list[float]]:
        captured["task_type"] = task_type
        return [_fake_vector(7)]

    monkeypatch.setattr(embeddings, "_embed", fake)

    vec = embeddings.embed_query("what does the doc say?")
    assert captured["task_type"] == "RETRIEVAL_QUERY"
    assert len(vec) == embeddings.EMBEDDING_DIM


def test_embed_query_rejects_blank_input() -> None:
    with pytest.raises(ValueError):
        embeddings.embed_query("   ")


def test_retry_eventually_raises_embedding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def always_fails(*_a, **_kw):
        attempts["n"] += 1
        raise RuntimeError("simulated 503")

    monkeypatch.setattr(embeddings, "_embed", always_fails)
    # Speed up the test by collapsing the backoff sleeps.
    monkeypatch.setattr(embeddings.time, "sleep", lambda _s: None)

    with pytest.raises(embeddings.EmbeddingError):
        embeddings.embed_documents(["hello"])

    assert attempts["n"] == 3  # default max_retries


def test_retry_recovers_on_second_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"n": 0}

    def flaky(texts: list[str], *, task_type: str) -> list[list[float]]:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return [_fake_vector(1) for _ in texts]

    monkeypatch.setattr(embeddings, "_embed", flaky)
    monkeypatch.setattr(embeddings.time, "sleep", lambda _s: None)

    result = embeddings.embed_documents(["text"])
    assert len(result) == 1
    assert attempts["n"] == 2
