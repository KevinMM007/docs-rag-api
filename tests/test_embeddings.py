"""Tests for the Gemini embedding wrapper.

We never hit the real API in tests - it would burn the free-tier rate limit
and turn CI non-deterministic. Every call to ``_embed`` is monkey-patched.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import embeddings
from app.services.embeddings import _embed as _ORIGINAL_EMBED


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


def test_client_rejects_unconfigured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper must refuse to call Gemini with the placeholder key
    rather than sending ``change-me`` to Google's servers.

    Note: ``embeddings.py`` does ``from app.core.config import get_settings``,
    so the binding to patch is the one imported into the embeddings module -
    patching ``app.core.config.get_settings`` would not be picked up.
    """
    from types import SimpleNamespace

    monkeypatch.setattr(
        embeddings,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_api_key="change-me",
            gemini_chat_model="x",
            gemini_embedding_model="x",
        ),
    )
    with pytest.raises(embeddings.EmbeddingError):
        embeddings._client()


def test_l2_normalize_produces_unit_vector() -> None:
    vec = embeddings._l2_normalize([3.0, 4.0])
    # 3-4-5 triangle -> normalised to [0.6, 0.8], magnitude 1.0
    assert vec[0] == pytest.approx(0.6)
    assert vec[1] == pytest.approx(0.8)


def test_l2_normalize_handles_zero_vector() -> None:
    # A degenerate all-zero embedding (shouldn't happen in practice but the
    # division would crash if we didn't guard it) is returned unchanged.
    assert embeddings._l2_normalize([0.0, 0.0]) == [0.0, 0.0]


def test_real_embed_normalises_and_validates_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real ``_embed`` body (which the conftest stub replaces
    in normal tests) by patching only ``_client``. Verifies the L2
    normalisation pass and the dimension sanity check both run."""
    # Build a fake genai client whose ``embed_content`` returns a vector with
    # a known magnitude so we can assert normalisation kicks in.
    raw = [3.0, 4.0] + [0.0] * (embeddings.EMBEDDING_DIM - 2)  # magnitude 5

    class _FakeModels:
        def embed_content(self, **_kw):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=raw)])

    monkeypatch.setattr(embeddings, "_client", lambda: SimpleNamespace(models=_FakeModels()))

    out = _ORIGINAL_EMBED(["hello"], task_type="RETRIEVAL_DOCUMENT")
    assert len(out) == 1
    assert len(out[0]) == embeddings.EMBEDDING_DIM
    # After L2 normalisation a 3-4-...-0 vector should have x=0.6, y=0.8.
    assert out[0][0] == pytest.approx(0.6)
    assert out[0][1] == pytest.approx(0.8)


def test_real_embed_raises_when_dim_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Gemini ever returns a vector of the wrong size, the wrapper must
    refuse rather than persist a vector that won't fit the ``vector(768)``
    column."""

    class _FakeModels:
        def embed_content(self, **_kw):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0, 3.0])])

    monkeypatch.setattr(embeddings, "_client", lambda: SimpleNamespace(models=_FakeModels()))

    with pytest.raises(embeddings.EmbeddingError, match="wrong dimension"):
        _ORIGINAL_EMBED(["hello"], task_type="RETRIEVAL_DOCUMENT")
