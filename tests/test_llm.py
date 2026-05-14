"""Unit tests for the Gemini chat wrapper.

The conftest autouse fixture patches ``llm._client`` (the SDK boundary), so
the real ``stream_chat`` function still runs end-to-end here. Tests that
need a specific SDK behaviour re-patch ``_client`` themselves; their patch
wins over the conftest fixture for the duration of the test.

To test the real ``_client`` body (which checks the API key before talking
to Google) we capture a reference at module import - that runs before any
autouse fixture, so the captured reference is the unpatched original.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import llm
from app.services.llm import _client as _ORIGINAL_CLIENT


def _fake_client(fragments: list[str | None]) -> SimpleNamespace:
    class _Stream:
        def __iter__(self):
            for fragment in fragments:
                yield SimpleNamespace(text=fragment)

    class _Models:
        def generate_content_stream(self, **_kw):
            return _Stream()

    return SimpleNamespace(models=_Models())


def test_stream_chat_yields_text_from_each_sdk_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llm, "_client", lambda: _fake_client(["Hello ", "world", None, "!"]))

    out = list(llm.stream_chat("prompt", system_instruction="be helpful"))
    # Events whose ``.text`` is empty/None are skipped so the SSE stream
    # never emits dangling ``{"type": "token", "content": ""}`` frames.
    assert out == ["Hello ", "world", "!"]


def test_stream_chat_wraps_sdk_failures_as_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def generate_content_stream(self, **_kw):
            raise RuntimeError("simulated 503")

    monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(models=_Boom()))

    with pytest.raises(llm.LLMError, match="Gemini chat failed"):
        list(llm.stream_chat("prompt"))


def test_client_rejects_unconfigured_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper must refuse to send the placeholder ``change-me`` key to
    Google's servers. We call the original ``_client`` captured at module
    import time, with ``get_settings`` patched to return the bad key.
    """
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: SimpleNamespace(
            gemini_api_key="change-me",
            gemini_chat_model="x",
            gemini_embedding_model="x",
        ),
    )
    with pytest.raises(llm.LLMError, match="not configured"):
        _ORIGINAL_CLIENT()
