from __future__ import annotations

import pytest

from app.services.chunking import chunk_text


def test_empty_text_returns_empty_list() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_returns_single_chunk() -> None:
    text = "A short paragraph that easily fits in one chunk."
    chunks = chunk_text(text, size_tokens=500, overlap_tokens=80)
    assert chunks == [text]


def test_long_text_produces_multiple_chunks() -> None:
    text = ("Sentence one is here. " * 200).strip()
    chunks = chunk_text(text, size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    # Every chunk fits roughly inside the configured size window
    # (size_tokens * 4 chars/token = 400 chars, allow some slack).
    assert all(len(c) <= 500 for c in chunks)


def test_chunks_overlap() -> None:
    text = ("Hello world. " * 200).strip()
    chunks = chunk_text(text, size_tokens=100, overlap_tokens=20)
    # Adjacent chunks should share some tail/head text.
    assert any(chunks[i][-40:].strip() and chunks[i][-40:] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_invalid_overlap_raises() -> None:
    with pytest.raises(ValueError):
        chunk_text("anything", size_tokens=50, overlap_tokens=50)
