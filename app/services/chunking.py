"""Text chunker.

We approximate token counts as ``chars / 4`` instead of running the Gemini
tokenizer per chunk. The approximation is good enough because:

* The retrieval prompt is built later from the top-K chunks, not from a hard
  token budget here.
* The chunker only needs *roughly* uniform chunk sizes for embedding quality
  - swapping in a real tokenizer is a one-line change later if needed.

The algorithm is a sentence-aware sliding window: we walk through the text in
``size``-character windows, but if a chunk would land in the middle of a
sentence we pull the window back to the last sentence boundary in the second
half. Adjacent chunks share ``overlap`` characters so retrieval doesn't miss
context that straddles a boundary.
"""

from __future__ import annotations

from app.core.config import get_settings

_CHARS_PER_TOKEN = 4

_SENTENCE_TERMINATORS = (". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n")


def chunk_text(
    text: str,
    *,
    size_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[str]:
    settings = get_settings()
    size = (size_tokens or settings.chunk_size_tokens) * _CHARS_PER_TOKEN
    overlap = (overlap_tokens or settings.chunk_overlap_tokens) * _CHARS_PER_TOKEN
    if overlap >= size:
        raise ValueError("overlap_tokens must be smaller than size_tokens")

    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Look for a sentence boundary in the second half of the window so
            # we don't split mid-sentence. Only honor it if it's past the
            # midpoint - otherwise the chunk becomes too small.
            slice_ = text[start:end]
            best = -1
            for terminator in _SENTENCE_TERMINATORS:
                idx = slice_.rfind(terminator)
                if idx > best:
                    best = idx
            if best > size // 2:
                end = start + best + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        # Slide forward, leaving an ``overlap`` tail of the previous chunk.
        start = max(end - overlap, start + 1)

    return chunks
