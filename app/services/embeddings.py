"""Gemini ``text-embedding-004`` wrapper.

One module owns the SDK so the rest of the app only sees plain
``list[float]`` vectors. Retries are local: a transient 5xx from Gemini
should not blow up a user's upload.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Final

from google import genai
from google.genai import types as genai_types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Must match the ``vector(N)`` column dim in migration 0003.
# gemini-embedding-001 returns 3072-dim vectors by default but supports
# Matryoshka-reduced output dimensions; 768 keeps our schema slim and the
# similarity quality remains strong at that size.
EMBEDDING_DIM: Final[int] = 768

# Gemini's documented retrieval task types. Document and query vectors live
# in the same space but the model nudges them in opposite directions, which
# improves asymmetric similarity (query -> document).
_DOC_TASK = "RETRIEVAL_DOCUMENT"
_QUERY_TASK = "RETRIEVAL_QUERY"


class EmbeddingError(RuntimeError):
    """Raised when the Gemini SDK fails repeatedly."""


def _client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "change-me":
        raise EmbeddingError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=settings.gemini_api_key)


def _l2_normalize(vec: list[float]) -> list[float]:
    """Unit-normalise a vector. Required when using Matryoshka-reduced
    embeddings - Gemini does not normalise the truncated output, and
    pgvector's cosine distance gives more stable rankings on unit vectors.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def _embed(texts: list[str], *, task_type: str) -> list[list[float]]:
    settings = get_settings()
    config = genai_types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
    )
    result = _client().models.embed_content(
        model=settings.gemini_embedding_model,
        contents=texts,
        config=config,
    )
    vectors = [_l2_normalize(list(e.values)) for e in result.embeddings]
    if any(len(v) != EMBEDDING_DIM for v in vectors):
        raise EmbeddingError(
            f"Gemini returned an embedding with the wrong dimension; "
            f"expected {EMBEDDING_DIM}"
        )
    return vectors


def _with_retry(
    func,
    *args,
    max_retries: int = 3,
    base_delay: float = 0.5,
    **kwargs,
):
    """Exponential-backoff retry. We swallow any SDK exception type because
    google-genai surfaces several (APIError, ServerError, ResourceExhausted)
    that we'd otherwise have to enumerate and pin to a SDK version.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Embedding call failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1,
                    max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise EmbeddingError(
        f"Gemini embedding failed after {max_retries} attempts: {last_exc}"
    ) from last_exc


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a batch of chunks for storage. Caller is responsible for keeping
    the input list size within Gemini's per-request limit (~100 items at the
    time of writing); the upload pipeline already splits documents into
    portfolio-sized chunks so this is not a practical concern.
    """
    if not texts:
        return []
    return _with_retry(_embed, texts, task_type=_DOC_TASK)


def embed_query(text: str) -> list[float]:
    """Embed a single user query for similarity search."""
    if not text.strip():
        raise ValueError("query must be non-empty")
    vectors = _with_retry(_embed, [text], task_type=_QUERY_TASK)
    return vectors[0]
