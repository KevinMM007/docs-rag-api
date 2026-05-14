"""Gemini chat wrapper with token streaming.

The HTTP layer needs incremental output so the user sees the answer build up
in the browser. Gemini's ``generate_content_stream`` yields one event per
chunk - we surface the text portions as a plain ``Iterator[str]`` so the SSE
generator in the chat endpoint stays simple.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Final

from google import genai
from google.genai import types as genai_types

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# Low temperature keeps the model anchored to the retrieved context; higher
# values let it free-associate, which is the opposite of what we want for RAG.
_DEFAULT_TEMPERATURE: Final[float] = 0.2


class LLMError(RuntimeError):
    """Raised when the Gemini chat SDK fails."""


def _client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key or settings.gemini_api_key == "change-me":
        raise LLMError("GEMINI_API_KEY is not configured")
    return genai.Client(api_key=settings.gemini_api_key)


def stream_chat(
    prompt: str,
    *,
    system_instruction: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> Iterator[str]:
    """Yield successive text fragments from a single-turn Gemini chat call.

    Caller is responsible for prompt construction (RAG context injection,
    etc.) - this function only owns the SDK glue. Any SDK exception is
    re-raised as ``LLMError`` so the HTTP layer can map it cleanly.
    """
    settings = get_settings()
    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
    )
    try:
        stream = _client().models.generate_content_stream(
            model=settings.gemini_chat_model,
            contents=prompt,
            config=config,
        )
        for event in stream:
            text = getattr(event, "text", None)
            if text:
                yield text
    except Exception as exc:
        logger.exception("Gemini chat stream failed")
        raise LLMError(f"Gemini chat failed: {exc}") from exc
