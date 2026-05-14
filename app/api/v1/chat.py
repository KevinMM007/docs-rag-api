"""RAG chat endpoint with Server-Sent Events streaming.

The endpoint sends three event shapes on ``text/event-stream``:

* ``{"type": "sources", "sources": [...]}``  - emitted first so the UI can
  show citations while the LLM is still warming up.
* ``{"type": "token", "content": "..."}``    - one per chunk from Gemini.
* ``{"type": "done"}`` or ``{"type": "error", "message": "..."}``.

We deliberately keep a single endpoint instead of "kick off + poll" because
chat answers are short and the SSE flow is what browsers expect from a
modern LLM UI.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, DbSession
from app.core.config import get_settings
from app.schemas.chat import ChatQuery
from app.services import embeddings, llm, rag

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(payload: dict) -> str:
    """Encode one SSE event. Each event ends in a blank line per the spec."""
    return f"data: {json.dumps(payload)}\n\n"


@router.post(
    "",
    responses={
        200: {
            "description": "Server-Sent Events stream of sources + tokens",
            "content": {"text/event-stream": {}},
        },
    },
)
def chat(
    payload: ChatQuery,
    db: DbSession,
    current_user: CurrentUser,
) -> StreamingResponse:
    settings = get_settings()
    top_k = payload.top_k or settings.retrieval_top_k

    # Retrieval runs synchronously *before* the streaming response opens. If
    # embedding the query or hitting Postgres fails we want a clean 503 over
    # JSON, not a half-opened SSE stream that the browser would have to
    # special-case.
    try:
        answer = rag.answer_question(
            db,
            user_id=current_user.id,
            query=payload.query,
            top_k=top_k,
        )
    except embeddings.EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embedding service unavailable: {exc}",
        ) from exc

    def event_stream() -> Iterator[str]:
        yield _sse(
            {
                "type": "sources",
                "sources": [
                    {
                        "chunk_id": h.chunk_id,
                        "document_id": h.document_id,
                        "document_filename": h.document_filename,
                        "chunk_index": h.chunk_index,
                        "distance": h.distance,
                    }
                    for h in answer.sources
                ],
            }
        )
        try:
            for token in answer.tokens:
                yield _sse({"type": "token", "content": token})
        except llm.LLMError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        # ``X-Accel-Buffering: no`` disables nginx/uvicorn proxy buffering so
        # tokens reach the browser as Gemini emits them, not at end-of-response.
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
