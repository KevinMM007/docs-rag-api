"""Similarity search over a user's chunks.

Plain SQLAlchemy + pgvector. We deliberately keep this module thin so the
RAG orchestrator in Sesión 4 can compose retrieval with a prompt template
and an LLM call without having to talk to the DB directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Chunk, Document


@dataclass(slots=True, frozen=True)
class SearchHit:
    chunk_id: int
    document_id: int
    document_filename: str
    chunk_index: int
    content: str
    # Cosine distance: 0.0 = identical direction, 2.0 = opposite. Lower = better.
    distance: float


def search_similar_chunks(
    db: Session,
    *,
    user_id: int,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[SearchHit]:
    """Return the top-K chunks across the user's documents, ranked by cosine
    distance to ``query_embedding``. Chunks with NULL embeddings are skipped.
    """
    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    stmt = (
        select(
            Chunk.id,
            Chunk.document_id,
            Document.filename,
            Chunk.chunk_index,
            Chunk.content,
            distance,
        )
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.user_id == user_id)
        .where(Chunk.embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    return [
        SearchHit(
            chunk_id=row.id,
            document_id=row.document_id,
            document_filename=row.filename,
            chunk_index=row.chunk_index,
            content=row.content,
            distance=float(row.distance),
        )
        for row in rows
    ]
