from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import Chunk, Document


def create_with_chunks(
    db: Session,
    *,
    user_id: int,
    filename: str,
    content_type: str,
    size_bytes: int,
    chunks: list[str],
    embeddings: list[list[float]] | None = None,
) -> Document:
    """Persist a document and its chunks (with optional embeddings) atomically.

    ``embeddings`` must be either ``None`` (chunks stored without vectors -
    useful for debug paths or migrations) or a list of the same length as
    ``chunks``. The caller is responsible for embedding generation; this CRUD
    layer stays pure SQL.
    """
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError(
            f"embeddings length ({len(embeddings)}) does not match chunks "
            f"length ({len(chunks)})"
        )

    doc = Document(
        user_id=user_id,
        filename=filename,
        content_type=content_type,
        size_bytes=size_bytes,
        chunk_count=len(chunks),
    )
    db.add(doc)
    db.flush()  # populate doc.id without committing

    for idx, content in enumerate(chunks):
        emb = embeddings[idx] if embeddings is not None else None
        db.add(
            Chunk(
                document_id=doc.id,
                chunk_index=idx,
                content=content,
                embedding=emb,
            )
        )

    db.commit()
    db.refresh(doc)
    return doc


def list_for_user(db: Session, user_id: int) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc())
    )
    return list(db.scalars(stmt))


def get_for_user(db: Session, doc_id: int, user_id: int) -> Document | None:
    stmt = select(Document).where(Document.id == doc_id, Document.user_id == user_id)
    return db.scalar(stmt)


def delete_for_user(db: Session, doc_id: int, user_id: int) -> bool:
    doc = get_for_user(db, doc_id, user_id)
    if doc is None:
        return False
    db.delete(doc)
    db.commit()
    return True
