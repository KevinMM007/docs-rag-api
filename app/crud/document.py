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
) -> Document:
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
        db.add(Chunk(document_id=doc.id, chunk_index=idx, content=content))

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
