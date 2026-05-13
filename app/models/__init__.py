"""SQLAlchemy ORM models.

Models added in subsequent sessions:
  - User      (Sesión 2: reused JWT auth pattern)
  - Document  (Sesión 2: uploaded file metadata, per-user)
  - Chunk     (Sesión 3: text chunk + Vector embedding for similarity search)
"""

from app.models.base import Base, TimestampMixin

__all__ = ["Base", "TimestampMixin"]
