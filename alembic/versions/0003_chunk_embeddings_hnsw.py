"""add embedding column to chunks with HNSW index

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-13
"""

from __future__ import annotations

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

# Must match app.models.document.EMBEDDING_DIM. Hard-coded here so the
# migration doesn't import application code (Alembic best practice -
# migrations should be reproducible even after the source moves).
EMBEDDING_DIM = 768


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
    )
    # HNSW with cosine_ops matches Gemini's recommended retrieval distance.
    # Building the index now (rather than at first query) means cold-start
    # latency is paid up-front during deploys, not per user request.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw "
        "ON chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.drop_column("chunks", "embedding")
