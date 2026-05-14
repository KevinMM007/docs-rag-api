from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    content_type: str
    size_bytes: int
    chunk_count: int
    created_at: datetime


class SearchQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


class ChunkSearchResult(BaseModel):
    chunk_id: int
    document_id: int
    document_filename: str
    chunk_index: int
    content: str
    # Cosine distance. 0.0 = identical, 2.0 = opposite. Lower = more similar.
    distance: float
