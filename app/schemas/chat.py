from __future__ import annotations

from pydantic import BaseModel, Field


class ChatQuery(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    # 0 means "use settings.retrieval_top_k"; explicit cap so a runaway client
    # can't pull a thousand chunks per turn.
    top_k: int = Field(default=0, ge=0, le=50)
