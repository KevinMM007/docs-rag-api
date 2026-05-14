"""Service layer for RAG building blocks.

Modules added in subsequent sessions:
  - llm.py         (Sesión 4: Gemini chat wrapper with streaming SSE)
  - rag.py         (Sesión 4: orchestrates retrieval + prompt + generation)
"""

from app.services import chunking, embeddings, parsers, retrieval

__all__ = ["chunking", "embeddings", "parsers", "retrieval"]
