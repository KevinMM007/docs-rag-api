"""Service layer for RAG building blocks.

Modules added in subsequent sessions:
  - embeddings.py  (Sesión 3: Gemini text-embedding-004 wrapper)
  - llm.py         (Sesión 4: Gemini chat wrapper with streaming)
  - rag.py         (Sesión 4: orchestrates retrieval + prompt + generation)
"""

from app.services import chunking, parsers

__all__ = ["chunking", "parsers"]
