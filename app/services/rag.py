"""RAG orchestrator: glue retrieval + prompt template + LLM streaming."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services import embeddings, llm, retrieval

# The system instruction is the single biggest lever for answer quality.
# Three concerns it tries to enforce:
#   1. Hallucinations: tell the model to refuse rather than invent.
#   2. Citations: the user must be able to verify any claim against a chunk.
#   3. Conciseness: long answers wander; short ones stay on the context.
SYSTEM_INSTRUCTION = """You are a helpful assistant that answers questions about
the user's uploaded documents.

Rules:
- Answer ONLY using the context provided in the user's message.
- If the context does not contain enough information to answer confidently,
  reply: "I cannot answer this from the provided documents."
- Cite the source filename in square brackets after each claim, e.g. [paper.pdf].
- Be concise. No preamble, no "Based on the context...", no summaries of the
  question - just the answer."""


@dataclass(slots=True, frozen=True)
class RagAnswer:
    """What the chat endpoint streams back to the client.

    ``sources`` is yielded first as a single SSE event so the UI can render
    citations before tokens start arriving; ``tokens`` is the streaming
    text body.
    """

    sources: list[retrieval.SearchHit]
    tokens: Iterator[str]


def build_prompt(query: str, hits: list[retrieval.SearchHit]) -> str:
    """Render retrieved chunks + the user's question into the LLM input."""
    if not hits:
        context = "(No relevant context was retrieved for this question.)"
    else:
        blocks = []
        for i, hit in enumerate(hits, start=1):
            blocks.append(
                f"[{i}] Source: {hit.document_filename}\n{hit.content.strip()}"
            )
        context = "\n\n---\n\n".join(blocks)
    return (
        "Context:\n"
        f"{context}\n\n"
        "---\n"
        f"Question: {query.strip()}\n\n"
        "Answer:"
    )


def answer_question(
    db: Session,
    *,
    user_id: int,
    query: str,
    top_k: int,
) -> RagAnswer:
    """End-to-end: embed query, retrieve, prompt, stream the answer."""
    query_vec = embeddings.embed_query(query)
    hits = retrieval.search_similar_chunks(
        db,
        user_id=user_id,
        query_embedding=query_vec,
        top_k=top_k,
    )
    prompt = build_prompt(query, hits)
    tokens = llm.stream_chat(prompt, system_instruction=SYSTEM_INSTRUCTION)
    return RagAnswer(sources=hits, tokens=tokens)
