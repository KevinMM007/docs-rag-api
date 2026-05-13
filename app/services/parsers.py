"""Document parsers: PDF (PyMuPDF) and Markdown / plain text.

The parsers return raw text; chunking happens downstream in ``chunking.py``.
PyMuPDF gives noticeably better text extraction than pypdf on multi-column
PDFs and embedded-font PDFs, which is exactly the kind of messy file users
upload in practice.
"""

from __future__ import annotations

import pymupdf  # PyMuPDF; imported as ``pymupdf`` since v1.24


class ParseError(ValueError):
    """Raised when a document is malformed or yields no extractable text."""


def parse_pdf(content: bytes) -> str:
    try:
        doc = pymupdf.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ParseError(f"Not a valid PDF: {exc}") from exc
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    # Double newline between pages so the chunker can use page breaks as a
    # natural split point.
    return "\n\n".join(pages)


def parse_markdown(content: bytes) -> str:
    # Keep the raw markdown syntax (``# heading``, ``**bold**``) intact. Those
    # tokens carry semantic weight that helps the embedding model distinguish
    # structural elements from prose - stripping them would lose signal.
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        # Fall back to a permissive decode so a stray non-UTF-8 byte doesn't
        # kill an otherwise valid upload.
        return content.decode("utf-8", errors="replace")
