from __future__ import annotations

import io

import pymupdf
import pytest

from app.services import parsers


def test_parse_markdown_decodes_utf8() -> None:
    text = parsers.parse_markdown("# Título\n\nContenido en español.".encode())
    assert "Título" in text
    assert "español" in text


def test_parse_markdown_falls_back_on_non_utf8_bytes() -> None:
    # 0xFF is invalid in UTF-8; the fallback decoder replaces it with U+FFFD
    # so the upload pipeline doesn't reject an otherwise-readable doc.
    payload = b"valid prefix \xff invalid byte"
    out = parsers.parse_markdown(payload)
    assert "valid prefix" in out
    assert "invalid byte" in out


def test_parse_pdf_returns_text_pages_joined() -> None:
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), "Page one content.")
    p2 = doc.new_page()
    p2.insert_text((72, 72), "Page two content.")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    text = parsers.parse_pdf(buf.getvalue())
    assert "Page one content." in text
    assert "Page two content." in text
    # Pages are separated by a double newline so the chunker can use the
    # boundary as a natural split point.
    assert "\n\n" in text


def test_parse_pdf_raises_parse_error_on_garbage() -> None:
    with pytest.raises(parsers.ParseError):
        parsers.parse_pdf(b"definitely not a pdf")
