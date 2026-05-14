from __future__ import annotations

import io

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session


def _register_and_login(
    client: TestClient,
    email: str = "alice@example.com",
    password: str = "secret-pw-12",
) -> dict[str, str]:
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _make_pdf_bytes(text_body: str = "Hello world. " * 50) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text_body)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def test_upload_markdown_creates_document_and_chunks(
    client: TestClient, db: Session
) -> None:
    headers = _register_and_login(client)
    md = b"# Title\n\n" + b"Some prose for the embedder. " * 200

    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("notes.md", md, "text/markdown")},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["filename"] == "notes.md"
    assert body["content_type"] == "text/markdown"
    assert body["size_bytes"] == len(md)
    assert body["chunk_count"] >= 1

    # Confirm the chunks landed in the DB
    db_chunk_count = db.execute(
        text("SELECT COUNT(*) FROM chunks WHERE document_id = :d"),
        {"d": body["id"]},
    ).scalar()
    assert db_chunk_count == body["chunk_count"]


def test_upload_pdf_creates_document(client: TestClient) -> None:
    headers = _register_and_login(client)
    pdf_bytes = _make_pdf_bytes()

    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("paper.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["filename"] == "paper.pdf"
    assert body["content_type"] == "application/pdf"
    assert body["chunk_count"] >= 1


def test_upload_rejects_unsupported_type(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")},
    )
    assert response.status_code == 415


def test_upload_rejects_empty_file(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert response.status_code == 400


def test_upload_rejects_corrupt_pdf(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("broken.pdf", b"not a real pdf", "application/pdf")},
    )
    assert response.status_code == 422


def test_upload_requires_auth(client: TestClient) -> None:
    response = client.post(
        "/api/v1/documents/upload",
        files={"file": ("notes.md", b"hello", "text/markdown")},
    )
    assert response.status_code == 401


def test_list_only_returns_caller_documents(client: TestClient) -> None:
    alice_headers = _register_and_login(client, "alice@example.com")
    bob_headers = _register_and_login(client, "bob@example.com")

    client.post(
        "/api/v1/documents/upload",
        headers=alice_headers,
        files={"file": ("alice.md", b"# Alice notes\n\nhello world " * 10, "text/markdown")},
    )
    client.post(
        "/api/v1/documents/upload",
        headers=bob_headers,
        files={"file": ("bob.md", b"# Bob notes\n\nhello world " * 10, "text/markdown")},
    )

    response = client.get("/api/v1/documents", headers=alice_headers)
    assert response.status_code == 200
    docs = response.json()
    assert len(docs) == 1
    assert docs[0]["filename"] == "alice.md"


def test_get_document_returns_404_for_other_user(client: TestClient) -> None:
    alice_headers = _register_and_login(client, "alice@example.com")
    bob_headers = _register_and_login(client, "bob@example.com")

    created = client.post(
        "/api/v1/documents/upload",
        headers=alice_headers,
        files={"file": ("alice.md", b"# private " * 50, "text/markdown")},
    ).json()

    response = client.get(f"/api/v1/documents/{created['id']}", headers=bob_headers)
    assert response.status_code == 404


def test_delete_document_cascades_chunks(client: TestClient, db: Session) -> None:
    headers = _register_and_login(client)
    created = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("notes.md", b"# Doc\n\n" + b"content " * 200, "text/markdown")},
    ).json()
    doc_id = created["id"]

    before = db.execute(
        text("SELECT COUNT(*) FROM chunks WHERE document_id = :d"), {"d": doc_id}
    ).scalar()
    assert before > 0

    response = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
    assert response.status_code == 204

    after = db.execute(
        text("SELECT COUNT(*) FROM chunks WHERE document_id = :d"), {"d": doc_id}
    ).scalar()
    assert after == 0


def test_delete_returns_404_for_unknown_document(client: TestClient) -> None:
    headers = _register_and_login(client)
    response = client.delete("/api/v1/documents/99999", headers=headers)
    assert response.status_code == 404


def test_upload_returns_503_when_embedding_service_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    """If Gemini is down the upload must bail before persisting any chunks;
    a half-indexed document would silently misrank in similarity search."""
    from app.services import embeddings

    def boom(_texts):
        raise embeddings.EmbeddingError("simulated downstream failure")

    monkeypatch.setattr(embeddings, "embed_documents", boom)

    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("notes.md", b"# Title\n\nReal content for embedding.", "text/markdown")},
    )
    assert response.status_code == 503
    assert "Embedding service unavailable" in response.json()["detail"]


def test_upload_rejects_empty_pdf_with_no_text(client: TestClient) -> None:
    """A born-digital PDF with zero text pages is rejected as 422 - the same
    error path scanned PDFs hit, so OCR-needed docs surface clearly."""
    blank = pymupdf.open()
    blank.new_page()  # empty page, no text
    buf = io.BytesIO()
    blank.save(buf)
    blank.close()

    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("blank.pdf", buf.getvalue(), "application/pdf")},
    )
    assert response.status_code == 422
    assert "no extractable text" in response.json()["detail"]


def test_upload_infers_pdf_from_extension_when_mime_is_generic(
    client: TestClient,
) -> None:
    """Browsers occasionally send application/octet-stream for drag-drop;
    the endpoint must still accept the file based on its extension."""
    pdf_bytes = _make_pdf_bytes()
    headers = _register_and_login(client)
    response = client.post(
        "/api/v1/documents/upload",
        headers=headers,
        files={"file": ("paper.pdf", pdf_bytes, "application/octet-stream")},
    )
    assert response.status_code == 201
    assert response.json()["content_type"] == "application/pdf"


def test_search_returns_503_when_embedding_service_fails(
    client: TestClient,
    monkeypatch,
) -> None:
    from app.services import embeddings

    headers = _register_and_login(client)

    def boom(_text):
        raise embeddings.EmbeddingError("query embedding failed")

    monkeypatch.setattr(embeddings, "embed_query", boom)

    response = client.post(
        "/api/v1/documents/search",
        headers=headers,
        json={"query": "anything"},
    )
    assert response.status_code == 503
