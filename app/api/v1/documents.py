from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from app.api.deps import CurrentUser, DbSession
from app.crud import document as crud_doc
from app.schemas.document import DocumentRead
from app.services import chunking, parsers

router = APIRouter(prefix="/documents", tags=["documents"])

# Accept the standard MIME types browsers send for these formats, plus
# octet-stream as a fallback for the cases where the browser doesn't send a
# type at all (we infer from extension below).
_PDF_TYPES = {"application/pdf"}
_MARKDOWN_TYPES = {"text/markdown", "text/x-markdown", "text/plain"}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB - portfolio-sized docs only.


def _infer_kind(content_type: str, filename: str) -> str | None:
    """Return ``"pdf"`` / ``"markdown"`` / ``None`` based on MIME type, falling
    back to file extension when the browser sends an empty or generic type.
    """
    if content_type in _PDF_TYPES:
        return "pdf"
    if content_type in _MARKDOWN_TYPES:
        return "markdown"
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if lower.endswith((".md", ".markdown", ".txt")):
        return "markdown"
    return None


@router.post(
    "/upload",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    db: DbSession,
    current_user: CurrentUser,
    file: UploadFile = File(...),
) -> DocumentRead:
    filename = file.filename or "untitled"
    content_type = file.content_type or ""
    kind = _infer_kind(content_type, filename)
    if kind is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {content_type or filename!r}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB limit",
        )

    try:
        text = parsers.parse_pdf(content) if kind == "pdf" else parsers.parse_markdown(content)
    except parsers.ParseError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document contains no extractable text",
        )

    chunks = chunking.chunk_text(text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document yielded no chunks",
        )

    # Normalise the stored MIME type so /list responses don't show
    # "application/octet-stream" for an obvious .md upload.
    stored_content_type = "application/pdf" if kind == "pdf" else "text/markdown"

    doc = crud_doc.create_with_chunks(
        db,
        user_id=current_user.id,
        filename=filename,
        content_type=stored_content_type,
        size_bytes=len(content),
        chunks=chunks,
    )
    return DocumentRead.model_validate(doc)


@router.get("", response_model=list[DocumentRead])
def list_documents(db: DbSession, current_user: CurrentUser) -> list[DocumentRead]:
    docs = crud_doc.list_for_user(db, current_user.id)
    return [DocumentRead.model_validate(d) for d in docs]


@router.get("/{doc_id}", response_model=DocumentRead)
def get_document(
    doc_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> DocumentRead:
    doc = crud_doc.get_for_user(db, doc_id, current_user.id)
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentRead.model_validate(doc)


@router.delete(
    "/{doc_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
def delete_document(
    doc_id: int,
    db: DbSession,
    current_user: CurrentUser,
) -> Response:
    if not crud_doc.delete_for_user(db, doc_id, current_user.id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
