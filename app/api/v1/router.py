from fastapi import APIRouter

from app.api.v1 import auth

api_router = APIRouter()

api_router.include_router(auth.router)


@api_router.get("/ping", tags=["health"])
def ping() -> dict[str, str]:
    return {"status": "ok"}


# Routers added in subsequent sessions:
#   - documents.py  (Sesión 2: upload / list / delete)
#   - chat.py       (Sesión 4: RAG question-answer with streaming)
