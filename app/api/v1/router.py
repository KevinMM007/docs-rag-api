from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/ping", tags=["health"])
def ping() -> dict[str, str]:
    return {"status": "ok"}


# Routers added in subsequent sessions:
#   - auth.py       (Sesión 2: register / login / me)
#   - documents.py  (Sesión 2: upload / list / delete)
#   - chat.py       (Sesión 4: RAG question-answer with streaming)
