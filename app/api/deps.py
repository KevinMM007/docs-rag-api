"""FastAPI dependencies. Auth deps land in Sesión 2 with the auth router."""

from app.core.database import get_db

__all__ = ["get_db"]
