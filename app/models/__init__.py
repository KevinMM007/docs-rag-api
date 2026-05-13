"""Re-export every model class so ``Base.metadata`` is fully populated by the
time Alembic's autogenerate or ``create_all`` inspect it.
"""

from app.models.base import Base, TimestampMixin
from app.models.user import User

__all__ = ["Base", "TimestampMixin", "User"]
