"""Alembic environment.

Pulls DATABASE_URL from app settings (so .env / Render env vars work) and
imports our SQLAlchemy Base so `alembic revision --autogenerate` sees every
model declared in app.models.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.core.config import get_settings
from app.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the runtime URL into the alembic config so both offline and online
# modes see the same value. Tests override this via -x db_url=... below.
settings = get_settings()
runtime_url = context.get_x_argument(as_dictionary=True).get("db_url") or settings.database_url
config.set_main_option("sqlalchemy.url", runtime_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=runtime_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
