from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.core.config import get_settings
from app.db.base import metadata as sqlmodel_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Migrations run over a sync driver (psycopg), while the app itself uses the
# async driver (asyncpg) — Alembic doesn't need async, so keep them separate.
config.set_main_option("sqlalchemy.url", get_settings().database_url_sync)

target_metadata = sqlmodel_metadata


def include_object(object, name, type_, reflected, compare_to):
    """Keep autogenerate from proposing to drop objects it cannot represent.

    The pgvector HNSW index is created with raw SQL in a migration because its
    `USING hnsw (embedding vector_cosine_ops)` form has no SQLModel/SQLAlchemy
    metadata equivalent. Autogenerate therefore sees it only in the database,
    finds no match in the model metadata, and emits a `drop_index` — silently
    removing the index that makes semantic search fast. Excluding it here makes
    autogenerate leave vector indexes alone; they are owned by hand-written
    migrations, not by the model metadata.
    """
    if type_ == "index" and name and name.endswith("_hnsw"):
        return False
    return True


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_object=include_object,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=include_object,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
