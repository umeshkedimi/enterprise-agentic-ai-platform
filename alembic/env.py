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

# Tables LangGraph's Postgres checkpointer creates and versions itself, via its
# own `setup()` migration chain. They exist in the database and will never exist
# in SQLModel metadata, which is exactly the shape autogenerate reacts to by
# emitting `drop_table`.
LANGGRAPH_TABLES = {
    "checkpoints",
    "checkpoint_blobs",
    "checkpoint_writes",
    "checkpoint_migrations",
}


def include_object(object, name, type_, reflected, compare_to):
    """Keep autogenerate from proposing to drop objects it cannot represent.

    Two kinds of object live in the database without a metadata equivalent, and
    autogenerate's default reading of "in the DB, not in the models" is "drop
    it" for both:

    * The pgvector HNSW index, created with raw SQL because its
      `USING hnsw (embedding vector_cosine_ops)` form has no SQLModel/SQLAlchemy
      equivalent. Dropping it silently turns semantic search into a sequential
      scan — a performance failure with no error attached.
    * The checkpointer's tables, which LangGraph creates and migrates itself. We
      do not own their schema and must not: hand-writing migrations for them
      would pin us to one library version forever, and a dropped checkpoint
      table takes every in-flight conversation with it.

    Both are excluded here. Neither is unmanaged — each is owned by something
    other than this metadata.
    """
    if type_ == "index" and name and name.endswith("_hnsw"):
        return False
    if type_ == "table" and name in LANGGRAPH_TABLES:
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
