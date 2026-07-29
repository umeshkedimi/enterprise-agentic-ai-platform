"""Process-wide ownership of the graph's Postgres checkpointer.

The checkpointer stores *execution* state — the scratchpad of a tool loop, the
calls still pending, how many steps have run — keyed by thread. It is what makes
a turn resumable rather than restartable, and what a future interrupt (an
approval step before a tool runs) suspends into. It is deliberately not the
conversation store: `conversation_messages` is, because a checkpoint is keyed by
an opaque thread id and cannot answer "which tenant owns this".

Two things are worth knowing about the connection here:

* It is psycopg, not asyncpg, and not the SQLAlchemy engine. LangGraph's saver
  speaks psycopg directly, so this pool is separate from `app/db/session.py` by
  necessity rather than by choice. Both point at the same database.
* The tables are created and versioned by LangGraph, not by Alembic. That is why
  `alembic/env.py` excludes them from autogenerate — owning their schema would
  pin the project to one library version forever, and a generated `drop_table`
  would take every in-flight conversation with it.
"""

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.agents.graph import clear_checkpointer, set_checkpointer
from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_pool: AsyncConnectionPool | None = None


def _conninfo(settings: Settings) -> str:
    # SQLAlchemy's `postgresql+psycopg://` names a dialect and a driver; psycopg
    # wants a plain libpq URI and rejects the driver half.
    return settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")


async def start_checkpointer(settings: Settings) -> None:
    """Open the pool, apply LangGraph's own migrations, and recompile the graph.

    Failure is not fatal. A platform that refuses to start because conversation
    *memory* is unavailable trades a degraded feature for a total outage — and
    the degraded mode here is well-defined: turns still run, history still
    replays from the transcript, and only mid-turn resumption is lost.
    """
    global _pool

    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    pool = AsyncConnectionPool(
        conninfo=_conninfo(settings),
        max_size=settings.checkpointer_pool_size,
        # autocommit because the saver runs its own transactions and nests badly
        # inside an outer one; dict_row because it reads rows by column name;
        # prepare_threshold=0 because a pooled connection behind PgBouncer cannot
        # rely on a server-side prepared statement surviving between checkouts.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=settings.checkpointer_open_timeout)
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
    except Exception as exc:  # noqa: BLE001 - driver and libpq errors are many types
        logger.warning("checkpointer_unavailable", error=type(exc).__name__, detail=str(exc))
        await pool.close()
        return

    _pool = pool
    set_checkpointer(saver)
    logger.info("checkpointer_started", pool_size=settings.checkpointer_pool_size)


async def stop_checkpointer() -> None:
    global _pool

    clear_checkpointer()
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("checkpointer_stopped")
