from sqlmodel import SQLModel

# Import every model module so its tables register on SQLModel.metadata.
# Alembic's env.py imports `metadata` from here as its autogenerate target —
# if a model isn't imported here, Alembic won't see it.
from app.models import agent, conversation, document, tenant  # noqa: F401

metadata = SQLModel.metadata
