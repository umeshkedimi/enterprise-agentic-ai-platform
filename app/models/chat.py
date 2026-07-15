import uuid
from datetime import UTC, datetime

from sqlalchemy import Column, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"

    id: str = Field(primary_key=True)  # == session_id from the API
    tenant_id: str = Field(default="default", index=True)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    session_id: str = Field(foreign_key="chat_sessions.id", index=True)
    role: str  # "user" | "assistant" | "tool"
    content: str = Field(sa_column=Column(Text, nullable=False))
    tokens_used: int | None = None
    message_metadata: dict = Field(
        default_factory=dict, sa_column=Column("metadata", JSONB, nullable=False)
    )
    created_at: datetime = Field(default_factory=_utcnow)
