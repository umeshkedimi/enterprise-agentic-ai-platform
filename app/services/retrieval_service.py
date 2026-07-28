import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.schemas import Citation
from app.services.embedding_service import embed_text

DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 240


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


async def semantic_search(
    session: AsyncSession,
    query: str,
    *,
    collection_id: uuid.UUID,
    top_k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks (cosine similarity),
    restricted to fully-processed documents in the given collection.

    Scoping by collection is the retrieval-time isolation boundary: an agent
    searches only its own collection, so one team's documents can never surface
    in another's answer even though all chunks share one physical table.
    """
    query_embedding = await embed_text(query)
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    stmt = (
        select(DocumentChunk, Document.filename, distance.label("distance"))
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            Document.collection_id == collection_id,
            Document.status == DocumentStatus.READY,
        )
        .order_by(distance)
        .limit(top_k)
    )
    result = await session.execute(stmt)

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            filename=filename,
            content=chunk.content,
            # pgvector's cosine_distance is 1 - cosine_similarity
            score=1.0 - dist,
        )
        for chunk, filename, dist in result.all()
    ]


def to_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            document_id=c.document_id,
            chunk_id=c.chunk_id,
            snippet=c.content[:SNIPPET_LENGTH],
            score=c.score,
        )
        for c in chunks
    ]
