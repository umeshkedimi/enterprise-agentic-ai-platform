import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics, tracing
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

    Instrumented here rather than in the graph's retrieve node, because this is
    also what the `search_knowledge_base` tool calls: measuring one caller would
    leave every model-initiated search out of the numbers.
    """
    started = time.perf_counter()
    # The span carries the collection id; the metrics deliberately do not. A
    # trace is where "which tenant's retrieval got slow" is answerable, and a
    # metric label is where asking that question would cost a time series per
    # collection forever.
    with tracing.span(
        "retrieval.search",
        **{tracing.COLLECTION_ID: str(collection_id), "eaap.retrieval.top_k": top_k},
    ) as current:
        try:
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
        except Exception:
            metrics.RETRIEVAL_REQUESTS.labels("error").inc()
            raise

        chunks = [
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

        metrics.RETRIEVAL_REQUESTS.labels("ok").inc()
        metrics.RETRIEVAL_DURATION.observe(time.perf_counter() - started)
        metrics.RETRIEVAL_CHUNKS.observe(len(chunks))
        top_score = chunks[0].score if chunks else None
        if top_score is not None:
            # Only when something came back. A search that found nothing has no
            # "best score" to record, and folding it in as a zero would drag the
            # distribution toward a floor that no real match ever sat at — which
            # is precisely the reading this histogram exists to get right.
            metrics.RETRIEVAL_TOP_SCORE.observe(top_score)

        tracing.set_attributes(
            current,
            **{
                tracing.RETRIEVED_CHUNKS: len(chunks),
                tracing.RETRIEVAL_TOP_SCORE: top_score,
            },
        )

    return chunks


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
