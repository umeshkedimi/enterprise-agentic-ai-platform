import time
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics, tracing
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.schemas import Citation
from app.services.embedding_service import embed_text

logger = get_logger(__name__)

DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 240


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    score: float  # cosine similarity in [-1, 1]; higher is more relevant


def _above_floor(chunks: list[RetrievedChunk], floor: float | None) -> list[RetrievedChunk]:
    """Keep only chunks that clear the configured relevance floor.

    A pure function on purpose: what counts as "above" is arithmetic, not a
    database concern, and testing it doesn't need pgvector. `None` means no
    floor is configured — the exact previous behaviour, unchanged.
    """
    if floor is None:
        return chunks
    return [c for c in chunks if c.score >= floor]


async def semantic_search(
    session: AsyncSession,
    query: str,
    *,
    collection_id: uuid.UUID,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
) -> list[RetrievedChunk]:
    """Embed the query and return the top_k most similar chunks (cosine similarity),
    restricted to fully-processed documents in the given collection.

    Scoping by collection is the retrieval-time isolation boundary: an agent
    searches only its own collection, so one team's documents can never surface
    in another's answer even though all chunks share one physical table.

    Instrumented here rather than in the graph's retrieve node, because this is
    also what the `search_knowledge_base` tool calls: measuring one caller would
    leave every model-initiated search out of the numbers. The relevance floor
    lives here for the same reason — applying it in the graph would leave a
    model-initiated reformulated search free to use evidence too weak to trust.
    """
    settings = settings or get_settings()
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
                    # A superseded version's chunks stay in the table for audit
                    # (see app/models/document.py) but must never compete for a
                    # search result — a re-uploaded policy is not "one more
                    # source", it is the only source, and the whole point of
                    # versioning is that the old text stops being retrievable
                    # the moment a new version supersedes it.
                    Document.is_current.is_(True),
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
        top_score = chunks[0].score if chunks else None
        if top_score is not None:
            # Only when something came back. A search that found nothing has no
            # "best score" to record, and folding it in as a zero would drag the
            # distribution toward a floor that no real match ever sat at — which
            # is precisely the reading this histogram exists to get right.
            # Recorded before the floor below, deliberately — see the metric's
            # own docstring in app/core/metrics.py.
            metrics.RETRIEVAL_TOP_SCORE.observe(top_score)

        floor = settings.retrieval_relevance_floor
        above_floor = _above_floor(chunks, floor)
        if floor is not None and chunks and not above_floor:
            # Not "found nothing" — found something and none of it was trusted.
            # Worth its own counter and its own log line: an operator watching
            # groundedness improve after setting a floor should also be able to
            # see the floor is where the improvement came from.
            metrics.RETRIEVAL_FLOOR_ABSTENTIONS.inc()
            logger.info(
                "retrieval_floor_abstained",
                collection_id=str(collection_id),
                floor=floor,
                top_score=top_score,
            )
        chunks = above_floor

        metrics.RETRIEVAL_CHUNKS.observe(len(chunks))
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
