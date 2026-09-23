import time
import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import metrics, tracing
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.agent import Agent
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.models.schemas import Citation
from app.services.completion_service import Turn, complete
from app.services.embedding_service import embed_text
from app.services.reranking_prompts import (
    RERANK_INSTRUCTIONS,
    build_rerank_request,
    parse_rerank_response,
)

logger = get_logger(__name__)

DEFAULT_TOP_K = 5
SNIPPET_LENGTH = 240

# Namespaced and fixed, same reasoning as the evaluation judge's own id: this
# agent belongs to the platform, is assembled fresh per call, and is never
# stored — there is no row for it, only something for logs and traces to name
# consistently across calls.
_RERANKER_AGENT_ID = uuid.uuid5(uuid.NAMESPACE_URL, "eaap:reranker")
# Deterministic ordering, not a creative task — a reranker that returns a
# different order each time it is asked is not measuring relevance, it is
# measuring itself. Same reasoning as the judge's own temperature.
_RERANKER_TEMPERATURE = 0.0

# How many candidates each ranker contributes to fusion, before the fused
# list is trimmed back to top_k. Wider than top_k on purpose: a chunk full-
# text ranks #3 but vector search ranks #40 still needs to be *in* the vector
# ranker's own list for fusion to see it at all, and the reverse. Not a
# Setting — this is sizing headroom for the fusion step, not a tunable
# retrieval-quality knob the way `hybrid_search_rrf_k` is.
_CANDIDATE_POOL_MULTIPLIER = 4
_MIN_CANDIDATE_POOL = 20


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


def _reciprocal_rank_fusion(rankings: list[list[uuid.UUID]], *, k: int) -> list[uuid.UUID]:
    """Merge several rank-ordered id lists into one, by Reciprocal Rank Fusion.

    Each list's contribution to a shared id is `1 / (k + rank)`, 1-indexed — a
    score derived purely from *position*, never from the two rankers' own
    incomparable underlying numbers. Cosine similarity is bounded and roughly
    linear; `ts_rank_cd` is neither, and normalising it well enough to combine
    directly with a cosine score is its own unsolved problem RRF sidesteps
    entirely by never looking at either score, only at rank.

    An id appearing in only one ranking still gets a score, from that ranking
    alone, rather than needing to appear in both to be considered — the whole
    point is to rescue a strong single-signal match (an exact code a keyword
    search nails and embeddings under-weight), not to require two different
    kinds of evidence to agree before either counts.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def _row_to_chunk(row) -> RetrievedChunk:
    chunk, filename, distance = row
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=filename,
        content=chunk.content,
        # pgvector's cosine_distance is 1 - cosine_similarity. Computed here
        # regardless of which ranker this row came from, so `score` always
        # means the same thing — a chunk full-text search rescued is scored
        # by its own real vector confidence, not by an incomparable full-text
        # rank turned into a fake similarity number. That is what keeps the
        # relevance floor and the calibration report meaningful unchanged:
        # neither has to know hybrid search exists.
        score=1.0 - distance,
    )


def _reranker_agent(*, tenant_id: uuid.UUID, settings: Settings) -> Agent:
    """Assembled fresh per call, never stored — same shape as the evaluation
    judge's synthetic agent, for the same reason: `complete()` takes an
    `Agent`, and routing through it rather than around it is what keeps
    retries, credential resolution, and token accounting from having a
    second, uncounted path to a provider."""
    return Agent(
        id=_RERANKER_AGENT_ID,
        # The real tenant, so a reranker's token spend is attributable to
        # whoever it was spent reordering results for, not a platform bucket.
        tenant_id=tenant_id,
        slug="platform-reranker",
        name="Platform reranker",
        system_prompt=RERANK_INSTRUCTIONS,
        model=settings.reranking_model,
        collection_id=None,
        tool_allowlist=[],
        temperature=_RERANKER_TEMPERATURE,
        max_output_tokens=settings.reranking_max_output_tokens,
        retrieval_top_k=0,
        enabled=True,
    )


async def _rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    top_k: int,
    tenant_id: uuid.UUID,
    settings: Settings,
) -> list[RetrievedChunk]:
    """Reorder fused candidates by a model's judgement of relevance, and cut
    to top_k — or fall back to the pre-rerank order unchanged.

    A misbehaving or unreachable reranker costs ranking quality for this one
    turn, never the turn itself: retrieval already found real candidates
    before this function was ever called, so the safe default on any
    failure is exactly what would have been returned without reranking at
    all. Same posture `conversation_service._advance_summary` already takes
    for its own request-path model call — log and degrade, don't raise.
    """
    numbered = list(enumerate(candidates, start=1))
    try:
        completion = await complete(
            agent=_reranker_agent(tenant_id=tenant_id, settings=settings),
            turns=[
                Turn(
                    role="user",
                    content=build_rerank_request(
                        query=query, passages=[(i, c.content) for i, c in numbered]
                    ),
                )
            ],
            workload=metrics.WORKLOAD_RERANKING,
            settings=settings,
        )
        order = parse_rerank_response(
            completion.text, expected_ids={i for i, _ in numbered}
        )
    except Exception as exc:  # noqa: BLE001 - provider/parse errors, all non-fatal here
        logger.warning(
            "reranking_failed",
            tenant_id=str(tenant_id),
            candidates=len(candidates),
            error=type(exc).__name__,
        )
        return candidates[:top_k]

    by_number = dict(numbered)
    return [by_number[i] for i in order][:top_k]


async def semantic_search(
    session: AsyncSession,
    query: str,
    *,
    collection_id: uuid.UUID,
    top_k: int = DEFAULT_TOP_K,
    settings: Settings | None = None,
    tenant_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    """Return the top_k chunks best matching `query`, restricted to fully-
    processed, current documents in the given collection.

    Two rankers, fused: cosine similarity over the embedding (always), and
    Postgres full-text search over a generated `tsvector` column (when
    `hybrid_search_enabled`). Vector search alone under-ranks exact-match
    queries — a policy code, an account number — against a query's more
    generic surrounding words; full text search alone misses the paraphrases
    and synonyms vector search exists for. Reciprocal rank fusion combines
    the two without ever comparing their incomparable raw scores.

    A wider shortlist of the fused result is then optionally reranked by a
    model (`reranking_enabled`, off by default — see the setting's own
    docstring for why) before being cut to top_k. `tenant_id` is only used
    for that step, to attribute a reranker's token spend to whoever it was
    spent for; every other caller may safely omit it.

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
    pool_size = max(top_k * _CANDIDATE_POOL_MULTIPLIER, _MIN_CANDIDATE_POOL)

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
            base_conditions = (
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

            vector_stmt = (
                select(DocumentChunk, Document.filename, distance.label("distance"))
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(*base_conditions)
                .order_by(distance)
                .limit(pool_size)
            )
            vector_rows = (await session.execute(vector_stmt)).all()

            fts_rows: list = []
            if settings.hybrid_search_enabled:
                tsquery = func.websearch_to_tsquery("english", query)
                fts_stmt = (
                    select(DocumentChunk, Document.filename, distance.label("distance"))
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(*base_conditions, DocumentChunk.content_tsv.op("@@")(tsquery))
                    .order_by(func.ts_rank_cd(DocumentChunk.content_tsv, tsquery).desc())
                    .limit(pool_size)
                )
                fts_rows = (await session.execute(fts_stmt)).all()
        except Exception:
            metrics.RETRIEVAL_REQUESTS.labels("error").inc()
            raise

        by_id = {row[0].id: _row_to_chunk(row) for row in (*vector_rows, *fts_rows)}
        fused_ids = _reciprocal_rank_fusion(
            [[row[0].id for row in vector_rows], [row[0].id for row in fts_rows]],
            k=settings.hybrid_search_rrf_k,
        )
        fused = [by_id[cid] for cid in fused_ids]

        reranked = False
        if settings.reranking_enabled and len(fused) > 1:
            if tenant_id is None:
                # A caller enabled reranking platform-wide but didn't wire
                # tenant_id through — a wiring gap, not a runtime failure, so
                # it costs this turn's ranking quality rather than raising.
                logger.warning("reranking_skipped_no_tenant_id", collection_id=str(collection_id))
                chunks = fused[:top_k]
            else:
                shortlist = fused[: settings.reranking_candidate_pool]
                chunks = await _rerank(
                    query, shortlist, top_k=top_k, tenant_id=tenant_id, settings=settings
                )
                reranked = True
        else:
            chunks = fused[:top_k]

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
                "eaap.retrieval.fts_candidates": len(fts_rows),
                "eaap.retrieval.reranked": reranked,
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
