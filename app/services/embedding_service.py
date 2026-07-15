from openai import APIError, APITimeoutError, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.llm import get_embedding_model, get_llm_client

# OpenAI's embeddings endpoint accepts up to 2048 inputs per request; we batch
# well under that so a single slow/failed batch doesn't waste a huge amount
# of already-computed work on retry.
_BATCH_SIZE = 100

_RETRYABLE_ERRORS = (APIError, APITimeoutError, RateLimitError)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
async def _embed_batch(texts: list[str]) -> list[list[float]]:
    client = get_llm_client()
    response = await client.embeddings.create(model=get_embedding_model(), input=texts)
    return [item.embedding for item in response.data]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching and retrying transient API failures."""
    if not texts:
        return []

    embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        embeddings.extend(await _embed_batch(batch))
    return embeddings


async def embed_text(text: str) -> list[float]:
    (embedding,) = await embed_texts([text])
    return embedding
