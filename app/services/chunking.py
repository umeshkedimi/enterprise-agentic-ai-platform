import io

import tiktoken
from pypdf import PdfReader

DEFAULT_CHUNK_SIZE_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 50

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def extract_text(content: bytes, content_type: str) -> str:
    """Extract plain text from an uploaded document's raw bytes."""
    if content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if content_type in ("text/plain", "text/markdown"):
        return content.decode("utf-8")

    raise ValueError(f"Unsupported content_type: {content_type}")


def chunk_text(
    text: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping, token-bounded chunks.

    Token-based (not character-based) so chunk sizes map directly to the
    embedding model's context limit, regardless of language/markup density.
    """
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    tokens = _encoding.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunk = _encoding.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(tokens):
            break
        start += step

    return chunks
