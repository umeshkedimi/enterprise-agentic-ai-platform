import csv
import io

import tiktoken
from bs4 import BeautifulSoup
from openpyxl import load_workbook
from pypdf import PdfReader

DEFAULT_CHUNK_SIZE_TOKENS = 400
DEFAULT_CHUNK_OVERLAP_TOKENS = 50

# Spelled out as a constant, unlike the other content-type strings in this
# module, because it is long enough that hand-typing it in three places
# (here, `document_service.SUPPORTED_CONTENT_TYPES`, and the API's
# extension map) is a typo waiting to silently reject every .xlsx upload.
XLSX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Elements stripped before an HTML page's text is read. This is boilerplate
# removal, not a readability algorithm — it drops the tags that are reliably
# never article content (navigation, scripts, forms) rather than trying to
# infer a "main content" region the way a reader-mode extractor would. Good
# enough for an internal knowledge-base upload; a scraped marketing page with
# heavy chrome would want more.
_HTML_BOILERPLATE_TAGS = ("script", "style", "nav", "header", "footer", "aside", "noscript", "form")

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _extract_html(content: bytes) -> str:
    # "html.parser" rather than lxml: it ships with the standard library, so
    # HTML support costs one pure-Python dependency (beautifulsoup4) instead
    # of two, at a parsing-speed cost this platform's ingestion volume does
    # not need to care about.
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup.find_all(_HTML_BOILERPLATE_TAGS):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _render_table_rows(header: list[str], rows: list[list[str]]) -> str:
    """Render each row as "Column: value" pairs, one row per line.

    A table flattened into raw cells loses the thing that made it a table —
    which value belonged to which column — the moment it is read back as an
    unstructured wall of text. Repeating the header on every row keeps that
    association intact even if a chunk boundary later lands in the middle of
    the table, which it still can: guaranteeing a table survives chunking as
    one atomic unit is Chunk 12's job, not this one. This is the ingestion
    half of table support; the retrieval half comes later.

    `strict=False` here deliberately — a ragged row (a trailing blank cell
    Excel didn't bother writing) is normal spreadsheet data, not a malformed
    document the way a truncated PDF would be.
    """
    lines = []
    for row in rows:
        pairs = [
            f"{h.strip()}: {v.strip()}" for h, v in zip(header, row, strict=False) if h.strip()
        ]
        if pairs:
            lines.append(", ".join(pairs))
    return "\n".join(lines)


def _extract_csv(content: bytes) -> str:
    # utf-8-sig: Excel's own CSV export writes a byte-order-mark that plain
    # utf-8 decoding would leave attached to the first header cell.
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    rows = list(reader)
    if not rows:
        return ""
    header, *data_rows = rows
    return _render_table_rows(header, data_rows)


def _extract_xlsx(content: bytes) -> str:
    # read_only: streams rows instead of loading the whole workbook into
    # objects, which matters once a sheet has thousands of rows behind the
    # upload-size cap. data_only: a formula cell's cached last-computed value,
    # not the formula text — "=SUM(A2:A9)" is not a fact worth embedding.
    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    sections = []
    for sheet in workbook.worksheets:
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            continue
        header = ["" if cell is None else str(cell) for cell in rows[0]]
        data_rows = [["" if cell is None else str(cell) for cell in row] for row in rows[1:]]
        rendered = _render_table_rows(header, data_rows)
        if rendered:
            # Named per sheet so a multi-sheet workbook doesn't collapse into
            # one block of rows with no way to tell which sheet a fact came
            # from once it is just chunked text.
            sections.append(f"Sheet: {sheet.title}\n{rendered}")
    return "\n\n".join(sections)


def extract_text(content: bytes, content_type: str) -> str:
    """Extract plain text from an uploaded document's raw bytes."""
    if content_type == "application/pdf":
        reader = PdfReader(io.BytesIO(content))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if content_type in ("text/plain", "text/markdown"):
        return content.decode("utf-8")

    if content_type == "text/html":
        return _extract_html(content)

    if content_type == "text/csv":
        return _extract_csv(content)

    if content_type == XLSX_CONTENT_TYPE:
        return _extract_xlsx(content)

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
