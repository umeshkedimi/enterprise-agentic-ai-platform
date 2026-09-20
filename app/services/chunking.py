import csv
import io
import re

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

# Block-level tags read out as separate paragraphs, blank-line joined. This is
# what makes "structure-aware chunking" mean anything for HTML: the chunker
# downstream only knows a paragraph boundary exists where it sees a blank
# line, so a page read out as one undifferentiated stream of text would give
# it nothing to respect. Text nodes outside all of these (rare — most pages
# wrap everything in some block tag) fall back to a flatter read, below.
_HTML_BLOCK_TAGS = ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote", "pre")

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

    blocks = []
    for element in soup.find_all(_HTML_BLOCK_TAGS):
        # Collapse each block's own internal whitespace/line-wrapping to one
        # line — the blank line is what marks a boundary *between* blocks;
        # a block's own text staying on one line is what stops it from being
        # mistaken for several.
        text = " ".join(element.get_text().split())
        if text:
            blocks.append(text)
    if blocks:
        return "\n\n".join(blocks)

    # A page with no recognised block tags at all (rare) falls back to a flat
    # read rather than producing nothing.
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


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def _token_slice(text: str, max_tokens: int) -> list[str]:
    """Cut one oversized unit by raw token position — the whole of v1's
    algorithm, now the last-resort fallback rather than the only strategy.

    Reached only by a single paragraph or line that alone exceeds the chunk
    budget: an unusually long sentence, or a table row wide enough that even
    one row can't fit. Genuinely nothing finer to respect at that point.
    """
    tokens = _encoding.encode(text)
    slices = []
    for i in range(0, len(tokens), max_tokens):
        piece = _encoding.decode(tokens[i : i + max_tokens]).strip()
        if piece:
            slices.append(piece)
    return slices


def _split_into_units(text: str, max_tokens: int) -> list[str]:
    """Break text into pieces a chunk boundary may fall between, never within.

    Two-tier, in order of preference: a blank-line-separated paragraph stays
    one unit if it fits; if it doesn't — most often a table, where
    `_render_table_rows` joins every row with a single newline and no blank
    lines between them — it splits at line boundaries instead, which is
    exactly what keeps a large table's individual rows intact rather than
    landing a cut in the middle of one. Only a single line that alone still
    exceeds the budget falls through to raw token slicing.

    This is why HTML extraction was changed to emit a blank line between
    block-level elements: a paragraph boundary that was never preserved at
    extraction time has nothing here to respect. PDF text has no reliable
    paragraph markers of its own — pypdf gives back one line per visual line
    of the original page with no semantic markup — so a PDF page is one
    large paragraph in practice and falls to line-level splitting for most
    documents; still an improvement over v1's raw token slicing, since a
    visual line is at least never cut in half, but honestly short of true
    paragraph awareness. Fixing that needs PDF layout analysis this platform
    doesn't have.
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if count_tokens(paragraph) <= max_tokens:
            units.append(paragraph)
            continue
        for line in paragraph.splitlines():
            line = line.strip()
            if not line:
                continue
            if count_tokens(line) <= max_tokens:
                units.append(line)
            else:
                units.extend(_token_slice(line, max_tokens))
    return units


def _overlap_tail(units: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """The trailing units of a filled chunk that carry into the next one.

    Whole units only, up to the overlap budget — never a fragment of one, so
    overlap can be slightly under or over `overlap_tokens` but never splits a
    table row (or a sentence) to hit the number exactly.
    """
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit)
        if tail and total + unit_tokens > overlap_tokens:
            break
        tail.insert(0, unit)
        total += unit_tokens
    return tail, total


def chunk_text(
    text: str,
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping, token-bounded chunks along structural
    boundaries — paragraphs, or table rows — rather than raw token position.

    Still token-bounded (chunk sizes map directly to the embedding model's
    context limit regardless of language/markup density), and every unit this
    packs is individually within budget by construction — `_split_into_units`
    guarantees that before a single chunk is assembled here. This is the
    packing pass: units are added to the current chunk until the next one
    would overflow it, then a new chunk starts, seeded with however many
    trailing units from the last one fit the overlap budget.
    """
    if chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    units = _split_into_units(text, chunk_size_tokens)
    if not units:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if current and current_tokens + unit_tokens > chunk_size_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = _overlap_tail(current, chunk_overlap_tokens)
            # The overlap tail plus this unit can still overflow the budget —
            # `_overlap_tail` only bounds itself against `chunk_overlap_tokens`,
            # not against whatever comes next. Drop the overlap rather than
            # let the chunk grow past `chunk_size_tokens`: a single unit is
            # always within budget by construction, so starting fresh always
            # fits, and the token bound is the guarantee that matters more
            # than the convenience of overlap.
            if current_tokens + unit_tokens > chunk_size_tokens:
                current, current_tokens = [], 0
        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append("\n\n".join(current))

    return chunks
