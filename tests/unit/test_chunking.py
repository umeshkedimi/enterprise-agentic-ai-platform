"""Text extraction for the document types Chunk 11 added: HTML, CSV, XLSX.

The PDF/txt/markdown paths predate this file and are exercised indirectly by
the integration suite's real uploads; these are new enough, and involved
enough (an HTML parser, a spreadsheet reader), to be worth pinning directly
and offline.
"""

import io

import pytest
from openpyxl import Workbook

from app.services.chunking import XLSX_CONTENT_TYPE, extract_text


def test_html_strips_boilerplate_and_keeps_the_content():
    html = b"""
    <html>
      <head><style>body { color: red; }</style></head>
      <body>
        <nav>Home | About | Contact</nav>
        <header>Site Header</header>
        <main>
          <h1>Vacation Policy</h1>
          <p>Full-time employees accrue twenty-five days per year.</p>
        </main>
        <footer>Copyright 2026</footer>
        <script>trackPageView();</script>
      </body>
    </html>
    """
    text = extract_text(html, "text/html")

    assert "Vacation Policy" in text
    assert "twenty-five days" in text
    assert "Home | About | Contact" not in text
    assert "Site Header" not in text
    assert "Copyright 2026" not in text
    assert "trackPageView" not in text
    assert "color: red" not in text


def test_csv_renders_each_row_with_its_column_labels():
    csv_bytes = b"Employee,Days\nAlex,25\nJordan,20\n"
    text = extract_text(csv_bytes, "text/csv")

    lines = text.splitlines()
    assert lines[0] == "Employee: Alex, Days: 25"
    assert lines[1] == "Employee: Jordan, Days: 20"


def test_csv_strips_a_byte_order_mark_from_the_header():
    csv_bytes = "﻿Employee,Days\nAlex,25\n".encode()
    text = extract_text(csv_bytes, "text/csv")
    assert text.startswith("Employee: Alex")


def test_csv_tolerates_a_ragged_row():
    """A trailing missing cell is normal spreadsheet data, not corruption —
    the row still renders with whatever columns it actually has."""
    csv_bytes = b"Employee,Days,Notes\nAlex,25\n"
    text = extract_text(csv_bytes, "text/csv")
    assert text == "Employee: Alex, Days: 25"


def test_empty_csv_extracts_to_empty_text():
    assert extract_text(b"", "text/csv") == ""


def _xlsx_bytes(sheets: dict[str, list[list]]) -> bytes:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for name, rows in sheets.items():
        sheet = workbook.create_sheet(name)
        for row in rows:
            sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_xlsx_renders_rows_with_column_labels():
    content = _xlsx_bytes({"Leave": [["Employee", "Days"], ["Alex", 25], ["Jordan", 20]]})
    text = extract_text(content, XLSX_CONTENT_TYPE)

    assert "Sheet: Leave" in text
    assert "Employee: Alex, Days: 25" in text
    assert "Employee: Jordan, Days: 20" in text


def test_xlsx_names_each_sheet_so_facts_are_not_ambiguous_across_sheets():
    content = _xlsx_bytes(
        {
            "Leave": [["Employee", "Days"], ["Alex", 25]],
            "Expenses": [["Employee", "Amount"], ["Alex", 100]],
        }
    )
    text = extract_text(content, XLSX_CONTENT_TYPE)

    assert "Sheet: Leave" in text
    assert "Sheet: Expenses" in text
    assert text.index("Sheet: Leave") < text.index("Sheet: Expenses")


def test_xlsx_skips_a_sheet_with_no_rows():
    content = _xlsx_bytes({"Empty": [], "Leave": [["Employee", "Days"], ["Alex", 25]]})
    text = extract_text(content, XLSX_CONTENT_TYPE)
    assert "Sheet: Empty" not in text
    assert "Sheet: Leave" in text


def test_an_unrecognised_content_type_still_raises():
    with pytest.raises(ValueError, match="Unsupported content_type"):
        extract_text(b"whatever", "application/x-nonsense")
