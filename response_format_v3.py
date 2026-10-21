"""Presentation guard for the portfolio chat UI.

The browser intentionally does not implement Markdown tables. Preserve the
information in generated tables by converting them to readable plain Markdown
headings and bullets, with no extra LLM/network calls.
"""
from __future__ import annotations

import re

_DIVIDER = re.compile(r"^:?-{3,}:?$")
_BR = re.compile(r"\\?<br\s*/?>", re.IGNORECASE)


def _cells(line: str) -> list[str] | None:
    clean = line.strip().replace(r"\|", "|")
    if not (clean.startswith("|") and clean.endswith("|")):
        return None
    cells = [item.strip() for item in clean[1:-1].split("|")]
    return cells if len(cells) >= 2 else None


def _separator(cells: list[str] | None) -> bool:
    return bool(cells and all(_DIVIDER.fullmatch(cell.replace(" ", "")) for cell in cells))


def _format_row(cells: list[str], headers: list[str]) -> list[str]:
    label = cells[0].strip()
    out = [label if label.startswith("**") else f"**{label}**"]
    for idx, value in enumerate(cells[1:], start=1):
        if not value:
            continue
        # Convert HTML-like breaks emitted by an LLM into separate bullets.
        fragments = [x.strip() for x in _BR.split(value) if x.strip()]
        heading = headers[idx] if idx < len(headers) else "Details"
        if len(cells) > 2:
            out.append(f"- **{heading}:** " + fragments[0].lstrip("•*- ").strip())
        elif fragments:
            out.append("- " + fragments[0].lstrip("•*- ").strip())
        for fragment in (fragments[1:] if len(cells) > 2 else fragments[1:]):
            out.append("- " + fragment.lstrip("•*- ").strip())
    return out


def no_markdown_tables(answer: str) -> str:
    """Convert GFM-style tables, leave other prose/code untouched.

    Only recognizes rows with a table separator; isolated pipes and malformed
    text remain unmodified instead of losing any user-visible facts.
    """
    lines = answer.splitlines()
    output: list[str] = []
    i = 0
    while i < len(lines):
        header = _cells(lines[i])
        next_cells = _cells(lines[i + 1]) if i + 1 < len(lines) else None
        if header is None or not _separator(next_cells) or len(header) != len(next_cells):
            output.append(lines[i])
            i += 1
            continue
        j = i + 2
        parsed = []
        while j < len(lines):
            cells = _cells(lines[j])
            if cells is None or len(cells) != len(header):
                break
            parsed.append(cells)
            j += 1
        # A header + divider without any actual rows is left unchanged.
        if not parsed:
            output.append(lines[i])
            i += 1
            continue
        if output and output[-1].strip():
            output.append("")
        for index, row in enumerate(parsed):
            if index:
                output.append("")
            output.extend(_format_row(row, header))
        output.append("")
        i = j
    return "\n".join(output).strip()
