"""Naive and heading-aware chunkers, written explicitly for learning."""

from pathlib import Path
import re

from findocs.types import Chunk


def naive_chunks(text: str, *, company: str, filing_date: str, size: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Make fixed character windows; this is intentionally the Day-0 baseline."""

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(Chunk(f"naive-{len(chunks)}", piece, company, "10-K", filing_date, "unknown"))
        start += size - overlap
    return chunks


ITEM_RE = re.compile(r"\b(Item\s+(?:1A?|1B|2|3|4|5|6|7A?|8|9A?|9B|10|11|12|13|14|15))\b", re.I)


def heading_aware_chunks(text: str, *, company: str, filing_date: str, size: int = 1800, overlap: int = 200) -> list[Chunk]:
    """Split at SEC Item boundaries, then sub-split oversized sections."""

    matches = list(ITEM_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if not matches:
        sections = [("unknown", text)]
    else:
        if matches[0].start() > 0:
            sections.append(("front matter", text[: matches[0].start()]))
        for number, match in enumerate(matches):
            end = matches[number + 1].start() if number + 1 < len(matches) else len(text)
            sections.append((match.group(1).upper(), text[match.start() : end]))
    chunks: list[Chunk] = []
    for section, body in sections:
        start = 0
        while start < len(body):
            piece = body[start : start + size]
            chunks.append(Chunk(f"section-{len(chunks)}", piece, company, "10-K", filing_date, section))
            start += size - overlap
    return chunks


def load_filing(path: str) -> tuple[str, str]:
    """Read a downloaded file and recover its filing date from the filename."""

    file = Path(path)
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", file.name)
    return file.read_text(encoding="utf-8"), date_match.group(1) if date_match else "unknown"

