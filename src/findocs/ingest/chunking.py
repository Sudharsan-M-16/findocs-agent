"""
chunking.py — Naive vs Heading-Aware Text Chunking
=====================================================
WHY CHUNKING MATTERS:
Embedding models have a maximum input length (typically 256-512 tokens).
A full 10-K filing is 50,000+ tokens. We must split it into smaller pieces
first. HOW we split it dramatically affects retrieval quality.

TWO STRATEGIES:
1. naive_chunks()          — Day-0 baseline: ignore document structure,
                             just slice fixed character windows.
2. heading_aware_chunks()  — Day-1 fix: slice at SEC Item boundaries first,
                             then apply the same window logic inside each section.

THE KEY INSIGHT:
Naive chunking can split a risk factor sentence in half and mix the second
half with unrelated text from the next section. When you then embed that
mixed chunk, the vector represents two different topics simultaneously,
which hurts both retrieval accuracy AND citation quality (you can't say
"this came from Item 1A" if the chunk contains both Item 1A and Item 2 text).
"""

from pathlib import Path
import re

from findocs.types import Chunk


# ── Naive chunker (intentional Day-0 baseline) ───────────────────────────────

def naive_chunks(
    text: str,
    *,
    company: str,
    filing_date: str,
    size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """
    Make fixed character windows; this is intentionally the Day-0 baseline.

    HOW IT WORKS:
    - Start at position 0.
    - Take `size` characters.
    - Advance by `size - overlap` characters (not `size`).
    - Repeat until the whole text is consumed.

    WHY OVERLAP?
    A sentence that falls exactly on a chunk boundary gets cut in half. The
    next chunk starts at the middle of that sentence, missing context.
    Overlapping by 150 characters means each chunk shares its last 150
    characters with the first 150 characters of the next chunk. This reduces
    (but doesn't eliminate) mid-sentence splits.

    WHY section="unknown"?
    Naive chunking doesn't know which SEC Item it's inside. This matters for
    citations: when the answer cites chunk "naive-42", the user can't tell
    whether it came from Item 1A (Risk Factors) or Item 7 (MD&A).

    INTERVIEW QUESTION: "What breaks with naive chunking?"
    Answer: 1) Sentences split at boundaries → incomplete evidence.
            2) Sections merged → a single chunk might embed both "risks" and
               "revenue", making the vector a noisy average of both topics.
            3) No section metadata → citations are useless.
    """

    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size]
        chunks.append(
            Chunk(
                f"naive-{len(chunks)}",  # unique ID, e.g. "naive-0", "naive-1"
                piece,
                company,
                "10-K",
                filing_date,
                "unknown",  # no section metadata — that's the whole problem
            )
        )
        start += size - overlap  # advance by less than `size` to create overlap
    return chunks


# ── SEC Item heading pattern ──────────────────────────────────────────────────

# 10-K filings always contain these numbered items (though their content varies).
# Item 1 = Business, Item 1A = Risk Factors, Item 7 = MD&A, Item 8 = Financials, etc.
# This regex matches "Item 1", "Item 1A", "Item 7A", "Item 15", etc., case-insensitively.
# The \b word boundaries prevent matching "item" inside a longer word.
ITEM_RE = re.compile(r"\b(Item\s+(?:1A?|1B|2|3|4|5|6|7A?|8|9A?|9B|10|11|12|13|14|15))\b", re.I)


# ── Heading-aware chunker ────────────────────────────────────────────────────

def heading_aware_chunks(
    text: str,
    *,
    company: str,
    filing_date: str,
    size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    """
    Split at SEC Item boundaries, then sub-split oversized sections.

    HOW IT WORKS — two passes:

    PASS 1 — Find section boundaries:
    Use ITEM_RE to find every "Item N" heading in the document. Record where
    each section starts and ends. The end of section N is the start of section
    N+1. Any text before the first Item heading is called "front matter"
    (usually the cover page and table of contents — low retrieval value).

    PASS 2 — Sub-split within each section:
    Even individual sections can be long (Item 1A Risk Factors is often 15,000+
    characters). Apply the same overlapping-window logic from naive_chunks(),
    but now each window gets the correct section metadata attached.

    RESULT:
    Every chunk now knows exactly which SEC Item it came from. This means:
    - Citations can say "Apple 2024 10-K, ITEM 1A" — that's a real citation.
    - Retrieval for "risk factors" will rank ITEM 1A chunks higher because
      the embedding captures a focused topic, not a mix.

    WHY size=1800 (larger than naive's 1200)?
    Inside a well-defined section, larger windows give more context without
    the topic-mixing problem. The embedding model can handle ~512 tokens
    (~2000 characters), so 1800 characters keeps us safe.

    INTERVIEW QUESTION: "Why not just use a sentence splitter?"
    Answer: SEC Items can be thousands of sentences. A sentence splitter
    produces too many tiny chunks and loses paragraph-level context. The
    two-pass approach preserves section coherence while keeping chunk size
    manageable for the embedding model.
    """

    # Pass 1: Collect all Item heading match objects
    matches = list(ITEM_RE.finditer(text))

    sections: list[tuple[str, str]] = []  # [(section_name, section_text), ...]

    if not matches:
        # No Item headings found (e.g. plain-text filing without standard structure)
        sections = [("unknown", text)]
    else:
        # Any text before the first Item heading → "front matter"
        if matches[0].start() > 0:
            sections.append(("front matter", text[: matches[0].start()]))

        # Each section: from this heading to the next heading (or end of file)
        for number, match in enumerate(matches):
            end = matches[number + 1].start() if number + 1 < len(matches) else len(text)
            # match.group(1) is e.g. "Item 1A"; .upper() gives "ITEM 1A"
            sections.append((match.group(1).upper(), text[match.start() : end]))

    # Pass 2: Sub-split each section with the overlapping window
    chunks: list[Chunk] = []
    for section, body in sections:
        start = 0
        while start < len(body):
            piece = body[start : start + size]
            chunks.append(
                Chunk(
                    f"section-{len(chunks)}",  # "section-0", "section-1", etc.
                    piece,
                    company,
                    "10-K",
                    filing_date,
                    section,  # "ITEM 1A", "ITEM 7", etc. — now meaningful!
                )
            )
            start += size - overlap
    return chunks


# ── Filing loader ─────────────────────────────────────────────────────────────

def load_filing(path: str) -> tuple[str, str]:
    """
    Read a downloaded file and recover its filing date from the filename.

    WHY RECOVER DATE FROM FILENAME?
    The raw text file contains the filing date deep inside the HTML, but
    extracting it requires XBRL parsing. The sec_edgar.py downloader
    encodes the date in the filename ("0000320193_2024-10-30_10-K.txt").
    Parsing the filename is simpler and more reliable.

    RETURNS: (full_text, filing_date_string)
    filing_date_string is e.g. "2024-10-30" or "unknown" if no date pattern.

    WHY RETURN A TUPLE?
    The caller (corpus.py → load_company_chunks) needs both the text to
    chunk and the date to attach to every Chunk's filing_date field.
    """

    file = Path(path)
    # Look for an ISO-format date like "2024-10-30" anywhere in the filename.
    date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", file.name)
    return (
        file.read_text(encoding="utf-8"),
        date_match.group(1) if date_match else "unknown",
    )
