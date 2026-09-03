"""
corpus.py — Shared Filing Loader
===================================
PURPOSE: Provide one canonical function for "give me chunks for company X"
that every CLI command and eval harness uses.

WHY A SHARED LOADER MATTERS:
Without corpus.py, each command would independently discover how to:
1. Find the local file for a company (what is AAPL's filename?).
2. Handle the case when the file doesn't exist (download from SEC EDGAR).
3. Call the chunker with the right parameters.
4. Combine chunks from multiple companies.

Duplicating this logic in cli.py, eval/pipeline.py, and eval/correction.py
would mean bugs get fixed in one place but not others. corpus.py is the
single source of truth for "how do we get chunks?"

LOCAL FILE STEM MAP:
Companies may have been downloaded with different filename conventions.
The mapping supports multiple known filenames per company so you don't
have to rename files when switching between downloading methods.
"""

from pathlib import Path

from findocs.ingest.chunking import heading_aware_chunks, load_filing
from findocs.ingest.sec_edgar import COMPANY_CIKS, latest_10k
from findocs.types import Chunk


# Map from ticker symbol to list of possible local filenames (in priority order).
# The first filename that exists on disk is used.
# WHY A LIST? Different downloads or manual saves might use different names.
LOCAL_FILE_STEMS = {
    "AAPL": ["aapl_10k.txt", "apple_10k.txt"],
    "MSFT": ["msft_10k.txt", "microsoft_10k.txt"],
    "AMZN": ["amzn_10k.txt", "amazon_10k.txt"],
    "GOOG": ["goog_10k.txt", "google_10k.txt", "alphabet_10k.txt"],
    "NVDA": ["nvda_10k.txt", "nvidia_10k.txt"],
    "TSLA": ["tsla_10k.txt", "tesla_10k.txt"],
}


def local_filing_path(company: str, data_dir: str = "data") -> Path:
    """
    Return the first matching local text path for this company.

    SEARCH ORDER:
    1. Try each filename in LOCAL_FILE_STEMS[company] in order.
    2. If none exist: return the default path (even if it doesn't exist).
       The caller (load_company_chunks) checks existence and handles downloads.

    WHY company.upper()?
    CLI input might be "aapl", "AAPL", "Aapl" — normalise to uppercase so the
    lookup always works regardless of how the user typed the company name.

    EXAMPLE:
    local_filing_path("AAPL") tries:
      data/aapl_10k.txt  → if it exists, return it.
      data/apple_10k.txt → if that exists, return it.
    If neither exists → return data/aapl_10k.txt (default fallback for error message).
    """

    symbol = company.upper()
    root = Path(data_dir)

    for filename in LOCAL_FILE_STEMS.get(symbol, [f"{symbol.lower()}_10k.txt"]):
        path = root / filename
        if path.exists():
            return path  # First match wins

    # Return the default path so the error message in load_company_chunks is informative
    return root / f"{symbol.lower()}_10k.txt"


def load_company_chunks(
    company: str,
    email: str | None = None,
    data_dir: str = "data",
) -> list[Chunk]:
    """
    Load one company's chunks, downloading the filing only when needed.

    DECISION TREE:
    1. Check if a local file exists (via local_filing_path).
    2. If local file exists: read it, chunk it, return chunks.
    3. If local file DOESN'T exist AND email is provided:
       Download from SEC EDGAR, save locally, chunk it, return chunks.
    4. If local file DOESN'T exist AND no email:
       Raise FileNotFoundError with a helpful message pointing to --email.

    WHY REQUIRE EMAIL FOR DOWNLOAD?
    SEC EDGAR's API requires a User-Agent header identifying you by email.
    Without it, requests are rate-limited or blocked. The email is used
    ONLY for the User-Agent header — it's not stored or transmitted elsewhere.

    WHY SAVE TO data/raw/?
    Downloads go to data/raw/ rather than data/ to keep the "primary" data
    directory clean. Local manually-placed files (apple_10k.txt) live in
    data/. Downloaded files get dated names like "0000320193_2024-10-30_10-K.txt".

    CHUNKING PARAMETERS:
    heading_aware_chunks uses the defaults (size=1800, overlap=200).
    These were chosen after observing that 1200-char naive chunks frequently
    cut sentences mid-way. 1800 gives each chunk enough context while staying
    within the embedding model's 512-token limit (~2000 characters).
    """

    symbol = company.upper()
    path = local_filing_path(symbol, data_dir)

    if not path.exists():
        if email is None:
            raise FileNotFoundError(
                f"{path} is missing; pass --email so the filing can be "
                "downloaded from SEC EDGAR."
            )
        # Download from SEC EDGAR and save locally
        path = latest_10k(
            COMPANY_CIKS[symbol],
            email,
            output_dir=str(Path(data_dir) / "raw"),
        )

    # Read the file and extract the filing date from the filename
    text, filing_date = load_filing(str(path))
    # Split into heading-aware chunks with company and date metadata attached
    return heading_aware_chunks(text, company=symbol, filing_date=filing_date)


def load_corpus(
    companies: list[str],
    email: str | None = None,
    data_dir: str = "data",
) -> list[Chunk]:
    """
    Combine chunks from multiple companies into one retrieval corpus.

    FOR MULTI-COMPANY EVALUATION:
    When eval_questions.json contains a multi-hop question (q010: NVDA vs MSFT),
    you need both companies' chunks in the same list so the retriever can find
    relevant chunks from both filings.

    ORDER MATTERS FOR chunk_id UNIQUENESS:
    All chunks use "section-N" IDs where N is the list index. If you load AAPL
    first (181 chunks), then MSFT, MSFT chunks get IDs starting at section-181.
    This prevents chunk_id collisions across companies.

    USAGE:
        chunks = load_corpus(["AAPL", "MSFT"], email="your@email.com")
        retriever = DenseRetriever(chunks)  # Encodes all 360+ chunks
    """

    chunks: list[Chunk] = []
    for company in companies:
        # Extend (not append) so all chunks go into one flat list
        chunks.extend(load_company_chunks(company, email=email, data_dir=data_dir))
    return chunks
