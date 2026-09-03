"""Build a reusable chunk corpus from local filings or SEC downloads."""

from pathlib import Path

from findocs.ingest.chunking import heading_aware_chunks, load_filing
from findocs.ingest.sec_edgar import COMPANY_CIKS, latest_10k
from findocs.types import Chunk


def local_filing_path(company: str, data_dir: str = "data") -> Path:
    """Return the simple local text path used during fast offline development."""

    return Path(data_dir) / f"{company.lower()}_10k.txt"


def load_company_chunks(company: str, email: str | None = None, data_dir: str = "data") -> list[Chunk]:
    """Load one company's chunks, downloading the filing only when needed."""

    symbol = company.upper()
    path = local_filing_path(symbol, data_dir)
    if not path.exists():
        if email is None:
            raise FileNotFoundError(f"{path} is missing; pass --email so the filing can be downloaded from SEC EDGAR.")
        path = latest_10k(COMPANY_CIKS[symbol], email, output_dir=str(Path(data_dir) / "raw"))
    text, filing_date = load_filing(str(path))
    return heading_aware_chunks(text, company=symbol, filing_date=filing_date)


def load_corpus(companies: list[str], email: str | None = None, data_dir: str = "data") -> list[Chunk]:
    """Combine chunks from multiple companies into one retrieval corpus."""

    chunks: list[Chunk] = []
    for company in companies:
        chunks.extend(load_company_chunks(company, email=email, data_dir=data_dir))
    return chunks
