"""
sec_edgar.py — SEC EDGAR Downloader
=====================================
Responsible for fetching a company's latest 10-K filing from the US
Securities and Exchange Commission's public EDGAR data API.

KEY CONCEPTS:
-------------
CIK (Central Index Key):
    Every company that files with the SEC gets a unique numeric ID called the
    CIK. For example, Apple's CIK is 320193. We zero-pad it to 10 digits
    because the SEC API requires exactly that format.

Accession Number:
    Every filing has a unique accession number like "0000320193-24-000081".
    Strip the dashes to get the folder path in the EDGAR archive.

User-Agent header (REQUIRED by SEC policy):
    The SEC's EDGAR API requires you to identify yourself in the HTTP header.
    If you omit this, your requests will be rate-limited or blocked.
    Format: "CompanyName contact@email.com" — using your email is intentional
    and is SEC policy, not optional.

WHY NOT USE BROWSER/SCRAPING?
    EDGAR provides a structured JSON API at data.sec.gov. We use it directly
    because it's stable, free, and doesn't require parsing HTML tables.
"""

from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


# ── Company CIK map ──────────────────────────────────────────────────────────
# You can find any company's CIK by searching https://www.sec.gov/cgi-bin/browse-edgar
# The six companies here cover the multi-company evaluation track.
COMPANY_CIKS = {
    "AAPL": "320193",
    "MSFT": "789019",
    "AMZN": "1018724",
    "GOOG": "1652044",
    "NVDA": "1045810",
    "TSLA": "1318605",
}


def _normalise_cik(cik: str) -> str:
    """
    SEC endpoints require a ten-digit, zero-padded CIK.

    Example: "320193" → "0000320193"

    WHY zero-padding?
    The EDGAR JSON API (data.sec.gov/submissions/CIK{cik10}.json) treats the
    CIK as a string identifier, not an integer. The padding is required for
    the URL to resolve — without it you get a 404.
    """

    return str(cik).strip().replace("-", "").zfill(10)


def latest_10k(cik: str, email: str, output_dir: str = "data/raw") -> Path:
    """
    Download and clean the latest 10-K HTML, returning its local text path.

    HOW IT WORKS — step by step:
    1. Validate the email (SEC policy requires a real contact address).
    2. Zero-pad the CIK to 10 digits.
    3. GET submissions JSON → a list of every filing this company has made.
    4. Find the first form == "10-K" (most recent, since EDGAR is newest-first).
    5. Build the archive URL from accession number + primary document filename.
    6. GET the filing HTML.
    7. Parse with BeautifulSoup — remove <script>, <style>, <ix:header> tags
       which are XBRL-specific and pure noise for text retrieval.
    8. Collapse all whitespace to single spaces (filings are full of \xa0, \n,
       and repeated spaces from HTML formatting).
    9. Save as a local .txt file named "{cik}_{date}_10-K.txt".

    WHY BEAUTIFULSOUP AND NOT requests.text DIRECTLY?
    10-K filings are HTML, not plain text. If you read the raw HTML you get
    thousands of tags like <div class="ix:nonFraction"> around every number.
    BeautifulSoup's get_text() extracts only the human-readable content.

    WHY SAVE TO DISK?
    Embedding 180+ chunks takes ~30 seconds on the first run. Saving locally
    means you never need to re-download the same filing.
    """

    # SEC requires a real email in the User-Agent. Without this your IP gets
    # blocked after a few requests.
    if "@" not in email:
        raise ValueError("Pass a real email address in the SEC User-Agent.")

    cik10 = _normalise_cik(cik)
    headers = {"User-Agent": f"FinDocs learning project {email}"}

    # Step 1: Fetch the submissions index for this company.
    # Returns JSON with 'filings' → 'recent' → parallel arrays of form types,
    # accession numbers, primary documents, and filing dates.
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    response = requests.get(submissions_url, headers=headers, timeout=30)
    response.raise_for_status()  # Raise HTTPError for 4xx/5xx responses
    recent = response.json()["filings"]["recent"]

    # Step 2: Find the index of the most recent 10-K.
    # "form" is a list like ["10-K", "10-Q", "8-K", ...]. We take the first
    # match because EDGAR lists filings newest-first.
    matches = [i for i, form in enumerate(recent["form"]) if form == "10-K"]
    if not matches:
        raise RuntimeError(f"No recent 10-K found for CIK {cik10}.")
    row = matches[0]

    # Step 3: Build the direct URL to the filing document.
    # accessionNumber looks like "0000320193-24-000081"; removing dashes gives
    # the archive folder name. primaryDocument is the main HTML file name.
    accession = recent["accessionNumber"][row].replace("-", "")
    document = recent["primaryDocument"][row]
    filing_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession}/{document}"
    )

    # Step 4: Download the filing HTML.
    filing = requests.get(filing_url, headers=headers, timeout=60)
    filing.raise_for_status()

    # Step 5: Parse and clean the HTML.
    soup = BeautifulSoup(filing.content, "html.parser")
    # Remove tags that contain no human-readable text. ix:header is the XBRL
    # inline metadata wrapper that appears in modern EDGAR filings.
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    # Collapse whitespace: filings use non-breaking spaces, multiple newlines,
    # and large runs of spaces as visual formatting. One space is enough.
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()

    # Step 6: Write to disk with a descriptive filename for easy identification.
    target = Path(output_dir) / f"{cik10}_{recent['filingDate'][row]}_10-K.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def latest_10ks(companies: list[str], email: str, output_dir: str = "data/raw") -> list[Path]:
    """
    Download several companies while reusing the single-company implementation.

    WHY THIS WRAPPER?
    Calling latest_10k in a loop would work, but this function validates all
    company symbols before starting network requests, giving a cleaner error
    if a symbol is missing from COMPANY_CIKS.
    """

    paths = []
    for company in companies:
        symbol = company.upper()
        if symbol not in COMPANY_CIKS:
            raise KeyError(f"Unknown company {symbol}; add its SEC CIK explicitly first.")
        paths.append(latest_10k(COMPANY_CIKS[symbol], email, output_dir))
    return paths
