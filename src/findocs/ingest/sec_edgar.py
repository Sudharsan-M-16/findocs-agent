"""Minimal SEC EDGAR client for downloading a company's latest 10-K."""

from pathlib import Path
import re

import requests
from bs4 import BeautifulSoup


COMPANY_CIKS = {
    "AAPL": "320193",
    "MSFT": "789019",
    "AMZN": "1018724",
    "GOOG": "1652044",
    "NVDA": "1045810",
    "TSLA": "1318605",
}


def _normalise_cik(cik: str) -> str:
    """SEC endpoints require a ten-digit, zero-padded CIK."""

    return str(cik).strip().replace("-", "").zfill(10)


def latest_10k(cik: str, email: str, output_dir: str = "data/raw") -> Path:
    """Download and clean the latest 10-K HTML, returning its local text path."""

    if "@" not in email:
        raise ValueError("Pass a real email address in the SEC User-Agent.")
    cik10 = _normalise_cik(cik)
    headers = {"User-Agent": f"FinDocs learning project {email}"}
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    response = requests.get(submissions_url, headers=headers, timeout=30)
    response.raise_for_status()
    recent = response.json()["filings"]["recent"]
    matches = [i for i, form in enumerate(recent["form"]) if form == "10-K"]
    if not matches:
        raise RuntimeError(f"No recent 10-K found for CIK {cik10}.")
    row = matches[0]
    accession = recent["accessionNumber"][row].replace("-", "")
    document = recent["primaryDocument"][row]
    filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/{accession}/{document}"
    filing = requests.get(filing_url, headers=headers, timeout=60)
    filing.raise_for_status()
    soup = BeautifulSoup(filing.content, "html.parser")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    target = Path(output_dir) / f"{cik10}_{recent['filingDate'][row]}_10-K.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def latest_10ks(companies: list[str], email: str, output_dir: str = "data/raw") -> list[Path]:
    """Download several companies while reusing the single-company implementation."""

    paths = []
    for company in companies:
        symbol = company.upper()
        if symbol not in COMPANY_CIKS:
            raise KeyError(f"Unknown company {symbol}; add its SEC CIK explicitly first.")
        paths.append(latest_10k(COMPANY_CIKS[symbol], email, output_dir))
    return paths
