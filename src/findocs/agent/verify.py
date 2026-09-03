"""Conservative claim-to-evidence verification for extractive answers."""

import re
from findocs.types import RetrievedChunk


def split_claims(answer: str) -> list[str]:
    """Split a generated answer into checkable sentence-level claims."""

    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part.strip()]


def verify_claims(answer: str, evidence: list[RetrievedChunk]) -> dict:
    """Mark a claim supported when its meaningful words overlap a source sentence."""

    source_text = " ".join(item.chunk.text.lower() for item in evidence)
    verdicts = []
    for claim in split_claims(answer):
        words = set(re.findall(r"[a-z]{4,}|\$?[0-9][0-9,.%]*", claim.lower()))
        hits = sum(word in source_text for word in words)
        supported = bool(words) and hits / len(words) >= 0.5
        sources = []
        if supported:
            for item in evidence:
                sources.append({
                    "chunk_id": item.chunk.chunk_id,
                    "citation": f"{item.chunk.company} {item.chunk.filing_date} {item.chunk.filing_type}, {item.chunk.section}",
                })
        verdicts.append({"claim": claim, "supported": supported, "sources": sources})
    return {"all_supported": all(item["supported"] for item in verdicts), "claims": verdicts}
