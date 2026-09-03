r"""
verify.py — Claim-Level Citation Verification
===============================================
CORE PROBLEM: Good retrieval does not guarantee a faithful answer.
An LLM (or even a simple generator) can produce answers that:
- Overstate what the filing actually says ("revenue grew 30%" when the filing says 23%).
- Mix up figures from different sections (Q1 vs full-year).
- Invent numbers that sound plausible but aren't in the retrieved chunks.

This is called HALLUCINATION. verify.py is the anti-hallucination layer.

HOW VERIFICATION WORKS:
1. Split the answer into individual claims (sentences).
2. For each claim: extract its "meaningful" words.
3. Check whether at least 50% of those meaningful words appear anywhere
   in the combined text of all retrieved chunks.
4. If yes: mark the claim "supported" and attach citations.
5. If no: mark it "unsupported" — this is a potential hallucination.

WHY LEXICAL OVERLAP AND NOT LLM ENTAILMENT?
Two approaches for claim checking:
  A) LLM: "Does this context entail this claim?" — high accuracy, costs money,
     adds latency, introduces another model dependency.
  B) Lexical overlap (this implementation): fast, free, interpretable, but
     misses paraphrase (the claim "sales fell" won't match "revenue declined").

This is the intentionally conservative baseline. The code comments explicitly
document this so you can say in interviews: "The current verifier uses lexical
overlap as a transparent baseline. A production version would use an entailment
model or NLI head — I documented this trade-off explicitly."

WHAT "MEANINGFUL WORDS" MEANS:
We ignore short common words ("the", "in", "of") by requiring 4+ character
words. We DO include numbers and financial figures (the pattern \$?[0-9][0-9,.%]*
captures "$85.8B", "23%", "2024").

INTERVIEW QUESTION: "What's the failure mode of this verifier?"
- False positive: "The company manufactures products" → always supported because
  "company", "manufactures", "products" appear in any filing chunk. The threshold
  is too lenient for generic claims.
- False negative: "Sales fell" → "revenue declined" → zero overlap → unsupported,
  even though the claim is perfectly grounded. The threshold is too strict for
  paraphrase.
This is exactly why you'd want an NLI model or a fine-tuned verifier in production.
"""

import re
from findocs.types import RetrievedChunk


def split_claims(answer: str) -> list[str]:
    r"""
    Split a generated answer into checkable sentence-level claims.

    WHY SENTENCE LEVEL?
    A sentence is the smallest natural unit that makes a standalone claim.
    Splitting at sub-sentence level would break the grammatical structure.
    Splitting at paragraph level would group multiple independent claims,
    making it harder to pinpoint which specific claim is unsupported.

    HOW THE REGEX WORKS:
    (?<=[.!?]) — positive lookbehind: match the position AFTER a period,
                 exclamation mark, or question mark.
    \s+        — one or more whitespace characters (the space after the period).

    Lookbehind doesn't consume the matched text, so the period stays with
    the sentence that ends with it, not the next sentence.

    EDGE CASE: "Apple's Q3 revenue was $85.8B. R&D was $7.8B."
    This correctly splits into two claims at the period-space boundary.
    "Dr. Smith" would wrongly split at "Dr." — known limitation for formal text,
    acceptable for financial statement sentences.
    """

    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", answer.strip())
        if part.strip()
    ]


def verify_claims(answer: str, evidence: list[RetrievedChunk]) -> dict:
    r"""
    Mark a claim supported when its meaningful words overlap a source sentence.

    STEP BY STEP:
    1. Concatenate all retrieved chunk texts into one lowercase string.
       We check against all evidence at once (not per-chunk) — a claim can be
       supported if ANY chunk contains the relevant words.

    2. For each claim:
       a. Extract meaningful words using the pattern:
          [a-z]{4,}      : lowercase words of 4+ characters (filters stopwords)
          \$?[0-9][0-9,.%]* : financial figures like "$85.8B", "23%", "2.1"
       b. Count how many of those words appear in the combined source text.
       c. supported = (at least one meaningful word) AND (≥50% hit rate).

    3. If supported: attach citation info from ALL evidence chunks.
       WHY ALL CHUNKS? The claim might draw from evidence across chunks.
       The citations list gives the reader chunks to verify against.

    4. Return a dict with:
       - "all_supported": True if EVERY claim is supported.
       - "claims": list of per-claim dicts with claim text, supported flag, sources.

    RETURNS STRUCTURE:
    {
        "all_supported": True/False,
        "claims": [
            {
                "claim": "Apple's revenue was $394B.",
                "supported": True,
                "sources": [
                    {"chunk_id": "section-42", "citation": "AAPL 2024 10-K, ITEM 7"}
                ]
            },
            ...
        ]
    }
    """

    # Combine all evidence into one searchable string
    source_text = " ".join(item.chunk.text.lower() for item in evidence)

    verdicts = []
    for claim in split_claims(answer):
        # Extract meaningful tokens (skip short stopwords, include numbers)
        words = set(re.findall(r"[a-z]{4,}|\$?[0-9][0-9,.%]*", claim.lower()))

        # How many of those words appear somewhere in the evidence?
        hits = sum(word in source_text for word in words)

        # Supported if: there are meaningful words AND ≥50% appear in evidence
        supported = bool(words) and hits / len(words) >= 0.5

        sources = []
        if supported:
            # Attach citation info from all evidence chunks (claim is grounded)
            for item in evidence:
                sources.append({
                    "chunk_id": item.chunk.chunk_id,
                    "citation": (
                        f"{item.chunk.company} "
                        f"{item.chunk.filing_date} "
                        f"{item.chunk.filing_type}, "
                        f"{item.chunk.section}"
                    ),
                })

        verdicts.append({
            "claim": claim,
            "supported": supported,
            "sources": sources,
        })

    return {
        "all_supported": all(item["supported"] for item in verdicts),
        "claims": verdicts,
    }
