"""
answer.py — Answer Formatting and Citation Rendering
=====================================================
This module sits between the retrieval/verification pipeline and the CLI output.

THREE FUNCTIONS, THREE RESPONSIBILITIES:
1. extractive_answer()    : Select what text becomes the answer.
2. format_cited_answer()  : Run verification and structure the result as a dict.
3. render_cited_answer()  : Produce the human-readable string for CLI/UI display.

WHY SEPARATE THESE?
- extractive_answer() can be swapped for an LLM generator without touching
  the citation and rendering logic.
- format_cited_answer() returns structured data (dict) so the eval harness
  can inspect verification results programmatically.
- render_cited_answer() is only for human display — keeping it separate means
  automated tests never have to parse formatted strings.

CURRENT GENERATOR: EXTRACTIVE (intentional baseline)
Instead of asking an LLM to synthesise an answer, we take the top retrieved
chunk's text verbatim. This is the most conservative possible approach:
- Zero hallucination risk (no generation, just selection).
- Fully auditable (you know exactly where the answer came from).
- Lower readability (raw filing text is dense and formal).

THE UPGRADE PATH:
Replace extractive_answer() with an LLM call like:
    def llm_answer(question: str, evidence: list[RetrievedChunk]) -> str:
        context = "\\n\\n".join(c.chunk.text for c in evidence)
        return call_llm(f"Answer based only on this context:\\n{context}\\n\\nQuestion: {question}")
Then verify_claims() catches any hallucinated claims before they reach the user.
"""

from findocs.agent.verify import verify_claims
from findocs.types import RetrievedChunk


def extractive_answer(evidence: list[RetrievedChunk]) -> str:
    """
    Use the top retrieved chunk as a conservative answer baseline.

    WHY TOP CHUNK ONLY?
    After hybrid retrieval + optional reranking, rank 1 is the model's best
    guess at the most relevant chunk. Taking its text directly is the safest
    answer we can produce without an LLM.

    EDGE CASE: empty evidence.
    If the retriever returned nothing (e.g., corpus is empty or query failed
    catastrophically), we return a explicit "no evidence" message rather than
    an empty string. This makes the failure mode visible rather than silent.

    LIMITATION:
    A real financial question might require combining information from two chunks
    ("What was total revenue?" might need the header chunk + the numbers table).
    Extractive single-chunk answers will miss multi-chunk synthesis entirely.
    This is acceptable for the baseline and documented as a known limitation.
    """

    if not evidence:
        return "No evidence was retrieved."
    # evidence[0] is rank 1 — highest-scoring chunk after all retrieval stages
    return evidence[0].chunk.text


def format_cited_answer(answer: str, evidence: list[RetrievedChunk]) -> dict:
    """
    Return answer text, verification details, and compact source citations.

    WHAT verify_claims() RETURNS:
    {
        "all_supported": True/False,
        "claims": [
            {"claim": "...", "supported": True, "sources": [{"chunk_id": ..., "citation": ...}]}
        ]
    }

    WHAT THIS FUNCTION DOES WITH THAT:
    1. Run verification.
    2. Collect unique chunk sources from all SUPPORTED claims.
       WHY unique? A supported claim cites all evidence chunks (not just the
       best-matching one). Multiple claims might all cite the same chunk. We
       deduplicate so the output sources list is clean.
    3. Return a dict with three keys:
       - "answer": the raw answer text
       - "sources": list of unique cited chunk dicts
       - "verification": the full per-claim verification structure

    WHY RETURN A DICT AND NOT A DATACLASS?
    The graph state (GraphState) stores verification as a plain dict. Returning
    a dict here means graph.py can store the result without any conversion.
    """

    verification = verify_claims(answer, evidence)

    # Collect unique sources from supported claims only
    seen: set[str] = set()
    sources: list[dict[str, str]] = []
    for claim in verification.get("claims", []):
        for source in claim.get("sources", []):
            chunk_id = source["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                sources.append(source)

    return {"answer": answer, "sources": sources, "verification": verification}


def render_cited_answer(answer: str, evidence: list[RetrievedChunk]) -> str:
    """
    Render the final answer in the form the CLI can display.

    OUTPUT FORMAT:
        {answer text}

        Sources:
        - [AAPL 2024 10-K, ITEM 7; chunk section-42]
        - [AAPL 2024 10-K, ITEM 1A; chunk section-7]
        — or —
        - No supporting citation found.

    WHY "No supporting citation found." INSTEAD OF EMPTY?
    If verification found no supported claims, the user should know the answer
    is unverified — not just see a blank Sources section. This is part of the
    honesty-by-design principle: make failure modes explicit, not silent.

    TRUNCATION NOTE:
    cli.py truncates this output to 3500 characters for readability. The full
    output is available via state["verification"] for programmatic access.
    """

    payload = format_cited_answer(answer, evidence)

    lines = [payload["answer"].strip(), "", "Sources:"]

    if not payload["sources"]:
        lines.append("- No supporting citation found.")
    for source in payload["sources"]:
        lines.append(f"- [{source['citation']}; chunk {source['chunk_id']}]")

    return "\n".join(lines)
