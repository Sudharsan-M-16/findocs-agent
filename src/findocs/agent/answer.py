"""Answer formatting helpers that keep citations visible and auditable."""

from findocs.agent.verify import verify_claims
from findocs.types import RetrievedChunk


def extractive_answer(evidence: list[RetrievedChunk]) -> str:
    """Use the top retrieved chunk as a conservative answer baseline."""

    if not evidence:
        return "No evidence was retrieved."
    return evidence[0].chunk.text


def format_cited_answer(answer: str, evidence: list[RetrievedChunk]) -> dict:
    """Return answer text, verification details, and compact source citations."""

    verification = verify_claims(answer, evidence)
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
    """Render the final answer in the form the README and UI can display."""

    payload = format_cited_answer(answer, evidence)
    lines = [payload["answer"].strip(), "", "Sources:"]
    if not payload["sources"]:
        lines.append("- No supporting citation found.")
    for source in payload["sources"]:
        lines.append(f"- [{source['citation']}; chunk {source['chunk_id']}]")
    return "\n".join(lines)
