"""Small shared data structures used by every pipeline stage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """A retrievable piece of a filing plus the metadata needed for citations."""

    chunk_id: str
    text: str
    company: str
    filing_type: str
    filing_date: str
    section: str
    page_hint: str = ""

    def metadata(self) -> dict[str, str]:
        """Return only JSON-safe metadata for indexes and final citations."""

        return {
            "company": self.company,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "section": self.section,
            "page_hint": self.page_hint,
        }


@dataclass
class RetrievedChunk:
    """A chunk together with comparable rank/score information from one stage."""

    chunk: Chunk
    score: float
    rank: int
    source: str


@dataclass
class AgentState:
    """The state carried through the corrective research graph."""

    question: str
    active_query: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grade: str = ""
    rewrite_count: int = 0
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

