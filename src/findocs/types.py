"""
types.py — The Shared Data Contract
====================================
Every other module in this project imports from here. It defines three
dataclasses:
  - Chunk:          a piece of a filing with citation metadata attached
  - RetrievedChunk: a Chunk plus ranking information from one retrieval stage
  - AgentState:     the state dict that flows through the LangGraph state machine

WHY DATACLASSES?
A plain dict would work but gives no type safety and no tab-completion.
A dataclass is a Python class that auto-generates __init__, __repr__, and
__eq__ from its field declarations, with zero boilerplate.

WHY A SEPARATE types.py?
It breaks import cycles: retrieval, agent, eval, and finetune modules all
need Chunk without depending on each other.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """
    A retrievable piece of a filing plus the metadata needed for citations.

    FIELDS:
    -------
    chunk_id    : Unique string ID, e.g. "section-42". Used as the primary key
                  in retrieval results and eval label sheets.
    text        : The actual filing text for this window. This is what gets
                  embedded by dense retrieval and tokenised by BM25.
    company     : Ticker symbol, e.g. "AAPL". Carried so citations can name the
                  source company even when chunks from several filings are mixed.
    filing_type : Always "10-K" in this project, but kept generic.
    filing_date : "unknown" when the filename has no date, otherwise ISO date
                  string extracted from the filename by load_filing().
    section     : The SEC Item name this chunk came from, e.g. "ITEM 1A". "unknown"
                  for naive chunks or front matter.
    page_hint   : Optional rough page number. Currently empty; kept so a later
                  version can add page-level citation precision.

    WHY ATTACH METADATA AT CHUNK LEVEL?
    Dense embeddings are stored as raw vectors; when a result comes back there
    is no way to recover the filing or section unless we attached that context
    before embedding. This design means every retrieved object is self-describing.
    """

    chunk_id: str
    text: str
    company: str
    filing_type: str
    filing_date: str
    section: str
    page_hint: str = ""

    def metadata(self) -> dict[str, str]:
        """
        Return only JSON-safe metadata for indexes and final citations.

        WHY EXCLUDE chunk_id and text?
        The vector DB stores the text separately. chunk_id is the DB's document
        ID, not payload. This method is called when writing Chroma metadata.
        """

        return {
            "company": self.company,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "section": self.section,
            "page_hint": self.page_hint,
        }


@dataclass
class RetrievedChunk:
    """
    A chunk together with comparable rank/score information from one stage.

    WHY WRAP CHUNK INSTEAD OF EXTENDING IT?
    A Chunk is a static document property. Score and rank are query-relative
    and stage-relative: the same chunk can rank 3rd in dense search and 1st
    after reranking. Wrapping keeps the two concerns separate.

    FIELDS:
    -------
    chunk  : The underlying Chunk dataclass.
    score  : Raw score from this stage (cosine similarity, BM25, RRF, or
             cross-encoder score). Do NOT compare scores across stages — their
             scales are incompatible. Only compare ranks.
    rank   : 1-indexed position in this stage's result list. Used by RRF and
             by metrics (recall@k checks whether rank <= k).
    source : Human-readable label for which stage produced this result, e.g.
             "dense", "bm25", "dense+bm25", "reranker". Used in the ablation
             CSV so you can trace back which pipeline stage each row came from.
    """

    chunk: Chunk
    score: float
    rank: int
    source: str


@dataclass
class AgentState:
    """
    The state carried through the corrective research graph.

    LangGraph nodes receive this dict-like object and return a partial update.
    Only the keys they return are merged; everything else is preserved.

    WHY TRACK rewrite_count?
    The loop cap (max_retries in graph.py) reads retries from state. Without
    this counter the retrieve→grade→rewrite cycle would spin forever on hard
    questions.

    WHY KEEP trace?
    The trace list accumulates one string per node visit. At the end of a run,
    cli.py prints it so you can see "retrieve: → grade:bad → rewrite: → ..."
    in sequence. This is what makes the agentic claim inspectable.

    NOTE: graph.py uses TypedDict (not this dataclass) because LangGraph's
    StateGraph expects a TypedDict schema. AgentState is kept here for
    documentation parity and could replace TypedDict if LangGraph ever
    supports Pydantic/dataclass states.
    """

    question: str
    active_query: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    grade: str = ""
    rewrite_count: int = 0
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)
