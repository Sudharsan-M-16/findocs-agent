"""Keyword retrieval for exact numbers, names, and acronyms."""

import re
from rank_bm25 import BM25Okapi

from findocs.types import Chunk, RetrievedChunk


def tokens(text: str) -> list[str]:
    """Use simple lowercase word tokens; retaining digits helps exact figures."""

    return re.findall(r"[a-zA-Z0-9$%.,-]+", text.lower())


class BM25Retriever:
    """Build one BM25 index over the same chunks used by dense retrieval."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.index = BM25Okapi([tokens(c.text) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return keyword-ranked chunks with BM25 scores."""

        scores = self.index.get_scores(tokens(query))
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [RetrievedChunk(self.chunks[i], float(scores[i]), rank + 1, "bm25") for rank, i in enumerate(indices)]

