"""
bm25.py — BM25 Keyword (Sparse) Retrieval
==========================================
BM25 = Best Match 25. It's the gold standard sparse retrieval algorithm,
used by Elasticsearch, Solr, and Lucene under the hood.

THREE COMPONENTS YOU MUST BE ABLE TO EXPLAIN IN AN INTERVIEW:
---------------------------------------------------------------
1. TF  (Term Frequency):
   How often the query term appears in this document.
   More occurrences → higher relevance. BUT not linearly — BM25 uses a
   saturation curve: the first occurrence matters most, and extra occurrences
   add diminishing returns. This prevents a document that repeats "revenue"
   100 times from dominating over one that uses it meaningfully twice.

2. IDF (Inverse Document Frequency):
   How rare the term is across all documents.
   "Revenue" appears in EVERY filing → low IDF (common term, less signal).
   "$85.8B" might appear in only one chunk → high IDF (specific, more signal).
   log((N - n + 0.5) / (n + 0.5)) where N = total docs, n = docs with term.

3. DL  (Document Length) Normalisation:
   Longer documents contain more terms by chance. BM25 normalises by document
   length so a long filing section doesn't win just because it has more words.
   The parameter b (default 0.75) controls how strongly length is penalised.

WHY BM25 BEATS DENSE RETRIEVAL FOR EXACT TERMS:
- "$85.8 billion", "ITEM 7A", "CIK 0000320193" — these don't have "semantic
  meaning" in the embedding space. Dense embeddings mix many training signal;
  exact rare tokens like figures and codes get diluted.
- BM25 treats them as high-IDF terms (rare = important) and ranks chunks
  containing them directly.

WHY BM25 FAILS WHERE DENSE WINS:
- "Sales dropped" vs "revenue declined" — zero shared tokens → BM25 score 0.
- Dense embeddings capture paraphrase; BM25 cannot.
"""

import re
from rank_bm25 import BM25Okapi

from findocs.types import Chunk, RetrievedChunk


def tokens(text: str) -> list[str]:
    """
    Tokenise text into a list of lowercase words and numbers.

    WHY THIS TOKENISER (NOT JUST .split())?
    .split() gives you "revenue," and "revenue." as different tokens.
    This regex extracts sequences of alphanumerics and finance-relevant
    punctuation: $ % . , - so that "$85.8B" is one token, not four.

    WHY LOWERCASE?
    BM25 is case-sensitive by default. "Revenue" and "revenue" would be
    different tokens. Lowercasing unifies them so "what is Revenue" and
    "what is revenue" match the same chunks.

    WHY KEEP DIGITS?
    Exact financial figures are high-IDF tokens (rare in the corpus). If we
    strip digits, "2.1 billion R&D" → ["billion", "r&d"] — we lose the
    specificity of the figure.

    EXAMPLE:
    "Apple's R&D was $29.9B in FY2024." →
    ["apple's", "r&d", "was", "$29.9b", "in", "fy2024."]

    rank_bm25's BM25Okapi takes a list-of-lists: one list of tokens per doc.
    """

    return re.findall(r"[a-zA-Z0-9$%.,-]+", text.lower())


class BM25Retriever:
    """
    Build one BM25 index over the same chunks used by dense retrieval.

    WHY SAME CHUNKS?
    Retrieval stage comparisons are only fair if all methods see the same
    corpus. We pass the identical list of Chunk objects to both DenseRetriever
    and BM25Retriever.

    RANK-BM25 LIBRARY:
    BM25Okapi is the standard BM25 variant. The library handles TF/IDF/DL
    calculations internally. We provide it with tokenised documents and it
    builds an inverted index in memory.

    MEMORY/SPEED:
    The BM25 index is essentially a sparse term-frequency matrix. For 181 Apple
    chunks with ~500 unique terms each, this is trivially small. At 1M chunks
    you would need an inverted index backed by disk (Elasticsearch-style).
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """
        Build the BM25 index from all chunk texts.

        BM25Okapi([tokens(c.text) for c in chunks]):
        - For each chunk, tokenise its text.
        - Pass the list of token lists to BM25Okapi, which builds the index.
        This runs once and is fast (<1s for hundreds of chunks).
        """

        self.chunks = chunks
        # Build the inverted index. Each element is a list of tokens for one document.
        self.index = BM25Okapi([tokens(c.text) for c in chunks])

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """
        Return keyword-ranked chunks with BM25 scores.

        HOW get_scores() WORKS:
        1. Tokenise the query.
        2. For each query token, look up its IDF (precomputed from corpus).
        3. For each document, sum the BM25 score contributions from all query
           tokens. This gives one score per document.
        4. Returns a numpy array of length = number of chunks.

        WHY NOT JUST USE TOP-K FROM THE LIBRARY?
        BM25Okapi has get_top_n() but we need RetrievedChunk objects with rank
        and source information for the rest of the pipeline. Sorting the raw
        score array ourselves gives us full control over the output format.

        NOTE: Many chunks will have score 0.0 (no query terms present).
        That's expected — BM25 is sparse. The top-k will be non-zero.
        """

        scores = self.index.get_scores(tokens(query))  # numpy array, length = len(self.chunks)
        # Sort indices by score descending, take top k
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedChunk(self.chunks[i], float(scores[i]), rank + 1, "bm25")
            for rank, i in enumerate(indices)
        ]
