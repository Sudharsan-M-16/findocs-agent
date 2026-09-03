# Days 0–6 learning notes

This document is part of the implementation, not optional decoration. After each run, add the exact query, returned chunk IDs,
what was correct or wrong, and what change should address the failure.

## Day 0: dense retrieval

An embedding is a learned feature vector for text. We L2-normalise vectors, so their dot product is cosine similarity. `k` is the
number of candidates returned; larger `k` improves recall but increases context and reranking cost.

## Day 1: chunking

`naive_chunks` establishes a deliberately weak baseline. `heading_aware_chunks` first identifies SEC Item boundaries and only then
cuts large sections into overlapping windows. Every heading-aware chunk carries a section, date, company, and filing type for later
citation display.

## Day 2: hybrid retrieval

BM25 is strong for exact tokens; dense retrieval is strong for paraphrases. Their score scales are not comparable, so RRF combines
rank positions using `1 / (60 + rank)`. Record Recall@5 and Recall@10 against a hand-labelled relevant chunk for each question.

## Day 3: reranking

The bi-encoder retrieves cheaply by encoding query and document separately. The cross-encoder reads the pair together and can order
subtle candidates better, but is too expensive to run over the complete filing.

## Days 4–5: corrective graph

The graph state contains the active query, evidence, grade, retry count, answer, verification, and trace. A bad grade rewrites the
query and loops back to retrieval. `max_retries` is the safety cap that prevents an infinite loop.

## Day 6: verification

Retrieval quality and answer faithfulness are different. `verify_claims` performs a conservative lexical evidence check and records
which retrieved chunk IDs support each sentence. It is intentionally a baseline to replace with an entailment model or judged LLM
after you have an honest benchmark.

