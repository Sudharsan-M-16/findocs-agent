# Code tour: how to explain the implementation

This is the line-by-line speaking guide. Read the docstring immediately above a function, then explain its body in this order:

1. `types.py` defines the contract. A `Chunk` is text plus citation metadata; a `RetrievedChunk` adds a score and rank; `AgentState`
   is the information that must survive between graph nodes.
2. `sec_edgar.py` uses the SEC submissions JSON to find the newest 10-K, constructs the archive URL from the accession number, removes
   scripts/styles from HTML, normalises whitespace, and writes a local text artifact. The User-Agent is both SEC policy and good API
   citizenship.
3. `chunking.py` has two intentionally different baselines. The naive loop advances by `size - overlap`; the heading-aware parser
   first records every `Item` boundary and only then applies the same overlapping-window idea inside each section.
4. `dense.py` encodes all chunks once. Normalisation makes `vectors @ query_vector` cosine similarity. `argsort(-scores)` means
   highest similarity appears first.
5. `bm25.py` tokenises words, digits, dollar values, and percentages. That makes exact finance terms visible to sparse search.
6. `hybrid.py` never adds a cosine score to a BM25 score. It adds `1/(k+rank)` contributions, aggregates by chunk ID, and sorts the
   resulting common scale. This is the core ranking decision you should be able to derive on a whiteboard.
7. `reranker.py` deliberately sees only the hybrid candidates. The cross-encoder receives `(query, chunk)` pairs together, which is
   slower but lets it model interactions that independent embeddings miss.
8. `graph.py` is a state machine. Retrieve populates evidence; grade decides good/bad; rewrite changes the query and increments the
   counter; answer creates an extractive baseline and invokes verification. The conditional edge either loops or terminates.
9. `verify.py` splits the answer into claims and checks meaningful token overlap with evidence. This baseline is intentionally
   conservative and imperfect; document its false positives/negatives before replacing it with an LLM or entailment model.
10. `eval/metrics.py` is the beginning of the resume-grade measurement story. Every question must eventually contain hand-checked
    relevant chunk IDs and accepted answer phrases; otherwise the numbers are not evidence.
11. `corpus.py` is the shared loading boundary. CLI commands should not each rediscover how to load files, download missing filings,
    or create heading-aware chunks.
12. `agent/answer.py` separates answer text from citation rendering. The graph stores structured verification, while the CLI renders
    a readable answer with sources.
13. `agent/decompose.py` is the first multi-hop planner. It is rule-based on purpose, so the trace is easy to defend before replacing
    it with an LLM planner.
14. `finetune/dataset.py` creates the CSV you manually label. This is the bridge between retrieval experiments and QLoRA training.
15. `finetune/qlora_train.py` keeps GPU-only training code isolated. The normal project imports must not break just because CUDA
    packages are absent.
16. `finetune/grader_eval.py` computes accuracy, precision, recall, F1, latency, cost, and memory fields for grader comparisons.
17. `eval/pipeline.py` builds all retrieval stages from the same chunk list. That keeps dense, BM25, RRF, and reranker comparisons fair.
18. `eval/correction.py` runs the same graph with different retry caps. This is the measurement behind the self-correction claim.
19. `cli.py` is the user-facing control surface. Each command maps to a project milestone: inspect retrieval, ask the agent, label
    candidates, run retrieval ablations, run correction ablations, or decompose multi-company questions.

For every experiment, save: question, stage, returned IDs, metric row, and a one-sentence diagnosis. Do not fill `eval_questions.json`
with IDs until you have inspected the actual filing; invented labels would make the evaluation meaningless.
