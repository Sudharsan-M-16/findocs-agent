# Evaluation plan for resume-grade claims

Do not write numbers into the README until these experiments have run on a frozen, hand-labelled benchmark.

## Retrieval ablation

For every question, store one or more relevant chunk IDs in `data/eval_questions.json`. Then pass the same question to four stage
functions and call `evaluate_stages`:

```python
stages = {
    "dense": dense.search,
    "bm25": sparse.search,
    "hybrid_rrf": hybrid.search,
    "hybrid_reranker": lambda q: reranker.rerank(q, hybrid.search(q, k=20), k=10),
}
```

`aggregate_by_stage` produces the table with Recall@5, Recall@10, MRR, and Precision@5. Answer accuracy is measured separately using
hand-written accepted phrases because retrieval can be correct while generation is wrong.

## Self-correction ablation

Run the same questions through a graph configured with `max_retries=0` and then with `max_retries=2`. Record answer accuracy,
citation correctness, average retries, and latency. The comparison is only meaningful if the question set contains genuinely weak
first-retrieval cases; otherwise retries are being tested on easy questions.

## Citation correctness

`verify_claims` emits one record per sentence, with `supported` and human-readable source labels such as `AAPL 2024-09-28 10-K,
ITEM 7`. Report the percentage of claims supported, and separately inspect false positives. Lexical overlap is a baseline, not proof of
entailment; upgrade it later to a judged verifier while retaining this baseline for comparison.

## Grader comparison

Use `measure_callable` for the prompted large grader and QLoRA grader. Feed both the same held-out query/chunk examples and record
agreement/accuracy, mean latency, cost per call, and peak GPU memory. A missing cost or memory measurement must remain blank rather
than being guessed.

## Multi-company discipline

Keep company and filing date in every chunk. Never mix companies in labels without retaining the company field. A comparison question
must have relevant evidence from both companies, so its ground-truth IDs should include both sources.

