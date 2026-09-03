# Day 7-10 Notes

## Day 7-8: QLoRA Grader

The grader's job is smaller than answer generation: given a question and a retrieved chunk, decide whether the chunk is relevant evidence. That makes it a good target for a small fine-tuned model.

The dataset starts from `label-candidates`. That command retrieves candidate chunks for each benchmark question and writes a CSV. You manually fill `label` with `1` or `0`. This same CSV supports two things:

- training the QLoRA grader;
- evaluating whether the QLoRA grader agrees with held-out human labels.

`src/findocs/finetune/qlora_train.py` keeps GPU-only imports inside `train()` so normal tests do not fail on machines without CUDA dependencies. The important hyperparameters are documented in `hyperparameter_notes()` because those are interview questions, not magic constants.

## Day 9: Query Decomposition

`decompose_query()` looks for supported company names and tickers. If it sees fewer than two companies, it leaves the question alone. If it sees two or more, it creates one company-specific sub-query per company.

This is deliberately simple. The goal for the first version is traceability: you can show exactly which sub-query pulled which evidence before adding a stronger planner later.

## Day 10: Evaluation Harness

The retrieval ablation compares the same questions and same labels across:

- `dense_only`;
- `bm25_only`;
- `dense_bm25_rrf`;
- `hybrid_reranker`.

The metrics are:

- Recall@5: did at least one relevant chunk appear in the top five?
- Recall@10: did at least one relevant chunk appear in the top ten?
- MRR: how high was the first relevant chunk?
- Precision@5: how much of the top five was actually relevant?

The self-correction evaluation runs the same graph twice:

- `without_retry` uses `max_retries=0`;
- `with_retry` uses `max_retries=2`.

That is how you make the agentic claim measurable instead of just saying "the agent can retry."
