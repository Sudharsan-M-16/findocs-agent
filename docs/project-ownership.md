# FinDocs Agent: Ownership, Completion Status, and Interview Preparation

This document is the source of truth for understanding and presenting the project. It deliberately separates implemented code from
measurements that still require human labels, additional filings, or a CUDA machine. Do not quote a number unless it appears in a
generated result file and you can reproduce it with the commands below.

## What the system is

FinDocs Agent is an inspectable research pipeline over SEC filings:

```text
filing -> cleaned text -> SEC-aware chunks -> dense/BM25 retrieval -> RRF -> optional reranking
                                                           |
                                      retrieve -> grade -> rewrite -> retrieve (bounded loop)
                                                           |
                                      extractive answer -> claim verification -> citations
```

The project has three distinct concerns:

1. Retrieval quality: which chunks are surfaced?
2. Agent behavior: does insufficient evidence cause a bounded rewrite and retry?
3. Evidence quality: do answer claims have support in retrieved chunks?

The QLoRA component is a separate relevance-grader experiment. It can replace the heuristic grader only after it is trained and
evaluated on held-out, human-labeled query/chunk pairs.

## Current implementation status

### Implemented

- SEC EDGAR request and local filing loading with identifying User-Agent support.
- HTML cleanup and text persistence.
- Naive fixed-window chunking for a baseline.
- Heading-aware SEC Item chunking with company, filing type, date, and section metadata.
- Dense retrieval using `sentence-transformers`.
- BM25 retrieval using `rank-bm25`.
- Hand-written RRF fusion.
- Optional cross-encoder reranking over a small candidate set.
- LangGraph retrieve/grade/rewrite/answer flow with a retry cap.
- Conservative extractive answer generation.
- Claim splitting, lexical support checks, citation rendering, and citation correctness scoring.
- Rule-based multi-company query decomposition.
- Retrieval ablation, self-correction ablation, timing helpers, and grader classification metrics.
- Candidate-label CSV generation, candidate inspection, and label-to-evaluation synchronization.
- QLoRA training entry point with lazy GPU-only imports.
- Offline unit tests and a dependency-free smoke command.
- Professional README with no UI requirement. There is intentionally no UI in the project scope.

### Verified in this workspace

- Runtime dependencies were installed successfully.
- The local Apple filing was loaded and indexed into 181 chunks.
- The real embedding retrieval command completed over that filing.
- `data/grader_label_sheet.csv` was generated with 100 candidate rows.
- Source compilation, smoke checks, and the offline test suite passed.
- The project venv now reports CUDA-enabled PyTorch `2.11.0+cu128`, CUDA `12.8`, and an NVIDIA GeForce RTX 4070 Laptop GPU.
- The QLoRA packages `unsloth`, `bitsandbytes`, `trl`, `peft`, and `datasets` are installed in that venv.
- The currently labeled six-question Apple subset produced `results/retrieval_ablation.csv` and
  `results/retrieval_ablation_summary.csv`; these are partial benchmark artifacts, not a full corpus benchmark.
- The same subset produced `results/self_correction.csv` and `results/self_correction_summary.csv`.
- The first QLoRA run completed on the RTX 4070, but its saved evaluation should be treated as a failed baseline: the 100-row dataset is
  90% negative, so `accuracy=0.90` coexists with `precision=recall=F1=0.0`. This indicates a model that predicts no positive examples.
  The latest run measured roughly 463.4 ms/call and 1.325 GiB peak GPU memory; these values describe this run only, not a general guarantee.

### Not yet a valid measured result

- Final retrieval metrics are not complete until every evaluation question has reviewed relevant chunk IDs.
- Final self-correction answer accuracy is not meaningful until answer references/phrases are reviewed against the intended filing evidence.
- A first QLoRA adapter has been trained, but it is not suitable for integration yet; its positive-class metrics were zero because the small
  label set was heavily imbalanced. A balanced held-out experiment is still required.
- LLM-versus-QLoRA latency, cost, and agreement numbers must be measured on actual implementations; placeholders are not results.
- Multi-company evaluation requires local filings for every company in the question.
- The reranked run was not completed because the model could not be loaded from the model hub in the final offline attempt; run it after
  the required models are cached or network access is available.
- The answer generator is intentionally extractive, not an unverified claim-producing LLM.

## Exact code-reading order

Read one file at a time and run the small checkpoint after each group. The order follows data flow, not alphabetical order.

### 1. Data contracts and ingestion

1. `src/findocs/types.py` — learn `Chunk`, `RetrievedChunk`, and `AgentState`. These are the nouns used everywhere else.
2. `src/findocs/ingest/sec_edgar.py` — follow CIK normalization, HTTP headers, HTML cleanup, and persistence.
3. `src/findocs/ingest/chunking.py` — compare naive windows with the custom SEC Item parser.
4. `src/findocs/corpus.py` — see how local aliases, downloads, and chunk metadata are assembled.

Checkpoint: explain why metadata is part of the result, not decoration. It is what makes company/section citations possible.

### 2. Retrieval

5. `src/findocs/retrieval/dense.py` — understand embedding batches, vector normalization, and cosine similarity.
6. `src/findocs/retrieval/bm25.py` — understand tokenization and why exact financial terms benefit from sparse search.
7. `src/findocs/retrieval/hybrid.py` — derive the RRF score by hand: `sum(1 / (rrf_k + rank))`.
8. `src/findocs/retrieval/reranker.py` — understand why a cross-encoder sees query and chunk together and why it only receives top candidates.
9. `src/findocs/eval/pipeline.py` — map each named ablation stage to the callable that implements it.

Checkpoint: explain dense-only, BM25-only, hybrid RRF, and hybrid-plus-reranker without looking at code.

### 3. Agent and evidence handling

10. `src/findocs/agent/answer.py` — follow how evidence becomes an extractive answer and how source labels are rendered.
11. `src/findocs/agent/verify.py` — understand claim splitting, token filtering, overlap threshold, and source attachment.
12. `src/findocs/agent/graph.py` — trace the state fields through retrieve, grade, rewrite, and answer.
13. `src/findocs/agent/decompose.py` — see how company mentions become independent sub-queries.

Checkpoint: draw this transition from memory:

```text
retrieve -> grade
  good   -> answer -> verify
  bad    -> rewrite -> retrieve
  retry limit reached -> answer with the best available evidence
```

### 4. Evaluation

14. `src/findocs/eval/metrics.py` — learn the exact definitions of Recall@5, Recall@10, MRR, Precision@5, answer accuracy, and citation correctness.
15. `src/findocs/eval/ablation.py` — understand per-question rows, stage aggregation, and why unlabeled questions are rejected.
16. `src/findocs/eval/correction.py` — compare `max_retries=0` with retry-enabled execution.
17. `src/findocs/eval/benchmark.py` — understand wall-clock measurement and resource fields.

Checkpoint: explain why retrieval recall and answer accuracy are different metrics. A correct chunk can still produce a bad answer, and a
good-looking answer can be unsupported.

### 5. QLoRA and training data

18. `src/findocs/finetune/dataset.py` — understand candidate generation, manual labels, strict `0/1` loading, and eval-ID synchronization.
19. `src/findocs/finetune/grader_eval.py` — understand confusion-matrix metrics and timing a prediction function.
20. `src/findocs/finetune/qlora_train.py` — understand the model, 4-bit loading, adapter targets, SFT format, and training configuration.

Checkpoint: explain that QLoRA trains small adapter weights while the quantized base is frozen. It is not “training a 1.5B model from
scratch.”

### 6. Entry point and tests

21. `src/findocs/cli.py` — now you understand every command because each command delegates to a module already studied.
22. `tests/test_core.py` — read the tests as executable specifications and identify which behavior is deterministic/offline.
23. `README.md` — read this last as the public product description, then compare every claim with the implementation.

## Commands to operate the project

Set the import path in each new PowerShell session:

```powershell
$env:PYTHONPATH='src'
```

Run deterministic checks:

```powershell
py -m compileall -q src
py -m unittest discover -s tests -v
py -m findocs.cli smoke
py -m findocs.finetune.qlora_train --notes
```

Inspect the real Apple corpus:

```powershell
py -m findocs.cli run --company AAPL
py -m findocs.cli ask --company AAPL --question "What was research and development spending in the latest fiscal year?"
```

Create and inspect labeling candidates:

```powershell
py -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
py -m findocs.cli show-candidates --labels data/grader_label_sheet.csv --question-id q002
```

After filling labels manually, synchronize only the rows marked `1` into the evaluation set:

```powershell
py -m findocs.cli sync-labels --labels data/grader_label_sheet.csv --output data/eval_questions.json
```

Then generate the retrieval and self-correction outputs:

```powershell
py -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv
py -m findocs.cli eval-retrieval --companies AAPL --rerank --output results/retrieval_ablation.csv
py -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

The two retrieval commands should be retained as separate result files if you want to compare a run without and with a reranker; do not
overwrite the first file before copying it or renaming it.

## Manual labeling protocol

The label sheet is the ground truth, not a model prediction. For each row:

- Use `1` only when the chunk contains direct evidence that materially helps answer the question.
- Use `0` when it is merely a table-of-contents entry, a cross-reference, a neighboring topic, or a generic mention.
- Label all candidate rows for a question, including negatives. Otherwise precision and grader metrics are distorted.
- For exact-number questions, require the chunk to contain the actual number or the table/discussion that directly states it.
- For broad questions, label multiple genuinely supporting chunks when each adds distinct evidence.
- Do not label a chunk merely because its section name sounds appropriate.

For `q010`, do not label anything from an Apple-only corpus. First add NVIDIA and Microsoft filings, then regenerate candidates with
`--companies NVDA,MSFT` and label the company-specific evidence separately.

## QLoRA: exact steps to complete the experiment

### A. Prepare the data correctly

1. Generate candidates from the filings you actually intend to evaluate.
2. Inspect every candidate with `show-candidates` or a spreadsheet.
3. Fill every candidate's `label` with exactly `0` or `1`.
4. Check that both classes exist and that no question has only positives or only negatives by accident.
5. Keep a held-out split by question or filing, not a random split of near-duplicate chunks. A random row split can leak neighboring
   chunks from the same filing into both train and validation.
6. Sync relevant IDs into `data/eval_questions.json` for retrieval evaluation.

Recommended minimum for a defensible first experiment: several hundred balanced pairs across multiple companies and question types. The
current 100-row candidate sheet is a starting point, not evidence that the model generalizes.

### B. Use a CUDA environment

The current machine reports CPU-only PyTorch (`torch.cuda.is_available() == False`). Run training on a CUDA-enabled environment with a
compatible NVIDIA driver, CUDA-enabled PyTorch, and enough VRAM. A 1.5B 4-bit QLoRA run is the intended first target, but actual memory
depends on sequence length, batch size, optimizer, and library versions.

Create a clean virtual environment rather than mixing GPU training packages into the global interpreter:

```powershell
py -m venv .venv-qlora
.\.venv-qlora\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-qlora.txt
$env:PYTHONPATH='src'
```

Verify before training:

```powershell
py -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

If the output says `False`, stop there; do not claim a QLoRA run.

### C. Train the adapter

First inspect the configuration and data format:

```powershell
py -m findocs.finetune.qlora_train --notes
```

Train only after the CSV is complete:

```powershell
py -m findocs.finetune.qlora_train `
  --labels data/grader_label_sheet.csv `
  --output models/qlora-grader
```

Record the base model, library versions, GPU name, peak memory, number of labeled rows, class balance, train/validation split, loss
curve, epoch count, and output adapter path. Those are part of the experiment, not optional notes.

### D. Evaluate honestly

Run the existing evaluator after the adapter exists:

```powershell
py -m findocs.finetune.grader_eval `
  --labels data/grader_label_sheet.csv `
  --model models/qlora-grader
```

For a real comparison, implement or connect a genuine reference LLM grader and call both graders on the exact same held-out rows. Record:

- accuracy, precision, recall, and F1;
- mean and preferably p95 latency per call;
- API cost per call for the reference grader;
- peak GPU memory for the local grader;
- agreement with the reference grader and agreement with human labels.

Do not compare a trained model to an uncalibrated word-overlap heuristic and call that “LLM versus QLoRA.” The heuristic is useful as a
cheap baseline, but it is not a large LLM.

### E. Integrate only after validation

The current graph uses a transparent heuristic grade so the offline system remains runnable. Replace it with the QLoRA grader only after
held-out F1 and error analysis show that the adapter is suitable. Keep a configuration switch so you can reproduce heuristic, reference,
and QLoRA runs independently.

## Interview defenses you must own

### Why hybrid retrieval?

Dense retrieval captures paraphrases; BM25 captures exact numbers, tickers, accounting phrases, and rare terms. SEC questions contain both.
The systems are complementary, so the comparison must be measured on the same labeled questions.

### Why RRF instead of averaging scores?

Cosine similarity and BM25 scores have different meanings and scales. Averaging them assumes comparable calibration. RRF uses rank position:

```text
RRF(d) = sum over retrievers of 1 / (k + rank(d))
```

It rewards agreement without pretending raw scores are commensurate.

### Why rerank only candidates?

A cross-encoder jointly attends to the question and chunk, which is more precise but requires a model pass per pair. Hybrid retrieval
narrows hundreds or thousands of chunks to a small candidate set; the cross-encoder spends expensive computation only where it matters.

### What makes the graph agentic?

The graph has state, conditional routing, and a bounded cycle. A failed evidence grade changes the active query and sends execution back to
retrieval. The retry cap makes the behavior reliable and measurable.

### Why train the grader rather than the generator?

The grader is a repeated, narrow binary decision that sits inside a loop, so reducing its latency and API cost has direct system impact.
The generator is a broader language task and needs a much larger, more carefully curated dataset.

### What are the weaknesses?

The current citation check is lexical, the generator is extractive, section detection can encounter messy filing formatting, labels are
human effort and may be incomplete, and the first multi-company decomposer is rule-based. These are limitations to explain, not hide.

### What changes at 800 companies?

The in-memory prototype would become a bottleneck. Production changes would include persistent vector indexes, incremental ingestion,
document/version IDs, metadata filtering, sharding or partitioning, caching, asynchronous embedding, observability, rate-limit handling,
and a more systematic evaluation set. The correctness contracts and stage metrics should remain unchanged.

## Claims you may and may not make

You may say the repository implements the retrieval stages, bounded correction graph, citation verifier, evaluation harness, and QLoRA
training scaffold.

You may not say that hybrid improved Recall@10 by a particular percentage, that retry improved answer accuracy, or that QLoRA achieved a
particular agreement/latency/memory result until the corresponding experiment has been run and the result file is committed.

The strongest honest interview sentence is: “I built the complete measurement path and refused to treat unlabeled or CPU-blocked stages as
results; here is the exact experiment I ran and the error analysis behind the number.”
