# FinDocs Agent

FinDocs Agent is an inspectable research assistant for SEC 10-K filings. It combines filing-aware chunking, dense and BM25 retrieval,
Reciprocal Rank Fusion, optional cross-encoder reranking, bounded query correction, and citation verification.

The system is designed for questions involving exact financial figures, paraphrased disclosures, risk factors, and comparative research
across companies. It intentionally has no UI requirement: the CLI, traces, CSV results, and tests are the project surface.

## Capabilities

- SEC EDGAR ingestion with identifying User-Agent support
- Heading-aware SEC Item chunking with citation metadata
- Dense semantic retrieval with `sentence-transformers`
- Sparse keyword retrieval with BM25
- Hand-written Reciprocal Rank Fusion
- Optional cross-encoder reranking over retrieved candidates
- Bounded LangGraph retrieve-grade-rewrite loop
- Conservative extractive answers with source citations
- Claim-level lexical citation verification
- Rule-based multi-company query decomposition
- Retrieval, correction, citation, and grader evaluation utilities
- QLoRA relevance-grader training scaffold with lazy GPU imports

## Architecture

```text
question -> optional company decomposition -> dense + BM25 -> RRF -> optional reranker
                                                      |
                                     retrieve -> grade -> rewrite -> retrieve
                                                      |
                                     extractive answer -> verify claims -> citations
```

## Repository

```text
findocs-agent/
|-- data/                  local filings and evaluation questions
|-- docs/                  ownership and implementation guides
|-- results/               generated evaluation CSVs
|-- src/findocs/
|   |-- agent/             graph, answers, verification, decomposition
|   |-- eval/              metrics and ablation runners
|   |-- finetune/          labels, QLoRA training, grader evaluation
|   |-- ingest/            SEC loading and chunking
|   `-- retrieval/         dense, BM25, RRF, reranking
|-- tests/                 deterministic offline tests
|-- requirements.txt
|-- requirements-qlora.txt
`-- pyproject.toml
```

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
$env:PYTHONPATH='src'
```

Run deterministic checks:

```powershell
py -m compileall -q src
py -m unittest discover -s tests -v
py -m findocs.cli smoke
```

## Usage

Run retrieval over a local filing:

```powershell
py -m findocs.cli run --company AAPL
```

Ask through the bounded agent:

```powershell
py -m findocs.cli ask --company AAPL --question "What was research and development spending in the latest fiscal year?"
py -m findocs.cli ask --company AAPL --question "What are Apple's main risk factors?" --rerank
```

If a filing is not local, download it from SEC EDGAR by providing a real identifying email:

```powershell
py -m findocs.cli run --company MSFT --email your.name@example.com
```

## Evaluation workflow

Create candidate rows for human labeling:

```powershell
py -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
py -m findocs.cli show-candidates --labels data/grader_label_sheet.csv --question-id q002
```

Fill every candidate row's `label` column with `1` only when the chunk materially supports the question, otherwise `0`. Then synchronize
the relevant chunk IDs into the benchmark:

```powershell
py -m findocs.cli sync-labels --labels data/grader_label_sheet.csv --output data/eval_questions.json
```

Run the retrieval-stage comparison:

```powershell
py -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv
py -m findocs.cli eval-retrieval --companies AAPL --rerank --output results/retrieval_ablation_reranked.csv
```

The outputs contain per-question and summary metrics for dense-only, BM25-only, RRF hybrid, and hybrid-plus-reranker:

- Recall@5
- Recall@10
- MRR
- Precision@5

Run the retry comparison:

```powershell
py -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

This reports answer-phrase accuracy, citation correctness, and retries used for retry-disabled and retry-enabled execution.

For multi-company decomposition:

```powershell
py -m findocs.cli decompose --question "Compare NVIDIA and Microsoft on research and development spending."
```

Multi-company evaluation is valid only when the corresponding filings are in the corpus.

## QLoRA relevance grader

The training path is intentionally separated from normal runtime dependencies:

```powershell
pip install -r requirements-qlora.txt
$env:PYTHONPATH='src'
py -m findocs.finetune.qlora_train --notes
py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv --output models/qlora-grader
py -m findocs.finetune.grader_eval --labels data/grader_label_sheet.csv --model models/qlora-grader
```

Training requires completed human labels and CUDA-compatible PyTorch, Unsloth, and bitsandbytes. The current implementation uses a
1.5B Qwen instruction model loaded in 4-bit mode with rank-16 LoRA adapters. The model choice and hyperparameters are documented in
`src/findocs/finetune/qlora_train.py` and explained in `docs/project-ownership.md`.

## Results policy

This repository does not hard-code retrieval, answer, citation, latency, cost, or QLoRA accuracy numbers. Those numbers are valid only
after the relevant experiment has been run against hand-labeled data and saved under `results/`. Empty ground-truth IDs are rejected by
the retrieval evaluator rather than converted into misleading zero scores.

## Documentation

Read [`docs/project-ownership.md`](docs/project-ownership.md) for the complete implementation status, code-reading order, labeling rules,
QLoRA procedure, evaluation protocol, and interview defenses. Read [`docs/finish-blocked-work.md`](docs/finish-blocked-work.md) for the
GPU, model-cache, additional-filings, benchmark, and QLoRA completion procedure.

## Limitations

- The default answer generator is extractive and conservative.
- Citation verification is a transparent lexical baseline, not a full entailment model.
- SEC filings can contain malformed HTML, tables, and references that challenge simple heading detection.
- Rule-based decomposition is a first transparent baseline, not a general planner.
- QLoRA and multi-company results require additional data and a suitable CUDA environment.
