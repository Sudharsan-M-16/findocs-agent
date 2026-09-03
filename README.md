# FinDocs Agent

FinDocs is an agentic research system over SEC 10-K filings. It is built to be explainable, measurable, and defensible in interviews: every retrieval stage can be compared, the corrective loop can be measured, and citation support can be checked.

## What It Builds

The current repo covers Days 0-10 of the roadmap:

1. SEC filing ingestion and text cleanup.
2. Naive chunking and heading-aware SEC Item chunking.
3. Dense retrieval with sentence-transformer embeddings.
4. BM25 keyword retrieval.
5. Reciprocal Rank Fusion over dense and BM25 rankings.
6. Optional cross-encoder reranking over the hybrid candidates.
7. A bounded retrieve, grade, rewrite, retrieve, answer graph.
8. Claim verification and citation formatting.
9. QLoRA grader dataset and training scaffold.
10. Multi-company query decomposition.
11. Retrieval, self-correction, citation, and grader evaluation utilities.

## Quick Start

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
$env:PYTHONPATH='src'
py -m findocs.cli smoke
```

For local Apple data already in `data/apple_10k.txt`:

```powershell
py -m findocs.cli run --company AAPL
py -m findocs.cli ask --company AAPL --question "What are Apple's main risk factors?"
```

To download a missing filing from SEC EDGAR, pass a real identifying email:

```powershell
py -m findocs.cli run --company MSFT --email your.name@example.com
```

SEC requires an identifying User-Agent. The code uses your email for that header.

## Evaluation Commands

Create a labeling sheet for retrieval labels and QLoRA relevance-grader data:

```powershell
py -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
```

Then fill the `label` column manually with `1` for relevant and `0` for irrelevant. Also copy the relevant chunk IDs into `data/eval_questions.json`.

Run retrieval-stage ablations:

```powershell
py -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv
py -m findocs.cli eval-retrieval --companies AAPL --rerank --output results/retrieval_ablation.csv
```

Run retry vs no-retry evaluation:

```powershell
py -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

Test Day 9 decomposition:

```powershell
py -m findocs.cli decompose --question "Compare NVIDIA and Microsoft on research and development spending."
```

## QLoRA Grader

The normal project dependencies do not install GPU training packages. For Day 7-8 training:

```powershell
pip install -r requirements-qlora.txt
py -m findocs.finetune.qlora_train --notes
py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv --output models/qlora-grader
```

The QLoRA script is isolated so the rest of the project remains runnable on machines without Unsloth, bitsandbytes, or a CUDA setup.

## Ownership Map

You wrote or can defend the SEC heading parser, chunk metadata model, RRF fusion logic, graph control flow, rewrite policy, query decomposition, label dataset format, citation verifier, and evaluation harness.

The project uses sentence-transformers for embeddings, rank-bm25 for BM25, sentence-transformers CrossEncoder for reranking, LangGraph for graph execution, and Unsloth/TRL/PEFT for QLoRA training.

## Do Not Fake These Numbers

`data/eval_questions.json` intentionally ships with empty `relevant_chunk_ids`. Fill them only after inspecting retrieved chunks. The ablation runner refuses to produce retrieval metrics for unlabeled questions because fake labels create fake resume bullets.
