# FinDocs Agent

FinDocs Agent is an agentic research assistant for SEC 10-K filings. It combines hybrid retrieval, reranking, self-correction, and citation verification to answer financial-document questions with traceable supporting evidence.

The system is designed for company filings where simple keyword search is not enough: questions may involve exact figures, paraphrased business descriptions, risk disclosures, or comparisons across companies.

## Features

- SEC EDGAR ingestion for supported public companies
- Filing-aware chunking using SEC Item sections
- Dense semantic retrieval with sentence-transformer embeddings
- Sparse keyword retrieval with BM25
- Reciprocal Rank Fusion (RRF) for hybrid dense + sparse retrieval
- Optional cross-encoder reranking over retrieved candidates
- Bounded retrieve-grade-rewrite loop using LangGraph
- Citation formatting and claim-support verification
- Multi-company query decomposition for comparative research
- Evaluation utilities for retrieval quality, answer accuracy, self-correction, citation correctness, and grader performance
- QLoRA relevance-grader training scaffold for replacing an expensive LLM-based retrieval grader

## Architecture

```text
User question
    |
    v
Query decomposition
    |
    v
Dense retrieval + BM25 retrieval
    |
    v
Reciprocal Rank Fusion
    |
    v
Cross-encoder reranking
    |
    v
Retrieved evidence
    |
    v
Retrieval grader
    |
    +-- insufficient evidence --> query rewrite --> retrieve again
    |
    v
Answer generation
    |
    v
Claim verification + citations
```

## Repository Structure

```text
findocs-agent/
|-- data/                  # Local filings and evaluation question files
|-- docs/                  # Implementation notes and code walkthroughs
|-- results/               # Generated evaluation outputs
|-- src/findocs/
|   |-- agent/             # Graph, answer formatting, verification, decomposition
|   |-- eval/              # Metrics, ablation runners, correction evaluation
|   |-- finetune/          # QLoRA grader dataset/training/evaluation utilities
|   |-- ingest/            # SEC EDGAR loading and filing chunking
|   `-- retrieval/         # Dense, BM25, RRF hybrid retrieval, reranking
|-- tests/                 # Offline unit tests
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

Run the lightweight smoke test:

```powershell
py -m findocs.cli smoke
```

## Usage

Run retrieval over a locally available filing:

```powershell
py -m findocs.cli run --company AAPL
```

Ask a question through the agent:

```powershell
py -m findocs.cli ask --company AAPL --question "What are Apple's main risk factors?"
```

Enable cross-encoder reranking:

```powershell
py -m findocs.cli ask --company AAPL --question "What are Apple's main risk factors?" --rerank
```

Download a missing filing from SEC EDGAR:

```powershell
py -m findocs.cli run --company MSFT --email your.name@example.com
```

SEC EDGAR requires an identifying User-Agent. The project uses the provided email address for that request header.

## Evaluation

Generate a candidate-labeling sheet:

```powershell
py -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
```

Fill the `label` column with `1` for relevant chunks and `0` for irrelevant chunks. Then copy the relevant chunk IDs into `data/eval_questions.json`.

Run retrieval ablations:

```powershell
py -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv
py -m findocs.cli eval-retrieval --companies AAPL --rerank --output results/retrieval_ablation.csv
```

The retrieval evaluation reports:

- Recall@5
- Recall@10
- MRR
- Precision@5

Run self-correction evaluation:

```powershell
py -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

This compares answer accuracy and citation correctness with retries disabled versus enabled.

## Multi-Company Questions

The query decomposer can split comparison questions into company-specific sub-queries:

```powershell
py -m findocs.cli decompose --question "Compare NVIDIA and Microsoft on research and development spending."
```

For full multi-company evaluation, add the relevant filings locally or download them with SEC EDGAR using `--email`.

## QLoRA Relevance Grader

The QLoRA training dependencies are separated from the main runtime dependencies because they require GPU-specific packages.

```powershell
pip install -r requirements-qlora.txt
py -m findocs.finetune.qlora_train --notes
py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv --output models/qlora-grader
```

The grader evaluation utilities support comparing an LLM-based relevance grader against a fine-tuned small model using:

- Accuracy
- Precision
- Recall
- F1
- Latency
- Estimated cost per call
- GPU memory usage

## Testing

```powershell
$env:PYTHONPATH='src'
py -m compileall -q src
py -m unittest discover -s tests -v
```

## Current Limitations

- Evaluation metrics require hand-labeled relevant chunk IDs before results are meaningful.
- The default answer generator is conservative and extractive.
- Citation verification currently uses lexical overlap as a transparent baseline, not full natural-language entailment.
- QLoRA training requires a CUDA-compatible environment and completed relevance labels.
- Multi-company comparisons require filings for each company to be present in the corpus.
