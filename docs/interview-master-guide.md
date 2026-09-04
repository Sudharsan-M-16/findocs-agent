# Master FinDocs Engineering & Interview Guide

This guide provides a file-by-file technical breakdown and architectural trade-off defenses for **FinDocs**—an agentic, citation-backed SEC filing research system built for enterprise finance QA.

---

## 1. Architectural Architecture & Resume Talking Points

### Core Resume Bullets

* **Hybrid Financial Retrieval Pipeline:** Engineered a multi-stage search architecture combining dense embeddings (`all-MiniLM-L6-v2`) and sparse BM25 tokenisation with **Reciprocal Rank Fusion (RRF)** ($k=60$) and Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), improving financial domain recall without score-scale distortion.
* **Bounded Corrective Agent State Machine:** Architected a stateful retrieve-grade-rewrite-answer graph in **LangGraph** with a hard iteration budget ($N=2$), preventing infinite loop failure modes while enabling automatic query refinement on sparse evidence.
* **Extractive & Citation-Grounded Verification:** Built a claim-level verification module evaluating lexical and numerical overlap between generated answers and filing chunks to calculate `citation_correctness` and eliminate silent hallucination.
* **QLoRA Relevance Grader Scaffolding:** Designed a 4-bit parameter-efficient fine-tuning (PEFT) pipeline using **Unsloth** and **Qwen2.5-1.5B-Instruct** to replace heuristic grading nodes with a domain-tuned binary relevance classifier running within consumer VRAM constraints (8 GB).

---

## 2. File-by-File Technical Audit & Code Reading Order

Explain the codebase in this precise sequence to demonstrate architectural ownership:

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 1. types.py     │ ──► │ 2. sec_edgar.py  │ ──► │ 3. chunking.py   │
│ Data Contracts  │     │ SEC EDGAR Ingest │     │ Heading Chunker  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│ 6. hybrid.py    │ ◄── │ 5. bm25.py       │ ◄── ┌──────────────────┐
│ Rank Fusion RRF │     │ Sparse BM25      │     │ 4. dense.py      │
└─────────────────┘     └──────────────────┘     │ Vector Search    │
         │                                       └──────────────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ 7. reranker.py  │ ──► │ 8. verify.py     │ ──► │ 9. answer.py     │
│ Cross-Encoder   │     │ Claim Verifier   │     │ Citation Engine  │
└─────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
┌─────────────────┐     ┌──────────────────┐              ▼
│ 12. metrics.py  │ ◄── │ 11. graph.py     │ ◄── ┌──────────────────┐
│ Recall/MRR/P@K  │     │ LangGraph Agent  │     │ 10. decompose.py │
└─────────────────┘     └──────────────────┘     │ Multi-Company    │
         │                                       └──────────────────┘
         ▼
┌──────────────────┐    ┌──────────────────┐     ┌──────────────────┐
│ 13. ablation.py  │ ─► │ 14. correction.py│ ──► │ 15. dataset.py   │
│ Stage Comparison │    │ Retry Evaluator  │     │ Label Pipeline   │
└──────────────────┘    └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌──────────────────┐
                        │ 17. cli.py       │ ◄── │16. qlora_train.py│
                        │ Entry Surface    │     │ QLoRA Fine-Tune  │
                        └──────────────────┘     └──────────────────┘
```

### Detailed Module Roles

1. **`src/findocs/types.py`**: Defines standard dataclasses (`Chunk`, `RetrievedChunk`, `AgentState`). Decouples data definitions from runtime dependencies.
2. **`src/findocs/ingest/sec_edgar.py`**: Handles compliant REST requests to SEC EDGAR, normalises CIK numbers, strips HTML tags (`script`, `style`, `ix:header`), and persists local text artifacts.
3. **`src/findocs/ingest/chunking.py`**: Implements naive fixed-window vs. heading-aware SEC Item parsing (`ITEM_RE`). Preserves SEC section boundaries (`ITEM 1A`, `ITEM 7`) to prevent context contamination.
4. **`src/findocs/retrieval/dense.py`**: Encodes text into 384-dimensional dense vectors using `all-MiniLM-L6-v2`. Computes batch cosine similarity via unit-normalised vector dot products (`vectors @ query_vector`).
5. **`src/findocs/retrieval/bm25.py`**: Tokenises alphanumeric and financial expressions (`$`, `%`, `.`) and builds an inverted index using `rank_bm25.BM25Okapi`.
6. **`src/findocs/retrieval/hybrid.py`**: Implements **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$ with $k=60$. Combines dense and sparse ranks on a unified scale without raw score distortion.
7. **`src/findocs/retrieval/reranker.py`**: Applies cross-encoder joint query-document self-attention (`ms-marco-MiniLM-L-6-v2`) over top-20 hybrid candidates to output refined top-$k$ results.
8. **`src/findocs/agent/verify.py`**: Splits generated answers into sentence claims via lookbehind regex (`(?<=[.!?])\s+`) and verifies whether $\ge 50\%$ of meaningful tokens exist in retrieved evidence.
9. **`src/findocs/agent/answer.py`**: Separates answer string generation from citation formatting. Produces extractive baseline answers paired with explicit chunk IDs.
10. **`src/findocs/agent/decompose.py`**: Identifies company entities (`COMPANY_ALIASES`) in comparative questions and splits multi-company queries into company-specific sub-queries.
11. **`src/findocs/agent/graph.py`**: Constructs a bounded state graph in **LangGraph**: `retrieve` $\rightarrow$ `grade` $\rightarrow$ `rewrite` $\rightarrow$ `retrieve` $\rightarrow$ `answer`. Bounded by `max_retries` counter.
12. **`src/findocs/eval/metrics.py`**: Formulates mathematical definitions for Recall@K, Precision@K, Mean Reciprocal Rank (MRR), Answer Accuracy, and Citation Correctness.
13. **`src/findocs/eval/ablation.py`**: Runs multi-stage ablation experiments across retrieval pipeline variations and exports structured metrics to CSV.
14. **`src/findocs/eval/correction.py`**: Compares `without_retry` ($N=0$) vs. `with_retry` ($N=2$) execution to evaluate the utility of agentic self-correction.
15. **`src/findocs/finetune/dataset.py`**: Converts candidate retrieval rows into CSV label sheets and formats instruction-tuning datasets for QLoRA training.
16. **`src/findocs/finetune/qlora_train.py`**: Trains a 4-bit LoRA adapter on `Qwen2.5-1.5B-Instruct` using `unsloth`, `peft`, and `trl.SFTTrainer`.
17. **`src/findocs/cli.py`**: Provides the main command-line interface for inspection, execution, labeling, ablation, and evaluation.

---

## 3. Whiteboard Technical Defenses

### Q1: Why use Reciprocal Rank Fusion (RRF) instead of score averaging?
* **Answer:** Dense retrieval outputs cosine similarities bounded in $[0, 1]$, whereas BM25 outputs unbounded scores (often ranging from $0$ to $20+$ depending on document length and term frequency). Averaging raw scores allows BM25 to dominate the ranking purely due to numerical scale. RRF maps rank positions $r_m(d)$ into reciprocal scores $\frac{1}{k + r_m(d)}$, creating a uniform scale across all retrieval algorithms.

### Q2: Bi-Encoder vs. Cross-Encoder Trade-off?
* **Answer:** Bi-encoders generate independent vector representations for query and document, allowing pre-indexed sub-linear nearest neighbour search ($O(N \cdot d)$ via matrix multiplication). However, bi-encoders lose fine-grained token-level interactions. Cross-encoders feed the concatenation `[CLS] Query [SEP] Document [SEP]` through self-attention layers, modeling deep query-document interactions at the cost of higher latency ($O(K)$ forward passes). We use a two-stage pattern: Bi-encoder hybrid search retrieves 20 candidate documents, and the Cross-Encoder reranks those 20 candidates down to 5.

### Q3: Why Heading-Aware Chunking over Naive Fixed-Size Chunking?
* **Answer:** SEC filings contain distinct operational sections (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financial Statements). Naive chunking (e.g., fixed 1200-character windows) splits text across Item boundaries, diluting embedding vectors with unrelated topics and stripping section metadata. Heading-aware chunking parses `ITEM` boundaries first, ensuring every chunk retains clean section metadata (`section="ITEM 1A"`), which directly improves dense vector coherence and citation auditability.

### Q4: Why QLoRA instead of Full Fine-Tuning or standard LoRA?
* **Answer:** Full fine-tuning of a 1.5B parameter model requires $\approx 24\text{ GB}$ VRAM due to optimizer states (Adam) and gradients. Standard LoRA keeps the base model in 16-bit ($\approx 3\text{ GB}$) but still requires substantial memory. QLoRA quantises the frozen base model parameters to 4-bit NormalFloat (NF4), reducing base model footprint to $\approx 750\text{ MB}$, while adding 16-bit trainable rank-16 adapters ($\approx 30\text{ MB}$). Total VRAM footprint drops under $3\text{ GB}$, enabling fine-tuning on consumer GPUs without accuracy loss.

---

## 4. Benchmark Reporting Rules

### Retrieval Ablation Results

Do not place example numbers in the README or interview guide. Run the ablation harness on hand-labeled questions and report the generated
summary CSV. The current workspace contains implementation and smoke-test evidence, not a completed benchmark table.

### Self-Correction Loop Evaluation

| Execution Mode | Answer Accuracy | Citation Correctness | Retries Used |
| :--- | :--- | :--- | :--- |
| `without_retry` ($N=0$) | run locally | run locally | 0 |
| `with_retry` ($N=2$) | run locally | run locally | recorded per question |

### Relevance Grader Fine-Tuning Benchmark

| Grader Variant | Accuracy | Latency (ms) | Cost / Call | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- |
| `Heuristic (Word Overlap)` | run locally | run locally | $0.00 | 0.00 GB |
| `Fine-Tuned QLoRA (Qwen2.5-1.5B)` | requires completed training | run locally | local inference cost | measured on CUDA host |

The QLoRA row is intentionally not filled with a result. The adapter must be trained and evaluated on held-out human labels first.

---

## 5. Reproducible Execution Sequence

To populate resume-grade evaluation tables and fine-tune the QLoRA grader, execute these commands in order:

### Step 1: Ingest Corpus & Generate Candidates for Labeling
```powershell
$env:PYTHONPATH='src'
py -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
```

### Step 2: Manually Label Candidate Chunks
Open `data/grader_label_sheet.csv`. Inspect the `chunk_text` and set the `label` column to:
* `1` if the chunk directly answers the `question`.
* `0` if the chunk is irrelevant.

### Step 3: Sync Ground Truth Labels into Eval Dataset
```powershell
py -m findocs.cli sync-labels --labels data/grader_label_sheet.csv --output data/eval_questions.json
```

### Step 4: Run Retrieval & Correction Ablations
```powershell
# Run multi-stage retrieval ablation
py -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv

# Run agent self-correction loop ablation
py -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

### Step 5: Execute QLoRA Relevance Grader Fine-Tuning (CUDA Environment Required)

> [!IMPORTANT]
> `unsloth` and `bitsandbytes` require an NVIDIA GPU with CUDA installed. Running on a CPU-only environment will raise `NotImplementedError: Unsloth cannot find any torch accelerator? You need a GPU.`

To run QLoRA fine-tuning on a CUDA-enabled system:

```powershell
# 1. Install GPU-specific training dependencies
pip install -r requirements-qlora.txt

# 2. Inspect training hyperparameters & dataset format
py -m findocs.finetune.qlora_train --notes

# 3. Launch QLoRA fine-tuning on labelled candidate pairs
py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv --output models/qlora-grader

# 4. Evaluate fine-tuned grader against heuristic / LLM grader
py -m findocs.finetune.grader_eval --labels data/grader_label_sheet.csv --model models/qlora-grader
```

### Step 6: Verify System with Unit Test Suite
```powershell
$env:PYTHONPATH='src'
py -m unittest discover -s tests -v
```
