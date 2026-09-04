# FinDocs Agent: Finishing the Blocked Work

This guide is for the remaining pieces that cannot be completed honestly without your data preparation and experiment runs:

1. Confirming that the correct virtual environment uses CUDA PyTorch and the RTX 4070.
2. Downloading/caching the embedding and reranker models so retrieval ablations can run reliably.
3. Completing human labels, additional company filings, and the full evaluation tables.
4. Training and evaluating the QLoRA relevance grader.

The GPU prerequisite is now verified in this project: the `.venv` reports PyTorch `2.11.0+cu128`, CUDA `12.8`, and an NVIDIA GeForce RTX
4070 Laptop GPU. `torch.cuda.is_available() == True` is still only a prerequisite, not a result. A result exists only after a successful
training/evaluation run produces an adapter and measured metrics.

Official references: [PyTorch Start Locally](https://pytorch.org/get-started/locally/), [Hugging Face installation and offline use](https://huggingface.co/docs/transformers/installation), [Hugging Face cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache), and [Unsloth installation](https://docs.unsloth.ai/get-started/installing-%2B-updating/pip-install).

## 0. Always identify the interpreter first

Open PowerShell in `C:\Users\sudha\findocs`. Do not assume that an activated prompt means the commands are using that environment.

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip --version
where.exe python
```

The first two outputs must point into the intended `.venv` directory. Use `python -m pip`, not a bare `pip` and preferably not `py -m pip`,
for the rest of this guide. The `py` launcher can select a different Python installation than the one you just activated.

If your environment has a different name, activate that environment and use its path consistently. You do not need to delete anything
until you have checked which interpreter is being used.

## 1. Diagnose the GPU before changing packages

Run these commands inside the activated environment:

```powershell
nvidia-smi
python -c "import torch; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Interpret the output as follows:

| Observation | Meaning | Action |
|---|---|---|
| `nvidia-smi` is not recognized | NVIDIA driver is missing or not on PATH | Install/update the official NVIDIA driver, reboot, retry |
| `nvidia-smi` shows RTX 4070 but `torch.version.cuda` is `None` | CPU-only PyTorch is installed | Replace PyTorch with a CUDA wheel |
| `torch.version.cuda` has a value but availability is `False` | Driver/PyTorch/environment mismatch | Check driver, restart the shell, then reinstall the supported CUDA wheel |
| CUDA available and GPU name is RTX 4070 | PyTorch can use the GPU | Continue to model and QLoRA checks |

The PyTorch wheel supplies the CUDA runtime components it needs; the NVIDIA driver still has to be installed and working. Select Windows,
Pip, Python, and a CUDA platform from the official PyTorch installer page rather than copying an old command from a random tutorial.

## 2. Repair a CPU-only PyTorch installation safely

Only run this inside the dedicated project venv. The commands remove packages from that venv, not from Windows globally.

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip cache purge
```

Now open [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select:

```text
OS: Windows
Package: Pip
Language: Python
Compute Platform: the current CUDA option appropriate for your system
```

Run the exact generated command in the activated venv. Do not append `+cu...` manually and do not install a CPU wheel afterward with a
second requirements command.

Then install the normal FinDocs dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify again:

```powershell
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is still unavailable'; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

A useful functional test is an actual GPU tensor operation:

```powershell
python -c "import torch; x=torch.randn((4096,4096), device='cuda'); y=x @ x; torch.cuda.synchronize(); print(y.device, round(torch.cuda.max_memory_allocated()/1024**3, 2), 'GiB peak')"
```

If this fails, do not proceed to QLoRA. Fix CUDA first.

## 3. Cache the Hugging Face models

FinDocs needs the embedding model for normal retrieval and a cross-encoder for reranking. A model can be cached once while online and
then loaded offline. Hugging Face documents `HF_HUB_CACHE` as the cache location and `HF_HUB_OFFLINE=1` as the switch that prevents HTTP
requests.

Choose a stable local cache path if desired:

```powershell
$env:HF_HUB_CACHE = "C:\Users\sudha\findocs\.hf-cache"
```

With network access available, download the exact models used by the code:

```powershell
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); print('retrieval models cached')"
```

For the QLoRA base model, cache it from the CUDA environment after the training dependencies are installed. Unsloth may download the
base model on the first training command; allow that first run to finish completely.

Inspect the cache when debugging:

```powershell
hf cache ls
```

Only after the models load successfully once should you test offline mode:

```powershell
$env:HF_HUB_OFFLINE = "1"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('offline embedding load passed')"
Remove-Item Env:HF_HUB_OFFLINE
```

If the offline command fails, the cache is incomplete or the cache variable points to a different directory. Remove the offline variable,
download again, and retry. Do not repeatedly run the full evaluator while the model is trying to download.

## 4. Finish the human ground truth

The candidate sheet is not ground truth until every candidate row has a label. Start from the existing file:

```powershell
$env:PYTHONPATH = "src"
python -m findocs.cli label-candidates --companies AAPL --output data/grader_label_sheet.csv
python -m findocs.cli show-candidates --labels data/grader_label_sheet.csv --question-id q001
```

For each question, inspect all candidate text and enter exactly:

```text
1 = the chunk contains direct evidence that materially helps answer the question
0 = it does not
```

Do not label table-of-contents references as evidence. For exact-number questions, require the actual value or the directly relevant
financial table/discussion. For broad questions, multiple chunks may legitimately be positive.

Check the sheet before syncing:

```powershell
python -c "import csv; rows=list(csv.DictReader(open('data/grader_label_sheet.csv', encoding='utf-8'))); print('rows:',len(rows)); print('unlabeled:',sum(not r['label'].strip() for r in rows)); print('positives:',sum(r['label'].strip()=='1' for r in rows)); print('negatives:',sum(r['label'].strip()=='0' for r in rows))"
```

You want zero unlabeled rows and both positive and negative classes. Then sync the relevant chunk IDs:

```powershell
python -m findocs.cli sync-labels --labels data/grader_label_sheet.csv --output data/eval_questions.json
```

Read `data/eval_questions.json` afterward. Check that every question intended for the current corpus has at least one relevant ID.
Questions for companies not yet downloaded must remain out of the single-company benchmark.

## 5. Add companies correctly

For a meaningful comparative experiment, download at least NVIDIA and Microsoft in addition to Apple. SEC requires a real identifying email
in the User-Agent:

```powershell
python -m findocs.cli run --company NVDA --email your.real.email@example.com
python -m findocs.cli run --company MSFT --email your.real.email@example.com
```

Use your real email address. Do not commit it to source code or `.env` files. Confirm local artifacts exist before labeling:

```powershell
Get-ChildItem data | Select-Object Name, Length
```

Regenerate candidates for the multi-company questions after both filings are available:

```powershell
python -m findocs.cli label-candidates --companies AAPL,NVDA,MSFT --output data/grader_label_sheet.csv
```

Label company-specific evidence separately. A chunk from Apple cannot be a positive answer for an NVIDIA or Microsoft sub-question.

## 6. Run the retrieval ablations

First run without reranking:

```powershell
python -m findocs.cli eval-retrieval --companies AAPL --output results/retrieval_ablation.csv
```

Then run with reranking to a separate file:

```powershell
python -m findocs.cli eval-retrieval --companies AAPL --rerank --output results/retrieval_ablation_reranked.csv
```

The four stages are:

```text
dense_only
bm25_only
dense_bm25_rrf
hybrid_reranker
```

Report Recall@5, Recall@10, MRR, and Precision@5 from the generated summary CSVs. Add answer accuracy separately from the correction
evaluation; retrieval rank metrics and answer correctness are not interchangeable.

If the reranked command tries to reach Hugging Face, cache the cross-encoder as described above. If it fails with a CUDA warning but the
model can run on CPU, it may still work slowly; the warning is not automatically a correctness failure. If the model cannot load, record
that as an environment limitation rather than replacing the reranker with invented numbers.

## 7. Run the self-correction evaluation

```powershell
python -m findocs.cli eval-correction --companies AAPL --output results/self_correction.csv
```

Inspect both:

```powershell
Get-Content results/self_correction_summary.csv
Get-Content results/self_correction.csv
```

The per-question file tells you whether a retry actually happened through `retries_used`. A result where every question uses zero retries
does not demonstrate recovery; it only shows that the current heuristic graded the first retrieval as good enough.

Before reporting answer accuracy, review `accepted_answer_phrases` in `data/eval_questions.json`. A phrase that is too narrow can make a
correct extractive answer score zero; a phrase that is too broad can inflate the score.

## 8. Install and validate the QLoRA environment

Use a clean environment for training. It avoids the global-package conflicts seen in the normal development interpreter.

```powershell
py -m venv .venv-qlora
.\.venv-qlora\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-qlora.txt
$env:PYTHONPATH = "src"
```

On Windows, follow the current Unsloth installation instructions if the pinned requirements produce an incompatibility. Unsloth’s current
guidance says PyTorch should be installed before `pip install unsloth`; do not install the QLoRA stack into the CPU-only environment.

Verify all prerequisites together:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
python -c "import unsloth, bitsandbytes, trl, peft; print('QLoRA packages import successfully')"
python -m findocs.finetune.qlora_train --notes
```

Stop if CUDA is false or if the import test fails.

## 9. Train the adapter

Confirm labels first:

```powershell
python -c "from findocs.finetune.dataset import load_labeled_pairs; rows=load_labeled_pairs('data/grader_label_sheet.csv'); print('completed labels:',len(rows)); print('classes:',sorted(set(r['label'] for r in rows)))"
```

Then train:

```powershell
python -m findocs.finetune.qlora_train `
  --labels data/grader_label_sheet.csv `
  --output models/qlora-grader
```

Record these details in your experiment notes:

- exact Python, PyTorch, Unsloth, Transformers, TRL, PEFT, and bitsandbytes versions;
- GPU model and driver version;
- number of labeled rows and positive/negative balance;
- train/validation split policy;
- base model and 4-bit setting;
- LoRA rank, alpha, target modules, learning rate, epochs, batch size, and sequence length;
- training loss and wall-clock time;
- peak GPU memory;
- adapter output directory.

Important evaluation warning: the current training entry point consumes every completed row in the CSV. For a defensible generalization
claim, create a held-out CSV by question or filing before training and evaluate on rows the adapter never saw. Do not report training-set
accuracy as model quality.

### Interpreting your first run

Your first completed run used 100 rows, trained for 2 epochs and 26 steps, and finished successfully on the RTX 4070. The subsequent
evaluation reported approximately:

```text
QLoRA accuracy:       0.90
QLoRA precision:      0.00
QLoRA recall:         0.00
QLoRA F1:             0.00
QLoRA latency:        463.4 ms/call
QLoRA peak memory:    1.325 GiB
```

This is a valid measured result, but it is a failed first grader, not a successful 90%-accurate grader. The label distribution was 90
negative and 10 positive, and the zero positive-class metrics indicate that the model predicted the majority class. Accuracy alone hid
that failure.

Correct it as follows:

1. Add more candidate questions and filings, especially direct positive evidence examples.
2. Aim for a substantially less skewed dataset; do not manufacture positives by copying the same rows.
3. Split by question or filing into train and held-out validation data before training.
4. Retrain with the same configuration as a controlled first comparison.
5. Require nonzero positive precision, recall, and F1, then inspect false positives and false negatives.
6. Only integrate the adapter into the graph if held-out performance is useful compared with the heuristic and reference grader.

The honest current conclusion is: “The first QLoRA run fit on the GPU and reduced the engineering risk of the training path, but the
imbalanced dataset produced a majority-class classifier. I expanded and rebalanced the labels before making a quality claim.”

## 10. Evaluate the grader honestly

Run the existing evaluator after an adapter exists:

```powershell
python -m findocs.finetune.grader_eval `
  --labels data/grader_label_sheet.csv `
  --model models/qlora-grader `
  --output results/grader_eval.json
```

The command also saves the measured JSON results, making the comparison auditable instead of leaving it only in terminal output.

For the final comparison, use the same held-out rows for:

| Measurement | Heuristic | Reference LLM | QLoRA |
|---|---:|---:|---:|
| Accuracy | measure | measure | measure |
| Precision | measure | measure | measure |
| Recall | measure | measure | measure |
| F1 | measure | measure | measure |
| Mean latency/call | measure | measure | measure |
| p95 latency/call | measure | measure | measure |
| Cost/call | local | configure | local |
| Peak GPU memory | measure | measure | measure |

The current CLI contains a heuristic baseline and QLoRA evaluator. A genuine “large LLM versus QLoRA” claim additionally requires a real
reference-LLM prediction function, not the heuristic renamed as an LLM. Report agreement with the reference grader and agreement with human
labels separately.

## 11. Final acceptance checklist

The blocked work is complete when all of the following are true:

- `torch.cuda.is_available()` is true in the same environment used for training.
- The GPU tensor test succeeds.
- Embedding and reranker models load from cache.
- All intended evaluation questions have reviewed relevant chunk IDs.
- At least two companies have local filings for comparative questions.
- Unreranked and reranked ablation CSVs exist.
- Self-correction CSV and summary exist, with retries-used traces inspected.
- A held-out grader split exists and contains both labels.
- `models/qlora-grader` contains a saved adapter produced by a successful run.
- QLoRA metrics, latency, memory, and configuration are recorded from the actual run.
- Every resume number can be traced to a result file and rerun command.
- `python -m unittest discover -s tests -v` still passes after any integration change.

## What to say if something still fails

Say exactly what failed and why:

> “The retrieval and correction pipeline is implemented and tested. The reranker could not be benchmarked because the model cache was
> unavailable. The QLoRA path is implemented, but I did not claim a training result until CUDA PyTorch, completed human labels, and a
> held-out evaluation split were available.”

That is a stronger engineering answer than quoting a number you cannot reproduce.
