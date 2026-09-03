"""
qlora_train.py — QLoRA Relevance Grader Fine-Tuning
=====================================================
THIS IS THE MOST IMPORTANT FILE IN THE PROJECT FOR INTERVIEWS.
-------------------------------------------------------------
Every other file implements standard RAG patterns. This file implements
something most candidates NEVER do: actually fine-tune a model, measure the
result, and make an engineering decision based on the data.

WHAT THIS SCRIPT DOES:
Trains a small language model (Qwen2.5-1.5B) using QLoRA to perform binary
relevance classification on (question, chunk) pairs from SEC filings.

AFTER TRAINING, THE MODEL CAN DO THIS:
Input:  "Question: What was R&D spending?  Chunk: Apple spent $29.9B on..."
Output: "relevant"

Input:  "Question: What was R&D spending?  Chunk: Apple's iPhone revenue..."
Output: "irrelevant"

This replaces the word-overlap heuristic in graph.py's grade() node with a
trained model that understands financial-domain semantics.

─────────────────────────────────────────────────────────────────────────────
THEORY: WHY QLORA AND NOT FULL FINE-TUNING?
─────────────────────────────────────────────────────────────────────────────
Full Fine-Tuning:
    Updates EVERY weight in the model.
    Qwen2.5-1.5B has 1.5 billion parameters × 4 bytes = 6 GB just for weights.
    Add gradients (6 GB) + optimizer state (12 GB for Adam) = ~24 GB total.
    A 4070's 8 GB VRAM cannot hold this. Full fine-tuning is impossible here.

LoRA (Low-Rank Adaptation):
    Instead of updating all weights, inject small adapter matrices into the
    attention layers. These adapters have rank=16 — each is a (d × 16) and
    (16 × d) matrix where d is the attention dimension (~2048 for this model).
    Only ~0.5% of the parameters are trained. Everything else is frozen.
    Memory: adapter params only (~30 MB). The frozen base model still takes 6 GB.
    Problem: base model still too large for the GPU.

QLoRA = LoRA + 4-bit Quantization:
    Load the FROZEN base model in 4-bit precision (NF4 format, 4 bits per weight
    instead of 32 bits). 1.5B params × 0.5 bytes = ~750 MB for the base model.
    Add LoRA adapters (~30 MB) + gradients for adapters only + optimizer state
    for adapters only → total ~3 GB. Fits comfortably in 8 GB VRAM.

DETTMERS ET AL. 2023 (THE QLORA PAPER):
    Showed that QLoRA fine-tuning achieves within a few percentage points of
    full fine-tuning accuracy while cutting memory by 10-15x. The key insight:
    4-bit quantization of the FROZEN base model (not the adapters themselves)
    introduces minimal quality loss because the adapters can compensate during
    training.

─────────────────────────────────────────────────────────────────────────────
HYPERPARAMETER DECISIONS (YOU MUST DEFEND THESE):
─────────────────────────────────────────────────────────────────────────────
base_model = Qwen2.5-1.5B-Instruct-bnb-4bit
    WHY QWEN? Strong instruction-following at 1.5B parameters. Already
    quantized in the Unsloth hub format — faster loading than converting yourself.
    WHY 1.5B? Smallest model that reliably follows instruction format.
    Smaller (0.5B) tends to fail at format adherence; larger (3B) needs more VRAM.

lora_rank = 16
    WHY 16? Standard starting value from the LoRA paper (Hu et al., 2021).
    Rank controls the adapter's "capacity": rank 16 means each adapter has
    d×16 + 16×d parameters. Too low → underfitting. Too high → overfitting
    on a small dataset + more memory. 16 is the safe first pass.

lora_alpha = 16 (= lora_rank)
    WHY EQUAL TO RANK? Alpha scales the adapter output: final = base + (alpha/rank) × adapter.
    Setting alpha = rank means the scale factor is 1.0 — adapters are not
    over-amplified. Some practitioners use alpha = 2 × rank; test both if you have time.

learning_rate = 2e-4
    WHY 2e-4? Standard for LoRA adapters. The base model is frozen and doesn't
    need careful LR scheduling — only the tiny adapters are updated.
    If validation loss is unstable, lower to 1e-4.

num_train_epochs = 2
    WHY 2? With ~100 labeled pairs (from labeling AAPL), 2 epochs gives the model
    ~200 gradient steps. More epochs risk overfitting (memorising the labels
    rather than learning the pattern). Check validation loss after epoch 1.

target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    WHY THESE? These are the projection matrices in the attention mechanism and
    the feed-forward network (MLP). The LoRA paper found that targeting all
    attention projections + MLP gives the best results. Targeting only q_proj
    and v_proj (the original LoRA paper's default) saves more memory but gives
    lower quality.

─────────────────────────────────────────────────────────────────────────────
WHY TRAINING DEPS ARE ISOLATED (requirements-qlora.txt):
─────────────────────────────────────────────────────────────────────────────
unsloth/bitsandbytes/trl require CUDA. On a machine without a GPU (CI server,
a colleague's MacBook, your dev environment during debugging), importing these
libraries would fail immediately. The training script imports them INSIDE the
train() function — so the rest of the project (cli.py, tests, retrieval) can
import and run without any GPU packages installed.
"""

import argparse

from findocs.finetune.dataset import load_labeled_pairs


# ── Constants ─────────────────────────────────────────────────────────────────
# All hyperparameters are module-level constants so they're easy to find,
# modify, and reference when explaining your choices in an interview.

DEFAULT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16          # Equal to rank → scale factor 1.0
DEFAULT_LR = 2e-4                # Standard LoRA learning rate
DEFAULT_EPOCHS = 2               # Enough for a few hundred labels without overfitting


def hyperparameter_notes() -> dict[str, str]:
    """
    Document the choices you must be able to defend in interviews.

    Run with: py -m findocs.finetune.qlora_train --notes
    This prints a dict you can read as your interview prep cheat sheet.
    """

    return {
        "base_model": (
            "Qwen2.5-1.5B-Instruct: 1.5B parameters fits an 8GB GPU in QLoRA 4-bit mode. "
            "Already quantized in Unsloth format for faster loading."
        ),
        "load_in_4bit": (
            "QLoRA: frozen base model loaded in 4-bit NF4 precision. "
            "~750MB instead of 6GB. Only adapter weights are trained in full precision."
        ),
        "lora_rank": (
            "Rank 16 is the standard starting point from Hu et al. 2021. "
            "Represents adapter capacity — too low underfits, too high overfits on small datasets."
        ),
        "lora_alpha": (
            "Equal to rank (16) → scale factor alpha/rank = 1.0. "
            "Adapters are not amplified or dampened — safe default."
        ),
        "learning_rate": (
            "2e-4 is the standard LoRA learning rate. "
            "Only the adapter weights are updated, so higher LR than full fine-tuning is safe."
        ),
        "epochs": (
            "2 epochs for ~100 labeled examples = ~200 gradient steps. "
            "Watch validation loss — if it starts rising at epoch 2, stop early."
        ),
        "target_modules": (
            "All attention projections (q/k/v/o) and MLP projections (gate/up/down). "
            "More coverage than original LoRA (q/v only) → better quality, more memory."
        ),
    }


def rows_to_sft_examples(rows: list[dict]) -> list[dict[str, str]]:
    """
    Convert label-sheet rows into instruction-tuning examples.

    WHAT IS SFT (SUPERVISED FINE-TUNING)?
    We format each labeled pair as a complete instruction → answer sequence.
    The model learns to predict the answer given the instruction. This is
    the same training paradigm used to create instruction-following models.

    THE QWEN2.5 INSTRUCTION FORMAT:
    Qwen2.5 was trained with specific header tokens (### Instruction, ### Answer).
    Using the same format during fine-tuning ensures the base model's instruction-
    following capability transfers to our relevance classification task.

    LABEL MAPPING:
    label == "1" → "relevant"    (the chunk helps answer the question)
    label == "0" → "irrelevant"  (the chunk doesn't help)

    The model learns to output exactly one word: "relevant" or "irrelevant".
    This binary output is easy to parse in the grader inference function.

    EXAMPLE OUTPUT:
    {
        "text": "### Instruction\\nDecide whether...\\n\\n### Question\\n...\\n\\n### Chunk\\n...\\n\\n### Answer\\nrelevant"
    }
    The SFTTrainer expects a dataset with a "text" field containing the full
    formatted training example.
    """

    examples: list[dict[str, str]] = []
    for row in rows:
        # Convert numeric label to natural language so the model learns English semantics
        label = "relevant" if str(row["label"]).strip() == "1" else "irrelevant"
        text = (
            "### Instruction\n"
            "Decide whether the SEC filing chunk contains enough evidence "
            "to help answer the question.\n\n"
            f"### Question\n{row['question']}\n\n"
            f"### Chunk\n{row['chunk_text']}\n\n"
            f"### Answer\n{label}"
        )
        examples.append({"text": text})
    return examples


def train(label_csv: str, output_dir: str) -> None:
    """
    Fine-tune the grader adapter when GPU training dependencies are installed.

    IMPORTS ARE INSIDE THIS FUNCTION INTENTIONALLY:
    If they were at module level, importing cli.py or any other module would
    fail on machines without CUDA/Unsloth. The lazy import means:
    - Normal usage (smoke test, retrieval, eval) never touches GPU imports.
    - Training only imports GPU packages when train() is explicitly called.

    WHAT UNSLOTH PROVIDES:
    FastLanguageModel is Unsloth's drop-in replacement for HuggingFace's
    AutoModelForCausalLM. It patches PyTorch operations for faster QLoRA
    fine-tuning (2-4x faster than vanilla HuggingFace QLoRA). Same API.

    TRAINING FLOW:
    1. Load labeled pairs from CSV (only completed 0/1 labels).
    2. Convert to SFT examples (instruction → answer format).
    3. Load Qwen2.5-1.5B in 4-bit mode with FastLanguageModel.
    4. Add LoRA adapters to the attention and MLP projections.
    5. Train with SFTTrainer from trl (Transformer Reinforcement Learning library).
    6. Save the adapter weights (NOT the full model — adapters are ~30MB).

    AFTER TRAINING:
    The models/ directory contains the LoRA adapter.
    To use it for inference: load base model → load adapter → merge or use PEFT.

    BATCH SIZE AND GRADIENT ACCUMULATION:
    per_device_train_batch_size=1: one example per GPU step (8GB VRAM limit).
    gradient_accumulation_steps=8: accumulate gradients for 8 steps before updating.
    Effective batch size = 1 × 8 = 8. This simulates a batch of 8 without needing
    8× the VRAM. Standard technique for QLoRA training on consumer GPUs.
    """

    try:
        import unsloth
        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError as exc:
        raise RuntimeError(
            "Install GPU training dependencies with: pip install -r requirements-qlora.txt"
        ) from exc

    rows = load_labeled_pairs(label_csv)
    if not rows:
        raise ValueError(
            "The label CSV has no completed labels. "
            "Fill the 'label' column with 1 or 0 before training."
        )

    # Convert to HuggingFace Dataset format
    dataset = Dataset.from_list(rows_to_sft_examples(rows))

    # Load base model in 4-bit + set up LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL,
        max_seq_length=2048,   # Max tokens per training example
        load_in_4bit=True,     # QLoRA: 4-bit quantisation of frozen base model
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=0,        # 0 dropout is recommended by Unsloth for QLoRA
        # Adapter targets: all attention + MLP projections
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
    )

    # SFTTrainer: supervised fine-tuning from trl
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",         # Column with full training examples
            output_dir=output_dir,             # Where checkpoints are saved
            per_device_train_batch_size=1,     # 1 example per GPU step
            gradient_accumulation_steps=8,     # Effective batch size = 8
            learning_rate=DEFAULT_LR,
            num_train_epochs=DEFAULT_EPOCHS,
            logging_steps=5,                   # Log loss every 5 steps
            save_strategy="epoch",             # Save checkpoint after each epoch
            eos_token=tokenizer.eos_token,     # Match tokenizer EOS token
        ),
    )

    trainer.train()

    # Save adapter weights (not the full model — adapters are ~30MB)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """
    Parse training arguments and start QLoRA fine-tuning.

    USAGE:
        # See hyperparameter explanations without training:
        py -m findocs.finetune.qlora_train --notes

        # Train on completed label sheet:
        py -m findocs.finetune.qlora_train --labels data/grader_label_sheet.csv --output models/qlora-grader
    """

    parser = argparse.ArgumentParser(description="QLoRA relevance grader training")
    parser.add_argument(
        "--labels",
        default="data/grader_label_sheet.csv",
        help="Path to the completed label sheet CSV",
    )
    parser.add_argument(
        "--output",
        default="models/qlora-grader",
        help="Directory to save the trained LoRA adapter",
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="Print hyperparameter explanations instead of training",
    )
    args = parser.parse_args()

    if args.notes:
        import json
        print(json.dumps(hyperparameter_notes(), indent=2))
        return

    train(args.labels, args.output)


if __name__ == "__main__":
    main()
