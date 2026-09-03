"""Optional QLoRA training entry point for the small relevance grader."""

import argparse

from findocs.finetune.dataset import load_labeled_pairs


DEFAULT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16
DEFAULT_LR = 2e-4
DEFAULT_EPOCHS = 2


def hyperparameter_notes() -> dict[str, str]:
    """Document the choices you must be able to defend in interviews."""

    return {
        "base_model": "1.5B parameters is small enough for an 8GB consumer GPU and strong enough for binary relevance grading.",
        "load_in_4bit": "QLoRA keeps the frozen base model in 4-bit precision to reduce memory.",
        "lora_rank": "Rank 16 is a standard first pass that adds capacity without making the adapter large.",
        "learning_rate": "2e-4 is common for LoRA adapters; lower it if validation loss is unstable.",
        "epochs": "Two epochs is enough for a few hundred labels without immediately memorising the dataset.",
    }


def rows_to_sft_examples(rows: list[dict]) -> list[dict[str, str]]:
    """Convert label-sheet rows into instruction-tuning examples."""

    examples: list[dict[str, str]] = []
    for row in rows:
        label = "relevant" if str(row["label"]).strip() == "1" else "irrelevant"
        text = (
            "### Instruction\n"
            "Decide whether the SEC filing chunk contains enough evidence to help answer the question.\n\n"
            f"### Question\n{row['question']}\n\n"
            f"### Chunk\n{row['chunk_text']}\n\n"
            f"### Answer\n{label}"
        )
        examples.append({"text": text})
    return examples


def train(label_csv: str, output_dir: str) -> None:
    """Fine-tune the grader adapter when GPU training dependencies are installed."""

    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise RuntimeError("Install GPU training dependencies with: pip install -r requirements-qlora.txt") from exc

    rows = load_labeled_pairs(label_csv)
    if not rows:
        raise ValueError("The label CSV has no completed labels. Fill label with 1 or 0 first.")
    dataset = Dataset.from_list(rows_to_sft_examples(rows))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DEFAULT_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=DEFAULT_LORA_RANK,
        lora_alpha=DEFAULT_LORA_ALPHA,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=DEFAULT_LR,
            num_train_epochs=DEFAULT_EPOCHS,
            logging_steps=5,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    """Parse training arguments and start QLoRA fine-tuning."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/grader_label_sheet.csv")
    parser.add_argument("--output", default="models/qlora-grader")
    parser.add_argument("--notes", action="store_true")
    args = parser.parse_args()
    if args.notes:
        print(hyperparameter_notes())
        return
    train(args.labels, args.output)


if __name__ == "__main__":
    main()
