"""Kaggle GPU driver: fine-tune FinBERT on the exported training set (Stage 2).

This file is GPU-bound glue only — training-set assembly (hygiene filters,
human overrides, time split) is pipeline/export_training_set.py and runs
locally; this script consumes its two CSVs. It is NOT part of the ingestion
pipeline and is never imported by it; it runs by hand in a Kaggle notebook,
same workflow as pipeline/kaggle_llama_labeling.py.

Run procedure (Kaggle):
  1. New Kaggle notebook → Settings → Accelerator = GPU (T4 x2 or P100).
  2. Upload finbert_<date>_train.csv / finbert_<date>_val.csv (from
     export_training_set) as a PRIVATE Kaggle dataset and attach it.
  3. Attach this repo (or upload pipeline/) so the imports resolve.
  4. python pipeline/kaggle_finbert_train.py \
       --train finbert_<date>_train.csv --val finbert_<date>_val.csv \
       --output-dir /kaggle/working/finbert_ft --run-date <YYYY-MM-DD>
  5. Download the output dir and publish it as a Kaggle dataset — weights are
     versioned OUTSIDE the repo (DESIGN.md); item_score.model_version records
     the model_version string from metrics.json.

Smoke test (required on first run):
  - 20-row mini train/val CSVs, --epochs 1: baseline and fine-tuned metrics
    both print, output dir contains model weights + metrics.json.
  - Baseline metrics alone should roughly reproduce across runs (eval is
    deterministic); the fine-tune itself is seeded by Trainer's default.

The pretrained checkpoint doubles as the comparison baseline: DESIGN.md
requires reporting fine-tuned vs pretrained-only FinBERT, or there is no
evidence the fine-tune helped.
"""

import argparse
import csv
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from pipeline.labeling import LABELS, build_model_meta

# calibration: baseline + init weights. Its config carries id2label
# ({0: positive, 1: negative, 2: neutral} as of transformers 4.x) — the
# label mapping below is read from the checkpoint, never hardcoded, so a
# checkpoint swap cannot silently scramble classes.
MODEL = "ProsusAI/finbert"
# calibration: GDELT rows are title-only (short); EDGAR excerpts are long —
# 256 tokens covers titles fully and truncates excerpts to their lead.
MAX_LENGTH = 256


def read_split(path: str) -> tuple[list[str], list[str]]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["text"] for r in rows], [r["label"] for r in rows]


class SplitDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label2id):
        self.enc = tokenizer(
            texts, truncation=True, padding=True, max_length=MAX_LENGTH
        )
        self.labels = [label2id[lb] for lb in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def per_class_metrics(y_true: list[int], y_pred: list[int], id2label) -> dict:
    """accuracy + per-class precision/recall/F1, no sklearn needed."""
    pairs = list(zip(y_true, y_pred, strict=True))
    out = {"accuracy": sum(t == p for t, p in pairs) / len(y_true)}
    for i, label in id2label.items():
        tp = sum(1 for t, p in pairs if t == p == i)
        prec = tp / max(1, sum(1 for p in y_pred if p == i))
        rec = tp / max(1, sum(1 for t in y_true if t == i))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        out[label] = {"precision": prec, "recall": rec, "f1": f1}
    return out


class WeightedTrainer(Trainer):
    """Trainer with inverse-frequency class weights — the corpus is ~90%
    neutral and the distress index cares most about the rare classes."""

    def __init__(self, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    # calibration: compute_loss signature varies by transformers version
    # (num_items_in_batch arrived in 4.46); **kwargs absorbs the drift.
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def evaluate(trainer: Trainer, dataset: SplitDataset, id2label) -> dict:
    pred = trainer.predict(dataset)
    y_pred = pred.predictions.argmax(axis=-1).tolist()
    return per_class_metrics(dataset.labels, y_pred, id2label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="finbert_<date>_train.csv")
    ap.add_argument("--val", required=True, help="finbert_<date>_val.csv")
    ap.add_argument(
        "--holdout",
        help="finbert_<date>_holdout.csv — human-labeled rows kept out of "
        "training. Report this separately from --val: val's labels are the "
        "labeler's, so val measures agreement with Llama, not accuracy.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-date", required=True, help="YYYY-MM-DD")
    # calibration: 3 epochs / 2e-5 are the standard BERT fine-tune defaults;
    # revisit only if the val metrics say so.
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    id2label = {int(i): lb for i, lb in model.config.id2label.items()}
    assert sorted(id2label.values()) == sorted(LABELS), id2label
    label2id = {lb: i for i, lb in id2label.items()}

    train_texts, train_labels = read_split(args.train)
    val_texts, val_labels = read_split(args.val)
    train_ds = SplitDataset(train_texts, train_labels, tokenizer, label2id)
    val_ds = SplitDataset(val_texts, val_labels, tokenizer, label2id)

    counts = torch.bincount(torch.tensor(train_ds.labels), minlength=len(LABELS))
    class_weights = len(train_ds) / (len(LABELS) * counts.clamp(min=1)).float()

    train_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        save_strategy="no",
        report_to="none",
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=train_args,
        train_dataset=train_ds,
    )

    hold_ds = None
    if args.holdout:
        hold_texts, hold_labels = read_split(args.holdout)
        hold_ds = SplitDataset(hold_texts, hold_labels, tokenizer, label2id)

    baseline = evaluate(trainer, val_ds, id2label)  # before any training
    baseline_holdout = evaluate(trainer, hold_ds, id2label) if hold_ds else None
    trainer.train()
    fine_tuned = evaluate(trainer, val_ds, id2label)
    fine_tuned_holdout = evaluate(trainer, hold_ds, id2label) if hold_ds else None

    model_version = f"finbert-ft-{args.run_date}"
    metrics = {
        "model_version": model_version,
        "model_meta": build_model_meta(MODEL, "none", "ft-v1", args.run_date),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "class_weights": class_weights.tolist(),
        "baseline_pretrained": baseline,
        "fine_tuned": fine_tuned,
        "holdout_size": len(hold_ds) if hold_ds else 0,
        "baseline_pretrained_holdout": baseline_holdout,
        "fine_tuned_holdout": fine_tuned_holdout,
    }
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    def report(title: str, before: dict, after: dict, n: int) -> None:
        print(f"\n{title}  (n={n})")
        print(f"{'':10} {'baseline':>10} {'fine-tuned':>10}")
        print(f"{'accuracy':10} {before['accuracy']:>10.3f} {after['accuracy']:>10.3f}")
        for lb in LABELS:
            print(f"f1 {lb:7} {before[lb]['f1']:>10.3f} {after[lb]['f1']:>10.3f}")

    print(f"\n{model_version}")
    report(
        "val — labels are the LABELER's, not truth", baseline, fine_tuned, len(val_ds)
    )
    if hold_ds:
        report(
            "holdout — human labels, never trained on",
            baseline_holdout,
            fine_tuned_holdout,
            len(hold_ds),
        )
    else:
        print("\nno --holdout given: nothing here measures human-truth accuracy")
    print(f"\nweights + metrics.json -> {args.output_dir}")


if __name__ == "__main__":
    main()
