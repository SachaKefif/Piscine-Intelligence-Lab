"""Optional synthetic hard-negative experiment on the existing QMSum E5 checkpoint.

The synthetic triplets are separate from QMSum. This script keeps two examples
per category for a small diagnostic holdout and evaluates QMSum validation
before and after training. It does not touch the QMSum test split or baseline.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from finetune_qmsum_e5 import evaluate, prepare_data, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRIPLETS = ROOT / "Data" / "Synthetic" / "hard_negative_triplets.jsonl"
DEFAULT_CHECKPOINT = ROOT / "Models" / "qmsum-e5-small" / "checkpoints" / "checkpoint-842"
DEFAULT_OUTPUT = ROOT / "Models" / "qmsum-e5-hardneg-synthetic"


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_triplets(path):
    rows = list(read_jsonl(path))
    required = {"id", "categorie", "question", "positif", "negatif_difficile"}
    if not rows:
        raise ValueError(f"No triplets in {path}")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Triplet IDs must be unique")
    for row in rows:
        if not required <= row.keys() or any(not row[key] for key in required):
            raise ValueError(f"Missing or empty field in triplet: {row.get('id')}")
        if row["positif"] == row["negatif_difficile"]:
            raise ValueError(f"Positive and negative are identical: {row['id']}")
    return rows


def split_by_category(rows):
    """Hold out every fifth example within each category (two of ten here)."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["categorie"]].append(row)
    train, holdout = [], []
    for category in sorted(groups):
        for index, row in enumerate(groups[category]):
            (holdout if index % 5 == 4 else train).append(row)
    if not train or not holdout:
        raise ValueError("Need train and holdout examples in the triplet file")
    return train, holdout


def diagnostic(model, rows, batch_size):
    import numpy as np

    queries = ["query: " + row["question"] for row in rows]
    positives = ["passage: " + row["positif"] for row in rows]
    negatives = ["passage: " + row["negatif_difficile"] for row in rows]
    vectors = model.encode(
        queries + positives + negatives,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    n = len(rows)
    margins = np.sum(vectors[:n] * vectors[n:2*n], axis=1) - np.sum(
        vectors[:n] * vectors[2*n:], axis=1
    )
    return {
        "accuracy": round(float(np.mean(margins > 0)), 4),
        "mean_cosine_margin": round(float(np.mean(margins)), 4),
        "count": n,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mini-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or not 1 <= args.mini_batch_size <= args.batch_size:
        parser.error("Require epochs >= 1 and 1 <= mini-batch-size <= batch-size")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")

    triplet_path = resolve_path(args.triplets)
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    if (output.resolve() == checkpoint.resolve()
            or checkpoint.resolve() in output.resolve().parents
            or output.resolve() in checkpoint.resolve().parents):
        parser.error("Output must not overwrite the input checkpoint")
    rows = load_triplets(triplet_path)
    train, holdout = split_by_category(rows)
    print(f"Synthetic triplets: {len(rows)} ({len(train)} train, {len(holdout)} holdout)")
    print(f"Categories: {dict(Counter(row['categorie'] for row in rows))}")
    if args.prepare_only:
        return
    if not (checkpoint / "modules.json").is_file():
        parser.error(f"Not a Sentence Transformers checkpoint: {checkpoint}")

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    model = SentenceTransformer(str(checkpoint))
    model.max_seq_length = 512
    _, validation, passages, meetings = prepare_data()
    before = {
        "synthetic_holdout": diagnostic(model, holdout, args.batch_size),
        "qmsum_validation": evaluate(model, validation, passages, meetings, args.batch_size),
    }
    print(f"Before: {before}")

    dataset = Dataset.from_dict({
        "query": ["query: " + row["question"] for row in train],
        "positive": ["passage: " + row["positif"] for row in train],
        "negative": ["passage: " + row["negatif_difficile"] for row in train],
    })
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        seed=42,
        data_seed=42,
        fp16=args.fp16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_pin_memory=False,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=5,
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size),
    )
    trainer.train()
    final_checkpoint = output / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
    if not (final_checkpoint / "modules.json").is_file():
        raise RuntimeError(f"Final checkpoint missing: {final_checkpoint}")
    after = {
        "synthetic_holdout": diagnostic(model, holdout, args.batch_size),
        "qmsum_validation": evaluate(model, validation, passages, meetings, args.batch_size),
    }
    print(f"After: {after}")
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "input_checkpoint": str(checkpoint),
        "output_checkpoint": str(final_checkpoint),
        "triplet_file": str(triplet_path),
        "triplet_sha256": hashlib.sha256(triplet_path.read_bytes()).hexdigest(),
        "train_ids": [row["id"] for row in train],
        "holdout_ids": [row["id"] for row in holdout],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "before": before,
        "after": after,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
