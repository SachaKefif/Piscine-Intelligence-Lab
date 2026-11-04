"""Fine-tune the existing QMSum E5 checkpoint on reviewed train triplets.

Compare before/after on the same 237 QMSum validation questions. The test
split is never loaded. The original checkpoint is never overwritten.
"""

import argparse
import hashlib
import json
from pathlib import Path

from finetune_qmsum_e5 import PROCESSED, ROOT, evaluate, prepare_data, read_jsonl


DEFAULT_TRIPLETS = ROOT / "Data/Curated/qmsum_hard_negatives_reviewed.jsonl"
DEFAULT_CHECKPOINT = ROOT / "Models/qmsum-e5-small/checkpoints/checkpoint-842"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_reviewed_triplets(path, *, control_nonoverlap=False):
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No reviewed triplets in {path}")
    if len({row.get("id") for row in rows}) != len(rows):
        raise ValueError("Duplicate or missing triplet IDs")
    train_meetings = {
        row["meeting_id"]: row for row in read_jsonl(PROCESSED / "qmsum.jsonl")
        if row["split"] == "train"
    }
    train_passages = {
        row["passage_id"]: row for row in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl")
        if row["split"] == "train"
    }
    for row in rows:
        required = ("question", "positif", "negatif_difficile", "meeting_id",
                    "positive_id", "negative_id", "review_status")
        if any(not row.get(field) for field in required):
            raise ValueError(f"Missing field in {row.get('id')}")
        if control_nonoverlap:
            if row["review_status"] != "control_nonoverlap":
                raise ValueError(f"Not a non-overlap control: {row['id']}")
        elif row["review_status"] not in ("accepted_after_semantic_review", "accepted_assistant_review"):
            raise ValueError(f"Triplet not reviewed and accepted: {row['id']}")
        if row["meeting_id"] not in train_meetings:
            raise ValueError(f"Triplet outside QMSum train: {row['id']}")
        for passage_id, field in ((row["positive_id"], "positif"),
                                  (row["negative_id"], "negatif_difficile")):
            passage = train_passages.get(passage_id)
            if (passage is None or passage["meeting_id"] != row["meeting_id"]
                    or passage["text"] != row[field]):
                raise ValueError(f"Passage provenance mismatch: {row['id']} / {passage_id}")
        if control_nonoverlap:
            query_id = row.get("query_id")
            questions = train_meetings[row["meeting_id"]]["queries"]["specific"]
            if not isinstance(query_id, str) or not query_id.startswith(row["meeting_id"] + ":"):
                raise ValueError(f"Control query ID invalid: {row['id']}")
            question_index = int(query_id.rsplit(":", 1)[1])
            if question_index >= len(questions) or questions[question_index]["query"] != row["question"]:
                raise ValueError(f"Control question mismatch: {row['id']}")
            negative = train_passages[row["negative_id"]]
            for start, end in questions[question_index]["relevant_text_span"]:
                if negative["start_turn"] <= int(end) and int(start) <= negative["end_turn"]:
                    raise ValueError(f"Control overlaps a relevant span: {row['id']}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True,
                        help="New directory for model and metrics, preferably on a drive with >= 1 GB free")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train-max-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--control-nonoverlap", action="store_true",
                        help="Allow only deterministic random negatives outside QMSum gold spans")
    parser.add_argument("--skip-save-model", action="store_true",
                        help="Record validation metrics without serializing the experimental weights")
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2
            or not 1 <= args.mini_batch_size <= args.batch_size
            or args.eval_batch_size < 1 or args.learning_rate <= 0
            or not 16 <= args.train_max_tokens <= 510):
        parser.error("Invalid epochs, batch sizes, learning rate, or train-max-tokens")
    if args.fp16 and args.device == "cpu":
        parser.error("--fp16 requires CUDA")

    triplet_path = resolve_path(args.triplets)
    checkpoint = resolve_path(args.checkpoint)
    output = resolve_path(args.output)
    if not triplet_path.is_file():
        parser.error(
            f"Reviewed triplets missing: {triplet_path}. "
            "Finish the review queue and run review_qmsum_hard_negatives.py export, "
            "or use Data/Curated/qmsum_hard_negatives_reviewed.jsonl."
        )
    if (output.resolve() == checkpoint.resolve()
            or checkpoint.resolve() in output.resolve().parents
            or output.resolve() in checkpoint.resolve().parents):
        parser.error("Output directory must not contain or overwrite the input checkpoint")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output directory is not empty: {output}")
    if args.prepare_only:
        rows = load_reviewed_triplets(triplet_path, control_nonoverlap=args.control_nonoverlap)
        print(f"Reviewed QMSum train triplets: {len(rows)}")
        return
    if not (checkpoint / "modules.json").is_file():
        parser.error(f"Not a Sentence Transformers checkpoint: {checkpoint}")

    # Snapshot the input before loading it: review can continue while training runs.
    # Both the dataset and its reported hash then refer to exactly these bytes.
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "train_triplets.jsonl"
    snapshot_bytes = triplet_path.read_bytes()
    snapshot_path.write_bytes(snapshot_bytes)
    rows = load_reviewed_triplets(snapshot_path, control_nonoverlap=args.control_nonoverlap)
    print(f"Reviewed QMSum train triplets: {len(rows)}")

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if args.fp16 and device != "cuda":
        parser.error("--fp16 requires CUDA")
    print(f"Device: {device}")
    model = SentenceTransformer(str(checkpoint), device=device)
    model.max_seq_length = 512
    _, validation, passages, meetings = prepare_data()
    before = evaluate(model, validation, passages, meetings, args.eval_batch_size)
    print(f"QMSum validation before: {before}")

    def clip(text):
        tokens = model.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= args.train_max_tokens:
            return text
        return model.tokenizer.decode(tokens[:args.train_max_tokens], skip_special_tokens=True)

    dataset = Dataset.from_dict({
        "query": [clip("query: " + row["question"]) for row in rows],
        "positive": [clip("passage: " + row["positif"]) for row in rows],
        "negative": [clip("passage: " + row["negatif_difficile"]) for row in rows],
    })
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output / "trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=0.1,
        seed=42,
        data_seed=42,
        fp16=args.fp16,
        use_cpu=(device == "cpu"),
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        dataloader_pin_memory=False,
        save_strategy="no",
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
    after = evaluate(model, validation, passages, meetings, args.eval_batch_size)
    print(f"QMSum validation after: {after}")

    model_path = output / "model"
    metrics = {
        "input_checkpoint": str(checkpoint),
        "output_model": None,
        "triplet_file": str(triplet_path),
        "training_snapshot": str(snapshot_path),
        "triplet_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "triplet_count": len(rows),
        "negative_policy": "random_nonoverlap_control" if args.control_nonoverlap else "reviewed_hard_negative",
        "validation_questions": len(validation),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "train_max_tokens": args.train_max_tokens,
        "learning_rate": args.learning_rate,
        "before": before,
        "after": after,
        "delta": {key: round(after[key] - before[key], 4) for key in before},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Metrics: {output / 'metrics.json'}")
    if not args.skip_save_model:
        model.save(str(model_path))
        metrics["output_model"] = str(model_path)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
