"""Fine-tune multilingual-e5-small for QMSum question-to-passage retrieval.

Uses only QMSum train meetings for optimization and validation meetings for
before/after retrieval evaluation. Run --prepare-only to inspect pairs without
downloading a model or installing training dependencies.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
MODEL_NAME = "intfloat/multilingual-e5-small"


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run the data processing pipeline first.")
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{number}") from exc


def relevant_passages(passages, spans):
    intervals = [(int(start), int(end)) for start, end in spans]
    return [
        passage for passage in passages
        if any(
            passage["start_turn"] <= end and passage["end_turn"] >= start
            for start, end in intervals
        )
    ]


def prepare_data():
    meetings = {"train": {}, "validation": {}}
    for meeting in read_jsonl(PROCESSED / "qmsum.jsonl"):
        split = meeting["split"]
        if split in meetings:
            meeting_id = meeting["meeting_id"]
            if meeting_id in meetings[split]:
                raise ValueError(f"Duplicate meeting: {meeting_id}")
            meetings[split][meeting_id] = meeting

    passages = defaultdict(list)
    for passage in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl"):
        if passage["split"] in meetings and passage["text"].strip():
            passages[passage["meeting_id"]].append(passage)

    pairs = {"query": [], "positive": []}
    validation = []
    for split in ("train", "validation"):
        for meeting_id, meeting in meetings[split].items():
            candidates = passages[meeting_id]
            if not candidates:
                raise ValueError(f"No passages for {meeting_id}")
            for number, question in enumerate(meeting["queries"]["specific"]):
                matches = relevant_passages(candidates, question["relevant_text_span"])
                if not matches:
                    raise ValueError(f"No relevant passage for {meeting_id}:{number}")
                if split == "train":
                    # Every overlapping passage is a positive. The sampler keeps
                    # repeated queries and passages out of the same training batch.
                    for passage in matches:
                        pairs["query"].append("query: " + question["query"])
                        pairs["positive"].append("passage: " + passage["text"])
                else:
                    validation.append({
                        "meeting_id": meeting_id,
                        "query": question["query"],
                        "gold_ids": {p["passage_id"] for p in matches},
                    })

    if not pairs["query"] or not validation:
        raise ValueError("Expected both training pairs and validation questions")
    return pairs, validation, passages, meetings


def evaluate(model, validation, passages, meetings, batch_size):
    import numpy as np

    questions = ["query: " + row["query"] for row in validation]
    query_vectors = model.encode(
        questions, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    passage_vectors = {}
    for meeting_id in meetings["validation"]:
        texts = ["passage: " + p["text"] for p in passages[meeting_id]]
        passage_vectors[meeting_id] = model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )

    totals = {"hit_at_5": 0.0, "recall_at_5": 0.0, "mrr_at_5": 0.0}
    for row, query_vector in zip(validation, query_vectors):
        meeting_id = row["meeting_id"]
        candidates = passages[meeting_id]
        scores = passage_vectors[meeting_id] @ query_vector
        top = np.argsort(-scores)[:5]
        hits = [candidates[int(i)]["passage_id"] in row["gold_ids"] for i in top]
        totals["hit_at_5"] += float(any(hits))
        totals["recall_at_5"] += sum(hits) / len(row["gold_ids"])
        totals["mrr_at_5"] += next((1 / rank for rank, hit in enumerate(hits, 1) if hit), 0.0)
    return {key: round(value / len(validation), 4) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("Models/qmsum-e5-small"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=2,
                        help="Pairs processed at once by the cached contrastive loss")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true", help="Use FP16 on a compatible GPU")
    parser.add_argument("--prepare-only", action="store_true", help="Check data without training")
    parser.add_argument("--evaluate-checkpoint", type=Path,
                        help="Evaluate a completed checkpoint without retraining")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        parser.error("epochs must be >= 1, batch size >= 2, and learning rate > 0")
    if not 1 <= args.mini_batch_size <= args.batch_size:
        parser.error("mini-batch-size must be between 1 and batch-size")
    if not 1 <= args.max_seq_length <= 512:
        parser.error("max sequence length must be between 1 and 512")

    pairs, validation, passages, meetings = prepare_data()
    print(f"Training pairs: {len(pairs['query'])} from {len(meetings['train'])} meetings")
    print(f"Validation questions: {len(validation)} from {len(meetings['validation'])} meetings")
    print("Gold passages use turn overlap; these relevance labels are approximate.")
    if args.prepare_only:
        return

    import torch
    from sentence_transformers import SentenceTransformer

    if torch.cuda.is_available():
        print(f"Training device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Training device: CPU (PyTorch {torch.__version__}, CUDA build: {torch.version.cuda})")
        if torch.version.cuda is None:
            print("This PyTorch installation cannot use an NVIDIA GPU.")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.evaluate_checkpoint:
        checkpoint = (args.evaluate_checkpoint if args.evaluate_checkpoint.is_absolute()
                      else ROOT / args.evaluate_checkpoint)
        if not (checkpoint / "modules.json").is_file():
            parser.error(f"Not a complete Sentence Transformers checkpoint: {checkpoint}")
        model = SentenceTransformer(str(checkpoint))
    else:
        model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = args.max_seq_length
    print("Evaluating pretrained model...")
    if args.evaluate_checkpoint:
        pretrained = SentenceTransformer(MODEL_NAME)
        pretrained.max_seq_length = args.max_seq_length
        before = evaluate(pretrained, validation, passages, meetings, args.batch_size)
        del pretrained
    else:
        before = evaluate(model, validation, passages, meetings, args.batch_size)
    print(f"Pretrained: {before}")

    if args.evaluate_checkpoint:
        final_model_path = checkpoint
    else:
        from datasets import Dataset
        from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
        from sentence_transformers.sentence_transformer.losses import CachedMultipleNegativesRankingLoss
        from sentence_transformers.sentence_transformer.training_args import BatchSamplers

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=0.1,
            fp16=args.fp16,
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            dataloader_pin_memory=False,
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=25,
            report_to="none",
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=Dataset.from_dict(pairs),
            loss=CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size),
        )
        trainer.train()
        final_model_path = output / "checkpoints" / f"checkpoint-{trainer.state.global_step}"
        if not (final_model_path / "modules.json").is_file():
            raise RuntimeError(f"Final checkpoint missing: {final_model_path}")

    print("Evaluating fine-tuned model...")
    after = evaluate(model, validation, passages, meetings, args.batch_size)
    print(f"Fine-tuned: {after}")
    output.mkdir(parents=True, exist_ok=True)
    metrics = {"model": MODEL_NAME, "model_path": str(final_model_path),
               "training_pairs": len(pairs["query"]),
               "validation_questions": len(validation), "pretrained": before, "fine_tuned": after}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Final model: {final_model_path}")
    print(f"Metrics: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
