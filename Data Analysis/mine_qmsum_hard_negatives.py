"""Mine *unverified* hard-negative candidates from QMSum train meetings.

Rank passages of the question's own meeting with the current E5 checkpoint.
Exclude gold passages and passages within ten turns of a gold span. A passage
outside the annotated span can still answer the question, so review candidates
with review_qmsum_hard_negatives.py before any training.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from finetune_qmsum_e5 import PROCESSED, ROOT, read_jsonl, relevant_passages


def far_from_spans(passage, spans, buffer_turns=10):
    return all(
        passage["end_turn"] < int(start) - buffer_turns
        or passage["start_turn"] > int(end) + buffer_turns
        for start, end in spans
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "Models/qmsum-e5-small/checkpoints/checkpoint-842")
    parser.add_argument("--output", type=Path,
                        default=PROCESSED / "qmsum_hard_negative_candidates.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--buffer-turns", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.top_k < 1 or args.buffer_turns < 0:
        parser.error("batch-size and top-k must be positive; buffer-turns must be nonnegative")

    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not (checkpoint / "modules.json").is_file():
        parser.error(f"Not a Sentence Transformers checkpoint: {checkpoint}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(checkpoint))
    model.max_seq_length = 512
    meetings = {
        row["meeting_id"]: row for row in read_jsonl(PROCESSED / "qmsum.jsonl")
        if row["split"] == "train"
    }
    passages = defaultdict(list)
    for row in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl"):
        if row["split"] == "train" and row["text"].strip():
            passages[row["meeting_id"]].append(row)

    rows = []
    for meeting_number, (meeting_id, meeting) in enumerate(meetings.items(), 1):
        candidates = passages[meeting_id]
        questions = meeting["queries"]["specific"]
        if not candidates or not questions:
            continue
        passage_vectors = model.encode(
            ["passage: " + passage["text"] for passage in candidates],
            batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        question_vectors = model.encode(
            ["query: " + question["query"] for question in questions],
            batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        id_to_index = {passage["passage_id"]: index for index, passage in enumerate(candidates)}
        for question_index, (question, question_vector) in enumerate(zip(questions, question_vectors)):
            spans = question["relevant_text_span"]
            gold = relevant_passages(candidates, spans)
            if not gold:
                raise ValueError(f"No gold passage for {meeting_id}:{question_index}")
            scores = passage_vectors @ question_vector
            positive = max(gold, key=lambda passage: scores[id_to_index[passage["passage_id"]]])
            gold_ids = {passage["passage_id"] for passage in gold}
            negatives = [
                (index, passage) for index, passage in enumerate(candidates)
                if passage["passage_id"] not in gold_ids
                and far_from_spans(passage, spans, args.buffer_turns)
            ]
            negatives.sort(key=lambda item: float(scores[item[0]]), reverse=True)
            for rank, (index, negative) in enumerate(negatives[:args.top_k], 1):
                rows.append({
                    "query_id": f"{meeting_id}:{question_index}",
                    "meeting_id": meeting_id,
                    "domain": meeting["domain"],
                    "question": question["query"],
                    "reference_answer": question["answer"],
                    "relevant_text_span": spans,
                    "positive_id": positive["passage_id"],
                    "positive": positive["text"],
                    "negative_id": negative["passage_id"],
                    "negative": negative["text"],
                    "negative_rank": rank,
                    "negative_score": round(float(scores[index]), 5),
                    "review_status": "unverified",
                })
        if meeting_number % 25 == 0:
            print(f"Mined {meeting_number}/{len(meetings)} train meetings", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} unverified candidates: {output}")


if __name__ == "__main__":
    main()
