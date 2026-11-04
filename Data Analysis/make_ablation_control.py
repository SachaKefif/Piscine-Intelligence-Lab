"""Create a matched random-negative control for the 25 reviewed QMSum triplets.

Questions and positives are identical. A seeded draw selects a different passage
from the same train meeting, disjoint from every annotated relevant span.
Non-overlap is an automatic label and does not imply human semantic review.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from finetune_qmsum_e5 import PROCESSED, ROOT, read_jsonl
from train_qmsum_hard_negatives import DEFAULT_TRIPLETS, load_reviewed_triplets


OUTPUT = ROOT / "Data/Curated/qmsum_random_negative_control_25.jsonl"


def make_control(source=DEFAULT_TRIPLETS, seed=42):
    reviewed = load_reviewed_triplets(source)
    meetings = {row["meeting_id"]: row for row in read_jsonl(PROCESSED / "qmsum.jsonl")
                if row["split"] == "train"}
    passages = defaultdict(list)
    for row in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl"):
        if row["split"] == "train":
            passages[row["meeting_id"]].append(row)
    rng = random.Random(seed)
    result = []
    for row in reviewed:
        query_id = row["query_id"]
        question_index = int(query_id.rsplit(":", 1)[1])
        spans = meetings[row["meeting_id"]]["queries"]["specific"][question_index]["relevant_text_span"]
        candidates = [passage for passage in passages[row["meeting_id"]]
                      if passage["passage_id"] not in {row["positive_id"], row["negative_id"]}
                      and not any(passage["start_turn"] <= int(end)
                                  and int(start) <= passage["end_turn"] for start, end in spans)]
        if not candidates:
            raise ValueError(f"No non-overlapping control for {row['id']}")
        choice = rng.choice(candidates)
        result.append({**row, "negatif_difficile": choice["text"],
                       "negative_id": choice["passage_id"], "negative_score": None,
                       "review_status": "control_nonoverlap",
                       "review_note": "Tirage aléatoire dans la même réunion, hors spans QMSum ; non revu sémantiquement."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_TRIPLETS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = make_control(args.source, args.seed)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result),
                           encoding="utf-8", newline="\n")
    load_reviewed_triplets(args.output, control_nonoverlap=True)
    print(f"{len(result)} contrôles hors spans QMSum enregistrés : {args.output}")


if __name__ == "__main__":
    main()
