"""Prepare a QMSum negative review queue and export only reviewed triplets.

The lexical filters make review manageable; they do not prove irrelevance.
Read each complete question, reference answer, positive and negative passage.
Set review_status to accepted or rejected and write a nonempty review_note.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from finetune_qmsum_e5 import PROCESSED, ROOT, read_jsonl
from mine_qmsum_hard_negatives import far_from_spans


CANDIDATES = PROCESSED / "qmsum_hard_negative_candidates.jsonl"
QUEUE = ROOT / "Data/Curated/qmsum_hard_negative_review_queue.jsonl"
REVIEWED = ROOT / "Data/Curated/qmsum_hard_negatives_reviewed.jsonl"
STOPWORDS = set("""
    the and that this with from were have been they their there about would
    could should which what when where why who how was are for into then than
    also some each after before between during group team meeting said
    discussed think thought because according considered suggested decided
    wanted needed using used make made them those these while more most much
    many very only still same such will can had has did does all one two
    project manager marketing user interface industrial designer
""".split())


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def content_tokens(text):
    return {token for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 3 and token not in STOPWORDS}


def answer_overlap(answer, passage):
    terms = content_tokens(answer)
    return len(terms & content_tokens(passage)) / max(1, len(terms))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args):
    output = resolve_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Review queue already exists: {output}; use --overwrite explicitly")
    shortlisted = []
    seen_queries = set()
    for row in read_jsonl(resolve_path(args.candidates)):
        if row["negative_rank"] != 1 or row["query_id"] in seen_queries:
            continue
        seen_queries.add(row["query_id"])
        if row["negative_score"] < args.min_score:
            continue
        if row["question"].lower().startswith("summarize"):
            continue
        if any(len(row[key]) > args.max_chars for key in ("positive", "negative")):
            continue
        if len(row["reference_answer"]) > 600:
            continue
        if answer_overlap(row["reference_answer"], row["positive"]) < 0.35:
            continue
        if answer_overlap(row["reference_answer"], row["negative"]) > 0.10:
            continue
        shortlisted.append({**row, "review_status": "pending", "review_note": ""})
    write_jsonl(output, shortlisted)
    print(f"Wrote {len(shortlisted)} candidates for semantic review: {output}")
    print("The answer-overlap filter is a triage heuristic, not a relevance label.")


def export(args):
    if args.buffer_turns < 0:
        raise ValueError("buffer-turns must be nonnegative")
    source = resolve_path(args.review_file)
    output = resolve_path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Reviewed triplets already exist: {output}; use --overwrite explicitly")
    reviews = list(read_jsonl(source))
    if not reviews:
        raise ValueError("Review file is empty")
    incomplete = [
        row["query_id"] for row in reviews
        if row.get("review_status") not in ("accepted", "rejected")
        or not row.get("review_note", "").strip()
    ]
    if incomplete:
        examples = ", ".join(incomplete[:5])
        raise ValueError(
            f"{len(incomplete)} candidate(s) still need review (e.g. {examples}). "
            f"Edit {source}: set review_status to accepted/rejected and add review_note "
            "for every line before exporting."
        )
    query_ids = [row["query_id"] for row in reviews]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Only one reviewed candidate per query_id is allowed")

    train_meetings = {
        row["meeting_id"]: row for row in read_jsonl(PROCESSED / "qmsum.jsonl")
        if row["split"] == "train"
    }
    train_passages = {
        row["passage_id"]: row for row in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl")
        if row["split"] == "train"
    }
    accepted, rejected = [], []
    for row in reviews:
        status = row.get("review_status")
        note = row.get("review_note", "").strip()
        if status not in ("accepted", "rejected") or not note:
            raise ValueError(f"Review status/note missing for {row['query_id']}")
        meeting = train_meetings.get(row["meeting_id"])
        if meeting is None:
            raise ValueError(f"Non-train meeting: {row['query_id']}")
        index = int(row["query_id"].rsplit(":", 1)[1])
        if row["query_id"] != f"{row['meeting_id']}:{index}":
            raise ValueError(f"Query ID and meeting mismatch: {row['query_id']}")
        question = meeting["queries"]["specific"][index]
        if row["question"] != question["query"] or row["reference_answer"] != question["answer"]:
            raise ValueError(f"Question or answer changed: {row['query_id']}")
        positive = train_passages.get(row["positive_id"])
        negative = train_passages.get(row["negative_id"])
        if positive is None or negative is None:
            raise ValueError(f"Missing train passage: {row['query_id']}")
        if positive["meeting_id"] != row["meeting_id"] or negative["meeting_id"] != row["meeting_id"]:
            raise ValueError(f"Passage from another meeting: {row['query_id']}")
        if positive["text"] != row["positive"] or negative["text"] != row["negative"]:
            raise ValueError(f"Passage text changed: {row['query_id']}")
        spans = question["relevant_text_span"]
        if not any(positive["start_turn"] <= int(end) and positive["end_turn"] >= int(start)
                   for start, end in spans):
            raise ValueError(f"Positive misses gold spans: {row['query_id']}")
        if not far_from_spans(negative, spans, args.buffer_turns):
            raise ValueError(f"Negative too close to gold spans: {row['query_id']}")
        if status == "rejected":
            rejected.append({"query_id": row["query_id"], "reason": note})
            continue
        accepted.append({
            "id": f"QH{len(accepted) + 1:03d}",
            "categorie": row["domain"],
            "question": row["question"],
            "positif": row["positive"],
            "negatif_difficile": row["negative"],
            "query_id": row["query_id"],
            "meeting_id": row["meeting_id"],
            "positive_id": row["positive_id"],
            "negative_id": row["negative_id"],
            "reference_answer": row["reference_answer"],
            "negative_score": row["negative_score"],
            "review_status": "accepted_after_semantic_review",
            "review_note": note,
        })
    if not accepted:
        raise ValueError("No accepted triplets")
    write_jsonl(output, accepted)
    audit = {
        "reviewed": len(reviews), "accepted": len(accepted), "rejected": len(rejected),
        "accepted_by_domain": dict(Counter(row["categorie"] for row in accepted)),
        "rejected_queries": rejected,
        "rule": "Every accepted triplet requires a semantic review and source/span checks.",
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(accepted)} reviewed train triplets: {output}")
    print(f"Audit: {audit_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    make = subparsers.add_parser("prepare", help="Create a queue for manual semantic review")
    make.add_argument("--candidates", type=Path, default=CANDIDATES)
    make.add_argument("--output", type=Path, default=QUEUE)
    make.add_argument("--min-score", type=float, default=0.40)
    make.add_argument("--max-chars", type=int, default=2000)
    make.add_argument("--overwrite", action="store_true")
    make.set_defaults(func=prepare)
    emit = subparsers.add_parser("export", help="Validate reviews and export accepted triplets")
    emit.add_argument("--review-file", type=Path, default=QUEUE)
    emit.add_argument("--output", type=Path, default=REVIEWED)
    emit.add_argument("--buffer-turns", type=int, default=10)
    emit.add_argument("--overwrite", action="store_true")
    emit.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
