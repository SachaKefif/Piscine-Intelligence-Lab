"""Audit the local meeting corpora and QMSum retrieval labels.

Run from anywhere with: python "Data Analysis/audit_week1_data.py"
Writes a deterministic summary to docs/audit_week1_data.json.
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
OUTPUT = ROOT / "docs" / "audit_week1_data.json"
CORPORA = ("ami", "icsi", "meetingbank", "qmsum")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transcript_hash(record):
    text = "\n".join(segment["text"] for segment in record["segments"])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_corpus(name):
    path = PROCESSED / f"{name}.jsonl"
    records = list(read_jsonl(path))
    split_counts = Counter((record.get("split") or "unassigned") for record in records)
    ids = defaultdict(set)
    texts = defaultdict(set)
    duplicate_ids = 0
    missing_sources = 0
    source_paths = set()
    questions = Counter()
    for record in records:
        split = record.get("split") or "unassigned"
        meeting_id = record["meeting_id"]
        duplicate_ids += meeting_id in ids[split]
        ids[split].add(meeting_id)
        texts[split].add(transcript_hash(record))
        sources = record.get("source_files") or []
        missing_sources += not bool(sources)
        source_paths.update(sources)
        if name == "qmsum":
            questions[split] += len(record["queries"]["specific"])
    split_pairs = [
        {"splits": [left, right], "same_id": len(ids[left] & ids[right]),
         "same_exact_transcript": len(texts[left] & texts[right])}
        for index, left in enumerate(sorted(ids)) for right in sorted(ids)[index + 1:]
    ]
    missing_paths = [source for source in sorted(source_paths)
                     if not (ROOT / source).is_file()]
    summary = {
        "records": len(records),
        "splits": dict(sorted(split_counts.items())),
        "specific_questions": dict(sorted(questions.items())) if questions else None,
        "missing_source_date": sum(record.get("source_date") is None for record in records),
        "missing_source_files": missing_sources,
        "missing_source_paths": len(missing_paths),
        "duplicate_ids_within_split": duplicate_ids,
        "cross_split_overlap": split_pairs,
        "sha256": sha256_file(path),
    }
    return summary, records, ids


def audit_qmsum_passages(meetings):
    path = PROCESSED / "qmsum_10_tours.jsonl"
    passages = list(read_jsonl(path))
    by_meeting = defaultdict(list)
    ids = set()
    duplicate_ids = 0
    for passage in passages:
        duplicate_ids += passage["passage_id"] in ids
        ids.add(passage["passage_id"])
        by_meeting[passage["meeting_id"]].append(passage)

    split_counts = Counter(passage["split"] for passage in passages)
    empty_texts = sum(not passage["text"].strip() for passage in passages)
    coverage_errors = []
    split_errors = []
    invalid_spans = []
    no_gold = []
    gold_counts = []
    for meeting in meetings:
        meeting_id = meeting["meeting_id"]
        candidates = sorted(by_meeting[meeting_id], key=lambda p: p["start_turn"])
        turns = len(meeting["segments"])
        expected_start = 0
        for passage in candidates:
            if passage["split"] != meeting["split"]:
                split_errors.append(passage["passage_id"])
            if passage["start_turn"] != expected_start or passage["end_turn"] < expected_start:
                coverage_errors.append(passage["passage_id"])
            expected_start = passage["end_turn"] + 1
        if expected_start != turns:
            coverage_errors.append(meeting_id)
        for question_number, question in enumerate(meeting["queries"]["specific"]):
            spans = []
            for start, end in question["relevant_text_span"]:
                start, end = int(start), int(end)
                if not 0 <= start <= end < turns:
                    invalid_spans.append(f"{meeting_id}:{question_number}")
                spans.append((start, end))
            gold = [p for p in candidates if any(
                p["start_turn"] <= end and p["end_turn"] >= start
                for start, end in spans)]
            gold_counts.append(len(gold))
            if not gold:
                no_gold.append(f"{meeting_id}:{question_number}")

    return {
        "records": len(passages),
        "splits": dict(sorted(split_counts.items())),
        "empty_texts": empty_texts,
        "duplicate_passage_ids": duplicate_ids,
        "coverage_errors": len(coverage_errors),
        "split_mismatches": len(split_errors),
        "invalid_question_spans": len(invalid_spans),
        "questions_without_gold_passage": len(no_gold),
        "gold_passages_per_question_median": median(gold_counts),
        "gold_passages_per_question_max": max(gold_counts),
        "questions_with_more_than_10_gold_passages": sum(n > 10 for n in gold_counts),
        "sha256": sha256_file(path),
    }


def audit_meetingbank_groups(records):
    """Infer a meeting from the city and date at the start of a MeetingBank UID."""
    groups = defaultdict(set)
    malformed = 0
    for record in records:
        parts = record["meeting_id"].split("_", 2)
        if len(parts) < 3:
            malformed += 1
            continue
        groups["_".join(parts[:2])].add(record["split"])
    return {
        "heuristic": "first two underscore-delimited UID fields: city and date",
        "inferred_meetings": len(groups),
        "malformed_uids": malformed,
        "meetings_in_multiple_splits": sum(len(splits) > 1 for splits in groups.values()),
        "meetings_in_train_and_validation": sum(
            {"train", "validation"} <= splits for splits in groups.values()),
        "meetings_in_train_and_test": sum(
            {"train", "test"} <= splits for splits in groups.values()),
        "meetings_in_validation_and_test": sum(
            {"validation", "test"} <= splits for splits in groups.values()),
    }


def main():
    report = {"corpora": {}}
    records_by_name = {}
    ids_by_name = {}
    for name in CORPORA:
        summary, records, ids = audit_corpus(name)
        report["corpora"][name] = summary
        records_by_name[name] = records
        ids_by_name[name] = ids
    report["qmsum_passages"] = audit_qmsum_passages(records_by_name["qmsum"])
    report["meetingbank_meeting_group_audit"] = audit_meetingbank_groups(
        records_by_name["meetingbank"])
    report["qmsum_overlap_with_source_corpora"] = {
        source: {
            split: len(ids_by_name[source]["unassigned"] & qmsum_ids)
            for split, qmsum_ids in sorted(ids_by_name["qmsum"].items())
        }
        for source in ("ami", "icsi")
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    for name, summary in report["corpora"].items():
        print(f"{name}: {summary['records']} records; splits={summary['splits']}; "
              f"missing source paths={summary['missing_source_paths']}")
    print("QMSum passage checks:", {key: report["qmsum_passages"][key] for key in (
        "coverage_errors", "split_mismatches", "invalid_question_spans",
        "questions_without_gold_passage")})
    print("Cross-corpus meeting ID overlaps:", report["qmsum_overlap_with_source_corpora"])
    print("MeetingBank inferred meeting groups:", report["meetingbank_meeting_group_audit"])


if __name__ == "__main__":
    main()
