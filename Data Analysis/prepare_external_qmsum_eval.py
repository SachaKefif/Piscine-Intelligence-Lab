"""Freeze an agent transfer test from QMSum's human-authored questions and answers.

This is a stress test in English. It does not supply independent email or recipient
requests, and the source annotations are answer summaries and relevant turn spans,
not reviewed decision/owner labels.
"""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data/Datasets/QMSum/data/Committee/test"
OUT = ROOT / "Data/Curated"
MEETINGS = ("education_13", "education_17")
PREFIX = "qmsum_external_agent"
WEIGHTS = "1882ffc39c626972770c4f4778a12e72db9d0a1f60e40943c52c9689e3f21a46"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def prepare():
    units, cases = [], []
    for meeting in MEETINGS:
        raw = json.loads((RAW / f"{meeting}.json").read_text(encoding="utf-8"))
        turns = raw["meeting_transcripts"]
        for index, item in enumerate(raw["specific_query_list"]):
            query = item["query"]
            if not query.startswith(("What ", "How ")):
                continue
            spans = item["relevant_text_span"]
            if len(spans) != 1:
                raise ValueError(f"Expected one annotated span: {meeting}/{index}")
            start, end = map(int, spans[0])
            if not 0 <= start <= end < len(turns):
                raise ValueError(f"Invalid annotated span: {meeting}/{index}")
            identifier = f"QMSum:{meeting}:{start}-{end}:{index}"
            units.append({
                "unit_id": identifier, "source_id": f"QMSum:{meeting}",
                "source_date": None, "type": "answer", "statement": item["answer"].strip(),
                "evidence": "\n".join(turn["content"] for turn in turns[start:end + 1]),
                "start_turn": start, "end_turn": end, "status": "asserted",
                "speaker": None, "owner": None, "due_date": None,
                "project_id": "Welsh committee",
            })
            cases.append({
                "id": f"{meeting}_{index}", "request": query,
                "expected_intent": "answer", "expected_status": "ok",
                "required_unit_ids": [identifier], "allowed_unit_ids": [identifier],
            })
    write_jsonl(OUT / f"{PREFIX}_units.jsonl", units)
    write_jsonl(OUT / f"{PREFIX}_cases.jsonl", cases)
    write_jsonl(OUT / f"{PREFIX}_relations.jsonl", [])
    topics = [{
        "topic_id": "welsh_committee", "project_id": "Welsh committee",
        "aliases": ["law", "out-of-court disposals", "registered intermediaries",
                    "CPS guidance", "unintended consequences", "prosecutions",
                    "Welsh bac", "north Wales", "teacher-training qualification",
                    "employment", "universities"],
        "unit_ids": [row["unit_id"] for row in units],
    }]
    write_jsonl(OUT / f"{PREFIX}_topics.jsonl", topics)
    manifest = {
        "source": "Yale-LILY/QMSum data/Committee/test; original English specific queries, answers and relevant_text_span",
        "meetings": list(MEETINGS), "cases": len(cases),
        "input_sha256": {
            **{f"raw_{meeting}": digest(RAW / f"{meeting}.json") for meeting in MEETINGS},
            **{key: digest(OUT / f"{PREFIX}_{key}.jsonl")
               for key in ("units", "cases", "relations", "topics")},
        },
        "frozen_model_weights_sha256": WEIGHTS,
        "limitations": [
            "Questions are English while EF-06 is trained for French.",
            "QMSum provides no user-authored email/task/event requests or recipient labels.",
            "Answer summaries and relevant spans are external annotations; unit conversion is automatic.",
        ],
    }
    (OUT / f"{PREFIX}_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
