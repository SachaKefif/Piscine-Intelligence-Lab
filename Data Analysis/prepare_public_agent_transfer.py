"""Freeze an untouched agent transfer fixture from original QMSum test Q&A.

The gold answers become evidence-linked units; extraction from raw transcripts
is therefore outside this test. The original annotators wrote the questions.
"""

import hashlib
import argparse
import json
from pathlib import Path

from ingest_semantic_units import ROOT
from prepare_agent_holdout import prepare


RAW = ROOT / "Data/Datasets/QMSum/data/Academic/test"
MEETINGS = ("Bed003", "Bed008")
BASE = ROOT / "Data/Curated/qmsum_academic_transfer"
CONVERSATIONS = ROOT / "Data/Processed/qmsum_academic_transfer_conversations.jsonl"


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def freeze(*, meetings=MEETINGS, base=BASE, conversations=CONVERSATIONS):
    units, cases = [], []
    raw_hashes = {}
    for meeting in meetings:
        path = RAW / f"{meeting}.json"
        raw_hashes[f"raw_{meeting}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        raw = json.loads(path.read_text(encoding="utf-8"))
        turns = raw["meeting_transcripts"]
        for index, row in enumerate(raw["specific_query_list"]):
            spans = [(int(start), int(end)) for start, end in row["relevant_text_span"]]
            start, end = min(s for s, _ in spans), max(e for _, e in spans)
            uid = f"QMSum:{meeting}:{start}-{end}:{index}"
            evidence = "\n".join(turns[i]["content"] for i in range(start, end + 1))
            units.append({"unit_id": uid, "source_id": f"QMSum:{meeting}",
                          "source_date": None, "type": "answer", "statement": row["answer"],
                          "evidence": evidence, "start_turn": start, "end_turn": end,
                          "status": "asserted", "speaker": None, "owner": None,
                          "due_date": None, "project_id": "QMSumAcademic"})
            cases.append({"id": f"{meeting}_{index}", "request": row["query"],
                          "expected_intent": "answer", "expected_status": "ok",
                          "required_unit_ids": [uid], "allowed_unit_ids": [uid],
                          "required_sections": ["answer"]})
    paths = {name: Path(str(base) + f"_{name}.jsonl") for name in ("units", "relations", "topics", "cases")}
    if any(path.exists() for path in paths.values()) or Path(str(base) + "_manifest.json").exists():
        raise ValueError("Le jeu de transfert est déjà figé ; aucun écrasement autorisé")
    write_jsonl(paths["units"], units)
    write_jsonl(paths["relations"], [])
    write_jsonl(paths["topics"], [{"topic_id": "academic_transfer", "project_id": "QMSumAcademic",
                                   "aliases": ["QMSumAcademic"],
                                   "unit_ids": [row["unit_id"] for row in units]}])
    write_jsonl(paths["cases"], cases)
    prepare(conversations, meetings=meetings, units=paths["units"], raw_dir=RAW)
    hashes = {**raw_hashes, **{name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for name, path in paths.items()}}
    manifest = {"input_sha256": hashes, "cases": len(cases), "source": "QMSum Academic test original questions"}
    Path(str(base) + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"cases": len(cases), "meetings": list(meetings), "manifest": str(base) + "_manifest.json"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meetings", nargs="+", default=MEETINGS)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--conversations", type=Path, default=CONVERSATIONS)
    args = parser.parse_args()
    print(json.dumps(freeze(meetings=args.meetings, base=args.base,
                            conversations=args.conversations), ensure_ascii=False, indent=2))
