"""Audit two blind human reviews and score the frozen AMI predictions.

No score is emitted while labels are missing or disagreements remain unresolved.
An optional adjudication file may supply only disputed items after both reviews.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_independent_ami_review import DEFAULT_OUTPUT, OWNER_FIELDS, PAIR_FIELDS, source_sha256


SEMANTIC = {"supports", "contradicts", "unrelated", "uncertain"}
TEMPORAL = {"supersedes", "not_supersedes", "unknown_order"}


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_owner(value):
    value = " ".join(value.casefold().split())
    return None if value in {"", "none", "unknown", "ambiguous"} else value


def load_set(directory, task, reviewer, expected):
    path = directory / f"reviewer_{reviewer}_{task}.csv"
    data = rows(path)
    fields = PAIR_FIELDS if task == "pairs" else OWNER_FIELDS
    if set(data[0]) != set(fields) or len(data) != len(expected):
        raise ValueError(f"Structure ou effectif modifié : {path}")
    by_id = {row["item_id"]: row for row in data}
    if set(by_id) != set(expected):
        raise ValueError(f"Identifiants modifiés ou dupliqués : {path}")
    for uid, original in expected.items():
        for key, value in original.items():
            if by_id[uid].get(key) != value:
                raise ValueError(f"Texte ou source modifié dans {path}: {uid}, {key}")
    return by_id


def item_map(path):
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {row["item_id"]: row for row in result}


def verify(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    for meeting, digest in manifest["source_sha256"].items():
        source = Path(__file__).resolve().parents[1] / f"Data/Datasets/AMI/abstractive/{meeting}.abssumm.xml"
        if source_sha256(source) != digest:
            raise ValueError(f"Source AMI modifiée : {meeting}")
    for meeting, digest in manifest.get("transcript_sha256", {}).items():
        if hashlib.sha256((directory / f"transcript_{meeting}.txt").read_bytes()).hexdigest() != digest:
            raise ValueError(f"Transcription figée modifiée : {meeting}")
    for name in ("pair_items", "owner_items", "predictions"):
        path = directory / ("model_predictions.jsonl" if name == "predictions" else f"{name}.jsonl")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[f"{name}_sha256"]:
            raise ValueError(f"Jeu figé modifié : {name}")
    for reviewer, digest in manifest.get("reviewer_packet_sha256", {}).items():
        packet = directory / f"reviewer_{reviewer}_packet.zip"
        if hashlib.sha256(packet.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Paquet d'annotation modifié : {reviewer}")
    return (item_map(directory / "pair_items.jsonl"),
            item_map(directory / "owner_items.jsonl"),
            item_map(directory / "model_predictions.jsonl"))


def review_state(directory, adjudicated_pairs=None, adjudicated_owners=None):
    pairs, owners, predictions = verify(directory)
    reviews = {reviewer: {"pairs": load_set(directory, "pairs", reviewer, pairs),
                          "owners": load_set(directory, "owners", reviewer, owners)}
               for reviewer in ("a", "b")}
    missing = []
    disagreements = []
    gold = {"pairs": {}, "owners": {}}
    for task, items in (("pairs", pairs), ("owners", owners)):
        for uid, item in items.items():
            a, b = reviews["a"][task][uid], reviews["b"][task][uid]
            if task == "pairs":
                fields = ("semantic_relation", "temporal_relation")
                for reviewer, row in (("a", a), ("b", b)):
                    if row["semantic_relation"] not in SEMANTIC or row["temporal_relation"] not in TEMPORAL:
                        missing.append(f"{reviewer}:{uid}")
                    if row["temporal_relation"] == "supersedes" and not row["chronology_evidence"].strip():
                        missing.append(f"{reviewer}:{uid}:chronology_evidence")
            else:
                fields = ("owner",)
                for reviewer, row in (("a", a), ("b", b)):
                    if not row["owner"].strip():
                        missing.append(f"{reviewer}:{uid}")
                    owner = normalized_owner(row["owner"])
                    if owner and (not row["owner_evidence"].strip()
                                  or row["owner_evidence"].casefold() not in item["text"].casefold()):
                        missing.append(f"{reviewer}:{uid}:owner_evidence")
            if any(a[field].casefold().strip() != b[field].casefold().strip() for field in fields):
                disagreements.append({"task": task, "item_id": uid,
                                      "a": {field: a[field] for field in fields},
                                      "b": {field: b[field] for field in fields}})
            else:
                gold[task][uid] = {field: a[field] for field in fields}
    if adjudicated_pairs or adjudicated_owners:
        for task, path in (("pairs", adjudicated_pairs), ("owners", adjudicated_owners)):
            if path is None:
                continue
            records = rows(path)
            fields = PAIR_FIELDS if task == "pairs" else OWNER_FIELDS
            if records and set(records[0]) != set(fields):
                raise ValueError(f"Colonnes d'arbitrage modifiées : {path}")
            adjudications = {row["item_id"]: row for row in records}
            disputed_ids = {row["item_id"] for row in disagreements if row["task"] == task}
            if len(adjudications) != len(records) or not set(adjudications) <= disputed_ids:
                raise ValueError(f"Arbitrage hors désaccord ou identifiant dupliqué : {path}")
            originals = pairs if task == "pairs" else owners
            for uid, row in adjudications.items():
                if any(row.get(key) != value for key, value in originals[uid].items()):
                    raise ValueError(f"Texte ou source modifié dans l'arbitrage : {uid}")
            for dispute in disagreements:
                if dispute["task"] != task or dispute["item_id"] not in adjudications:
                    continue
                row = adjudications[dispute["item_id"]]
                fields = ("semantic_relation", "temporal_relation") if task == "pairs" else ("owner",)
                if task == "pairs" and (row["semantic_relation"] not in SEMANTIC
                                        or row["temporal_relation"] not in TEMPORAL
                                        or row["temporal_relation"] == "supersedes" and not row["chronology_evidence"].strip()):
                    raise ValueError(f"Arbitrage invalide : {dispute['item_id']}")
                if task == "owners" and (not row["owner"].strip() or normalized_owner(row["owner"])
                                          and row["owner_evidence"].casefold() not in owners[dispute["item_id"]]["text"].casefold()):
                    raise ValueError(f"Arbitrage invalide : {dispute['item_id']}")
                gold[task][dispute["item_id"]] = {field: row[field] for field in fields}
    pending_disputes = [row for row in disagreements if row["item_id"] not in gold[row["task"]]]
    state = {"pair_items": len(pairs), "owner_items": len(owners),
             "missing_reviews": sorted(set(missing)), "disagreements": disagreements,
             "pending_adjudication": pending_disputes,
             "pair_agreement_rate": None if missing else round((len(pairs) - sum(row["task"] == "pairs" for row in disagreements)) / len(pairs), 4),
             "owner_agreement_rate": None if missing else round((len(owners) - sum(row["task"] == "owners" for row in disagreements)) / len(owners), 4)}
    if missing or pending_disputes:
        state["status"] = "pending_human_review"
        return state
    pair_gold = gold["pairs"]
    owner_gold = gold["owners"]
    semantic_rows = [(pair_gold[uid]["semantic_relation"], predictions[uid]["prediction"]) for uid in pairs
                     if pair_gold[uid]["semantic_relation"] != "uncertain"]
    temporal_rows = [(pair_gold[uid]["temporal_relation"] == "supersedes",
                      predictions[uid]["prediction"] == "supersedes") for uid in pairs
                     if pair_gold[uid]["temporal_relation"] != "unknown_order"]
    owner_rows = [(normalized_owner(owner_gold[uid]["owner"]),
                   normalized_owner(predictions[uid]["prediction"] or "")) for uid in owners]
    def binary(rows, positive):
        tp = sum(gold == positive and predicted == positive for gold, predicted in rows)
        fp = sum(gold != positive and predicted == positive for gold, predicted in rows)
        fn = sum(gold == positive and predicted != positive for gold, predicted in rows)
        return {"positives": tp + fn, "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None,
                "tp": tp, "fp": fp, "fn": fn}
    state.update({"status": "scored", "semantic_evaluated": len(semantic_rows),
                  "contradicts": binary(semantic_rows, "contradicts"),
                  "temporal_evaluated": len(temporal_rows),
                  "supersedes": binary(temporal_rows, True),
                  "owner_items_evaluated": len(owner_rows),
                  "owner_exact_accuracy": round(sum(g == p for g, p in owner_rows) / len(owner_rows), 4),
                  "owner_wrong_assignments": sum(p is not None and p != g for g, p in owner_rows),
                  "owner_unjustified_abstentions": sum(g is not None and p is None for g, p in owner_rows)})
    return state


def write_adjudication_templates(directory, state, destination):
    """Prepare only disputed items for a third reviewer, without model outputs."""
    pairs, owners, _ = verify(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for task, items, fields in (("pairs", pairs, PAIR_FIELDS),
                                ("owners", owners, OWNER_FIELDS)):
        disputes = [row for row in state["disagreements"] if row["task"] == task]
        if not disputes:
            continue
        path = destination / f"adjudicated_{task}.csv"
        with path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for dispute in disputes:
                writer.writerow({**items[dispute["item_id"]],
                                 "notes": f"A: {dispute['a']} | B: {dispute['b']}"})
        paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adjudicated-pairs", type=Path)
    parser.add_argument("--adjudicated-owners", type=Path)
    parser.add_argument("--adjudication-dir", type=Path,
                        help="Create blank third-reviewer CSVs for current disagreements")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = review_state(args.directory, args.adjudicated_pairs, args.adjudicated_owners)
    if args.adjudication_dir:
        result["adjudication_templates"] = write_adjudication_templates(
            args.directory, result, args.adjudication_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
