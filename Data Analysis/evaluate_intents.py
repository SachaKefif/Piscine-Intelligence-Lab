"""Evaluate EF-06 on held-out synthetic requests and report slot accuracy."""

import argparse
import hashlib
import json
from pathlib import Path

from ingest_semantic_units import ROOT
from parse_intent import DEFAULT_MODEL, INTENTS, load_classifier, parse_intent


DEFAULT_TRAIN = ROOT / "Data" / "Curated" / "intent_train.jsonl"
DEFAULT_VALIDATION = ROOT / "Data" / "Curated" / "intent_validation.jsonl"
DEFAULT_EVAL = ROOT / "Data" / "Curated" / "intent_eval_holdout.jsonl"
DEFAULT_REPORT = ROOT / "Data" / "Curated" / "intent_eval_metrics.json"
SLOTS = ("subject", "person", "due_date", "action")


def load_jsonl(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(ids) != len(set(ids)) or {row["intent"] for row in rows} != set(INTENTS):
        raise ValueError(f"Jeu vide, identifiants dupliqués ou intentions manquantes : {path}")
    for row in rows:
        if not isinstance(row["request"], str) or not row["request"].strip():
            raise ValueError(f"Demande invalide : {row['id']}")
        for slot in SLOTS:
            if row[slot] is not None and not isinstance(row[slot], str):
                raise ValueError(f"Paramètre invalide : {row['id']} / {slot}")
    return rows, hashlib.sha256(raw).hexdigest()


def evaluate(train_path=DEFAULT_TRAIN, validation_path=DEFAULT_VALIDATION, eval_path=DEFAULT_EVAL,
             *, classifier=None, use_rules=False):
    train, train_hash = load_jsonl(train_path)
    validation, validation_hash = load_jsonl(validation_path)
    held_out, eval_hash = load_jsonl(eval_path)
    train_ids = {row["id"] for row in train}
    validation_ids = {row["id"] for row in validation}
    eval_ids = {row["id"] for row in held_out}
    if train_ids & validation_ids or train_ids & eval_ids or validation_ids & eval_ids:
        raise ValueError("Les identifiants des splits se chevauchent")
    train_projects = {row["subject"].split()[-1] for row in train if row["subject"]}
    validation_projects = {row["subject"].split()[-1] for row in validation if row["subject"]}
    eval_projects = {row["subject"].split()[-1] for row in held_out if row["subject"]}
    if train_projects & validation_projects or train_projects & eval_projects or validation_projects & eval_projects:
        raise ValueError("Les projets des splits se chevauchent")

    details = []
    for row in held_out:
        prediction = parse_intent(row["request"], classifier=classifier, use_rules=use_rules)
        details.append({"id": row["id"], "gold": {key: row[key] for key in ("intent", *SLOTS)},
                        "predicted": prediction})

    by_class = {}
    for label in INTENTS:
        true_positive = sum(d["gold"]["intent"] == d["predicted"]["intent"] == label for d in details)
        support = sum(d["gold"]["intent"] == label for d in details)
        predicted = sum(d["predicted"]["intent"] == label for d in details)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        by_class[label] = {"support": support, "precision": precision, "recall": recall,
                           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

    slot_metrics = {}
    routed = [d for d in details if d["gold"]["intent"] == d["predicted"]["intent"]]
    for slot in SLOTS:
        matches = sum(d["gold"][slot] == d["predicted"][slot] for d in details)
        present = [d for d in details if d["gold"][slot] is not None]
        routed_present = [d for d in routed if d["gold"][slot] is not None]
        slot_metrics[slot] = {
            "exact_accuracy": matches / len(details),
            "gold_present_count": len(present),
            "gold_present_exact_accuracy": (sum(d["gold"][slot] == d["predicted"][slot]
                                                for d in present) / len(present) if present else None),
            "correct_intent_exact_accuracy": (sum(d["gold"][slot] == d["predicted"][slot]
                                                  for d in routed) / len(routed) if routed else None),
            "gold_present_and_correct_intent_count": len(routed_present),
            "gold_present_and_correct_intent_exact_accuracy": (
                sum(d["gold"][slot] == d["predicted"][slot] for d in routed_present) / len(routed_present)
                if routed_present else None),
        }
    gold_clarify = [d for d in details if d["gold"]["intent"] == "clarify"]
    gold_direct = [d for d in details if d["gold"]["intent"] != "clarify"]
    confusion = {label: {candidate: 0 for candidate in INTENTS} for label in INTENTS}
    for d in details:
        confusion[d["gold"]["intent"]][d["predicted"]["intent"]] += 1
    predicted_clarify = [d for d in details if d["predicted"]["intent"] == "clarify"]
    true_clarify = sum(d["gold"]["intent"] == "clarify" for d in predicted_clarify)
    return {
        "method": "conservative_rules_v1" if use_rules else "trained_classifier_with_slot_guards_v1",
        "split": "project_disjoint_synthetic_requests",
        "independence_note": ("The 23 evaluation seed examples were previously inspected, so this is a descriptive synthetic-data check."
                              if eval_path == DEFAULT_EVAL else
                              "New manually drafted synthetic requests, evaluated after the model and guard policy were fixed; not independent human data."),
        "training": {"path": str(train_path.relative_to(ROOT)) if train_path.is_relative_to(ROOT) else str(train_path),
                     "sha256": train_hash, "count": len(train), "projects": sorted(train_projects)},
        "validation": {"path": str(validation_path.relative_to(ROOT)) if validation_path.is_relative_to(ROOT) else str(validation_path),
                       "sha256": validation_hash, "count": len(validation), "projects": sorted(validation_projects)},
        "evaluation": {"path": str(eval_path.relative_to(ROOT)) if eval_path.is_relative_to(ROOT) else str(eval_path),
                       "sha256": eval_hash, "count": len(held_out), "projects": sorted(eval_projects)},
        "metrics": {
            "intent_macro_f1": sum(item["f1"] for item in by_class.values()) / len(INTENTS),
            "intent_accuracy": sum(d["gold"]["intent"] == d["predicted"]["intent"] for d in details) / len(details),
            "joint_intent_and_slots_exact_accuracy": sum(
                d["gold"]["intent"] == d["predicted"]["intent"]
                and all(d["gold"][slot] == d["predicted"][slot] for slot in SLOTS)
                for d in details) / len(details),
            "by_class": by_class, "confusion_matrix": confusion,
            "slots": slot_metrics,
            "clarification_precision": true_clarify / len(predicted_clarify) if predicted_clarify else 0.0,
            "clarification_recall": true_clarify / len(gold_clarify),
            "appropriate_clarification_rate": sum(d["predicted"]["intent"] == "clarify" for d in gold_clarify) / len(gold_clarify),
            "false_clarification_rate": sum(d["predicted"]["intent"] == "clarify" for d in gold_direct) / len(gold_direct),
        },
        "errors": [d for d in details if d["gold"]["intent"] != d["predicted"]["intent"]
                   or any(d["gold"][slot] != d["predicted"][slot] for slot in SLOTS)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--rules", action="store_true", help="Évaluer la baseline à règles")
    args = parser.parse_args()
    try:
        classifier = None if args.rules else load_classifier(args.model.resolve())
        report = evaluate(args.train.resolve(), args.validation.resolve(), args.eval.resolve(),
                          classifier=classifier, use_rules=args.rules)
        output = args.output or (DEFAULT_REPORT if args.rules else ROOT / "Data" / "Curated" / "intent_trained_eval_metrics.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, KeyError, OSError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps({"count": report["evaluation"]["count"], **report["metrics"],
                      "errors": len(report["errors"]), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
