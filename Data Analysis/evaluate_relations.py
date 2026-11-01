"""Evaluate EF-04 proposals on manually labelled, held-out conversations."""

import argparse
import hashlib
import json
from pathlib import Path

from ingest_semantic_units import ROOT
from predict_relations import predict_pair


LABELS = ("supports", "contradicts", "supersedes", "creates_action", "answers", "unrelated")
DEFAULT_FIXTURE = ROOT / "Data" / "Curated" / "relation_eval_holdout.json"
DEFAULT_OUTPUT = ROOT / "Data" / "Curated" / "relation_eval_metrics.json"


def load_fixture(path):
    raw = path.read_bytes()
    fixture = json.loads(raw)
    conversations = {row["source_id"]: row for row in fixture["conversations"]}
    if len(conversations) != len(fixture["conversations"]) or len(conversations) < 2:
        raise ValueError("Il faut au moins deux conversations distinctes")
    units = {}
    for row in fixture["units"]:
        source_id, turn = row["source_id"], row["turn"]
        if source_id not in conversations or not isinstance(turn, int) or turn < 0:
            raise ValueError(f"Source ou tour invalide : {row}")
        conversation = conversations[source_id]
        if turn >= len(conversation["segments"]):
            raise ValueError(f"Tour hors limites : {row}")
        unit_id = f"{source_id}:{turn}"
        if unit_id in units:
            raise ValueError(f"Unité dupliquée : {unit_id}")
        units[unit_id] = {"unit_id": unit_id, "source_id": source_id,
                          "source_date": conversation["source_date"], "start_turn": turn,
                          "end_turn": turn, "project_id": conversation["project_id"],
                          "evidence": conversation["segments"][turn],
                          "type": row["type"], "status": row["status"]}
    pairs = fixture["pairs"]
    if not pairs or len({(p["from"], p["to"]) for p in pairs}) != len(pairs):
        raise ValueError("Paires absentes ou dupliquées")
    if {p["gold"] for p in pairs} != set(LABELS):
        raise ValueError("Le jeu de contrôle doit couvrir les six classes")
    for pair in pairs:
        if pair["from"] not in units or pair["to"] not in units or pair["from"] == pair["to"]:
            raise ValueError(f"Paire invalide : {pair}")
    return units, pairs, conversations, hashlib.sha256(raw).hexdigest()


def compute_metrics(gold, predicted):
    if len(gold) != len(predicted) or not gold:
        raise ValueError("Listes de labels vides ou de tailles différentes")
    if any(label not in LABELS for label in gold):
        raise ValueError("Label de référence inconnu")
    if any(label not in (*LABELS, None) for label in predicted):
        raise ValueError("Prédiction inconnue")
    matrix = {label: {prediction: 0 for prediction in (*LABELS, "abstain")} for label in LABELS}
    for actual, guess in zip(gold, predicted):
        matrix[actual][guess or "abstain"] += 1
    by_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        by_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return {"count": len(gold), "coverage": sum(p is not None for p in predicted) / len(predicted),
            "accuracy_with_abstention": sum(a == p for a, p in zip(gold, predicted)) / len(gold),
            "macro_f1": sum(item["f1"] for item in by_class.values()) / len(LABELS),
            "by_class": by_class, "confusion_matrix": matrix}


def evaluate(path):
    units, pairs, conversations, digest = load_fixture(path)
    details = []
    for pair in pairs:
        proposal = predict_pair(units[pair["from"]], units[pair["to"]])
        source = proposal["evidence_source_id"]
        if source is not None and (source not in conversations or
                                   proposal["evidence"] not in conversations[source]["segments"]):
            raise ValueError("Prédiction sans preuve exacte dans la transcription")
        details.append({"from_id": pair["from"], "to_id": pair["to"], "gold": pair["gold"],
                        "predicted": proposal["relation"], "reason": proposal["reason"]})
    return {"fixture": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "fixture_sha256": digest, "method": "conservative_rules_v1", "split": "held_out_synthetic_project",
            "metrics": compute_metrics([p["gold"] for p in pairs], [d["predicted"] for d in details]),
            "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        report = evaluate(args.fixture.resolve())
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"count": report["metrics"]["count"], "macro_f1": report["metrics"]["macro_f1"],
                      "coverage": report["metrics"]["coverage"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
