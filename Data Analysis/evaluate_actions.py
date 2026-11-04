"""Development diagnostics for EF-08 drafts and EF-09 recipient decisions."""

import argparse
import json
from pathlib import Path

from generate_response import ResponseEngine
from ingest_semantic_units import ROOT, read_jsonl
from resolve_recipient import resolve_recipient, validate_resolution
from simulate_actions import ActionSimulator, validate_draft


RECIPIENT_CASES = ROOT / "Data" / "Curated" / "recipient_eval_cases.jsonl"
ACTION_CASES = ROOT / "Data" / "Curated" / "action_eval_cases.jsonl"
BOREAL_CONVERSATIONS = ROOT / "Data" / "Synthetic" / "boreal_action_conversations.jsonl"
BOREAL_UNITS = ROOT / "Data" / "Curated" / "boreal_action_units.jsonl"
BOREAL_RELATIONS = ROOT / "Data" / "Curated" / "boreal_action_relations.jsonl"
BOREAL_TOPICS = ROOT / "Data" / "Curated" / "boreal_action_topics.jsonl"
BOREAL_RECIPIENT_CASES = ROOT / "Data" / "Curated" / "recipient_boreal_eval_cases.jsonl"
BOREAL_ACTION_CASES = ROOT / "Data" / "Curated" / "action_boreal_eval_cases.jsonl"


def ratio(numerator, denominator):
    return round(numerator / denominator, 4) if denominator else None


def evaluate_recipients(engine, cases):
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Scénarios de destinataires absents ou dupliqués")
    details = []
    for case in cases:
        output = resolve_recipient(case["request"], engine=engine, as_of=case.get("as_of"))
        evidence_valid = validate_resolution(output, engine=engine, request=case["request"])
        details.append({"id": case["id"], "expected_status": case["expected_status"],
                        "actual_status": output["status"],
                        "expected_name": case["expected_name"], "actual_name": output["name"],
                        "evidence_valid": evidence_valid,
                        "correct": (output["status"] == case["expected_status"]
                                    and output["name"] == case["expected_name"])})
    predicted = [row for row in details if row["actual_name"] is not None]
    role_predicted = [row for row in details if row["actual_status"] == "resolved"]
    no_gold = [row for row in details if row["expected_name"] is None]
    resolvable = [row for row in details if row["expected_name"] is not None]
    return {
        "cases": len(details),
        "precision_at_1": ratio(sum(row["actual_name"] == row["expected_name"]
                                    for row in predicted), len(predicted)),
        "role_precision_at_1": ratio(sum(row["actual_name"] == row["expected_name"]
                                         for row in role_predicted), len(role_predicted)),
        "false_recipients": sum(row["actual_name"] != row["expected_name"]
                                for row in predicted),
        "justified_abstention_rate": ratio(sum(row["actual_name"] is None for row in no_gold),
                                           len(no_gold)),
        "resolvable_recall": ratio(sum(row["actual_name"] == row["expected_name"]
                                       for row in resolvable), len(resolvable)),
        "status_accuracy": ratio(sum(row["correct"] for row in details), len(details)),
        "evidence_integrity": ratio(sum(row["evidence_valid"] for row in details), len(details)),
        "details": details,
    }


def evaluate_drafts(simulator, cases):
    if not cases or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Scénarios d'actions absents ou dupliqués")
    details = []
    for case in cases:
        output = simulator.prepare(case["request"])
        valid = validate_draft(output, simulator.engine.units)
        details.append({"id": case["id"], "expected_intent": case["expected_intent"],
                        "actual_intent": output["intent"],
                        "expected_status": case["expected_status"],
                        "actual_status": output["status"],
                        "expected_context_id": case.get("expected_context_id"),
                        "actual_context_id": output.get("context_unit_id"),
                        "invariant_valid": valid and output["executed"] is False,
                        "correct": (output["intent"] == case["expected_intent"]
                                    and output["status"] == case["expected_status"]
                                    and output.get("context_unit_id") == case.get("expected_context_id"))})
    return {"cases": len(details),
            "case_accuracy": ratio(sum(row["correct"] for row in details), len(details)),
            "safe_output_rate": ratio(sum(row["invariant_valid"] for row in details), len(details)),
            "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipient-cases", type=Path, default=RECIPIENT_CASES)
    parser.add_argument("--action-cases", type=Path, default=ACTION_CASES)
    parser.add_argument("--boreal-recipient-cases", type=Path, default=BOREAL_RECIPIENT_CASES)
    parser.add_argument("--boreal-action-cases", type=Path, default=BOREAL_ACTION_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    engine = ResponseEngine()
    boreal = ResponseEngine(BOREAL_CONVERSATIONS, BOREAL_UNITS, BOREAL_RELATIONS,
                            BOREAL_TOPICS)
    result = {
        "note": "Atlas : développement. Boréal : autre projet synthétique figé après le code initial ; pas de test humain indépendant.",
        "recipient": evaluate_recipients(engine, list(read_jsonl(args.recipient_cases))),
        "actions": evaluate_drafts(ActionSimulator(engine), list(read_jsonl(args.action_cases))),
        "recipient_boreal": evaluate_recipients(
            boreal, list(read_jsonl(args.boreal_recipient_cases))),
        "actions_boreal": evaluate_drafts(
            ActionSimulator(boreal), list(read_jsonl(args.boreal_action_cases))),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {k: v for k, v in value.items() if k != "details"}
                      if isinstance(value, dict) else value for key, value in result.items()},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
