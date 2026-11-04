"""Evaluate EF-06/09 on frozen French MASSIVE recipient and non-recipient requests."""

import hashlib
import json
from pathlib import Path

from generate_response import ResponseEngine
from parse_intent import parse_intent
from resolve_recipient import resolve_recipient, validate_resolution


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "Data/Curated/massive_fr_recipient_cases.jsonl"
MANIFEST = ROOT / "Data/Curated/massive_fr_recipient_manifest.json"
METRICS = ROOT / "Data/Curated/massive_fr_recipient_metrics.json"
OUTPUTS = ROOT / "Data/Curated/massive_fr_recipient_outputs.jsonl"


def evaluate():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(CASES.read_bytes()).hexdigest() != manifest["cases_sha256"]:
        raise ValueError("Frozen MASSIVE cases changed")
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines()]
    if len(cases) != manifest["cases"]:
        raise ValueError("Frozen MASSIVE case count changed")
    engine = ResponseEngine()
    details = []
    for case in cases:
        request = case["request"]
        parsed = parse_intent(request)
        recipient = resolve_recipient(request, engine=engine)
        validate_resolution(recipient, engine=engine, request=request)
        expected = case["expected_person"]
        chosen = recipient["name"]
        correct = (chosen is not None and chosen.casefold() == expected.casefold()) if expected else chosen is None
        desired_intent = "answer" if case["category"] == "person_mentioned_in_query" else "draft_email"
        details.append({
            "id": case["id"], "category": case["category"], "request": request,
            "expected_person": expected, "actual_person": chosen,
            "recipient_status": recipient["status"], "recipient_reason": recipient["reason"],
            "expected_intent": desired_intent, "actual_intent": parsed["intent"],
            "recipient_correct": correct, "intent_correct": parsed["intent"] == desired_intent,
            "email_remained_null": recipient["email"] is None,
        })
    positives = [row for row in details if row["expected_person"]]
    negatives = [row for row in details if not row["expected_person"]]
    chosen = [row for row in details if row["actual_person"]]
    metrics = {
        "dataset": "MASSIVE fr-FR test; public human-localized and reviewed utterances",
        "fixture_sha256": manifest["cases_sha256"], "source_sha256": manifest["source_sha256"],
        "cases": len(details), "categories": manifest["categories"],
        "recipient_case_accuracy": round(sum(row["recipient_correct"] for row in details) / len(details), 4),
        "direct_named_correct_at_1": sum(row["recipient_correct"] for row in positives),
        "direct_named_total": len(positives),
        "unjustified_abstentions": sum(row["actual_person"] is None for row in positives),
        "wrong_named_recipients": sum(row["actual_person"] is not None and not row["recipient_correct"]
                                      for row in positives),
        "precision_at_1": round(sum(row["recipient_correct"] for row in chosen) / len(chosen), 4)
                          if chosen else None,
        "appropriate_abstention_rate": round(sum(row["recipient_correct"] for row in negatives)
                                             / len(negatives), 4),
        "false_recipients_on_negatives": sum(row["actual_person"] is not None for row in negatives),
        "intent_accuracy": round(sum(row["intent_correct"] for row in details) / len(details), 4),
        "email_null_rate": round(sum(row["email_remained_null"] for row in details) / len(details), 4),
        "limitations": [
            "Person slots plus direct email syntax are used as recipient targets; MASSIVE has no recipient-role label.",
            "No email contact directory or meeting responsibility attribution is supplied by MASSIVE.",
            "The agent's Atlas evidence base is used only to satisfy the resolver interface; requests are unrelated to Atlas.",
        ],
        "details": details,
    }
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUTS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
                       encoding="utf-8")
    return {key: value for key, value in metrics.items() if key != "details"}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
