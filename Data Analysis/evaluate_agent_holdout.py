"""Evaluate the frozen mini-agent on two QMSum test meetings and manual requests."""

import argparse
import hashlib
import json
from pathlib import Path

from generate_response import ResponseEngine, validate_response
from ingest_semantic_units import ROOT, read_jsonl
from prepare_agent_holdout import OUTPUT as CONVERSATIONS, RAW, MEETINGS, prepare
from search_semantic_units import DEFAULT_MODEL, METHODS
from semantic_agent import SemanticPipeline
from simulate_actions import validate_draft


CURATED = ROOT / "Data/Curated"
UNITS = CURATED / "qmsum_agent_holdout_units.jsonl"
RELATIONS = CURATED / "qmsum_agent_holdout_relations.jsonl"
TOPICS = CURATED / "qmsum_agent_holdout_topics.jsonl"
CASES = CURATED / "qmsum_agent_holdout_cases.jsonl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(method="bm25", model=DEFAULT_MODEL, device="cpu", *, meetings=MEETINGS,
             conversations=CONVERSATIONS, units=UNITS, relations=RELATIONS,
             topics=TOPICS, cases_path=CASES, expected_cases=12, evaluation_note=None,
             frozen_manifest=None, raw_dir=RAW, request_description=None):
    input_sha256 = {
        **{f"raw_{meeting}": digest(raw_dir / f"{meeting}.json") for meeting in meetings},
        "units": digest(units), "relations": digest(relations),
        "topics": digest(topics), "cases": digest(cases_path),
        "model_weights": digest(Path(model) / "model.safetensors") if method != "bm25" else None,
    }
    if frozen_manifest:
        frozen = json.loads(frozen_manifest.read_text(encoding="utf-8"))
        for key, expected in frozen["input_sha256"].items():
            if input_sha256.get(key) != expected:
                raise ValueError(f"Frozen fixture mismatch: {key}")
        expected_model = frozen.get("frozen_model_weights_sha256")
        if method != "bm25" and expected_model and input_sha256["model_weights"] != expected_model:
            raise ValueError("Frozen model weights mismatch")
    prepare(conversations, meetings=meetings, units=units, raw_dir=raw_dir)
    engine = ResponseEngine(conversations, units, relations, topics)
    pipeline = SemanticPipeline(engine=engine, method=method, model_path=model, device=device)
    cases = list(read_jsonl(cases_path))
    if len(cases) != expected_cases or len({case["id"] for case in cases}) != expected_cases:
        raise ValueError(f"Le jeu tenu à l'écart doit contenir {expected_cases} demandes distinctes")
    details = []
    outputs = []
    for case in cases:
        result = pipeline.run(case["request"])
        if "sections" in result:
            structure_ok = validate_response(result, engine.units, engine.relations)
            cited = {uid for section in result["sections"].values()
                     for claim in section for uid in claim["source_unit_ids"]}
            sections = result["sections"]
        else:
            structure_ok = validate_draft(result, engine.units)
            cited = set(result["source_unit_ids"])
            sections = {}
        required = set(case["required_unit_ids"])
        allowed = set(case["allowed_unit_ids"])
        checks = {
            "intent": result["intent"] == case["expected_intent"],
            "status": result["status"] == case["expected_status"],
            "required_facts": required <= cited,
            "citation_relevance": cited <= allowed,
            "required_sections": all(sections.get(name) for name in case.get("required_sections", [])),
            "source_contract": bool(structure_ok),
            "no_external_effect": result.get("executed") is False if "executed" in result else True,
        }
        if "expected_recipient" in case:
            checks["recipient"] = (result.get("recipient") or {}).get("name") == case["expected_recipient"]
        details.append({
            "id": case["id"], "request": case["request"],
            "expected_intent": case["expected_intent"], "actual_intent": result["intent"],
            "expected_status": case["expected_status"], "actual_status": result["status"],
            "expected_unit_ids": sorted(required), "cited_unit_ids": sorted(cited),
            "missing_unit_ids": sorted(required - cited),
            "irrelevant_unit_ids": sorted(cited - allowed), "checks": checks,
            "passed": all(checks.values()),
        })
        outputs.append({"id": case["id"], "response": result})
    expected_facts = sum(len(case["required_unit_ids"]) for case in cases)
    covered_facts = sum(len(set(case["required_unit_ids"]) & set(detail["cited_unit_ids"]))
                        for case, detail in zip(cases, details))
    expected_clarifications = [row for row in details if row["expected_status"] == "clarify"]
    actual_clarifications = [row for row in details if row["actual_status"] == "clarify"]
    cases_with_citations = [row for row in details if row["cited_unit_ids"]]
    metrics = {
        "dataset": f"QMSum test, {', '.join(meetings)}; transcript dates unknown",
        "requests": request_description or f"{len(cases)} manually written French requests; not a sample of user traffic",
        "evaluation_note": evaluation_note or (
            "Frozen fixture: no agent rule or model was selected on these cases."
            if frozen_manifest else
            "The initial BM25 run scored 10/12; h11 prompted a missing-owner safety fix, "
            "and undated QMSum decisions prompted a temporal-label fix. Final scores are "
            "post-fix diagnostics, not an untouched blind test."),
        "method": method, "cases": len(details), "passed": sum(row["passed"] for row in details),
        "case_accuracy": round(sum(row["passed"] for row in details) / len(details), 4),
        "intent_accuracy": round(sum(row["checks"]["intent"] for row in details) / len(details), 4),
        "status_accuracy": round(sum(row["checks"]["status"] for row in details) / len(details), 4),
        "fact_coverage": round(covered_facts / expected_facts, 4),
        "cases_with_citations": len(cases_with_citations),
        "citation_relevance_rate": round(sum(row["checks"]["citation_relevance"] for row in cases_with_citations)
                                         / len(cases_with_citations), 4) if cases_with_citations else None,
        "clarification_recall": round(sum(row["actual_status"] == "clarify" for row in expected_clarifications)
                                      / len(expected_clarifications), 4) if expected_clarifications else None,
        "clarification_precision": round(sum(row["expected_status"] == "clarify" for row in actual_clarifications)
                                         / len(actual_clarifications), 4) if actual_clarifications else None,
        "source_contract_rate": round(sum(row["checks"]["source_contract"] for row in details) / len(details), 4),
        "input_sha256": input_sha256,
        "details": details,
    }
    return metrics, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="bm25")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--meetings", nargs="+", default=MEETINGS)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--conversations", type=Path, default=CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=UNITS)
    parser.add_argument("--relations", type=Path, default=RELATIONS)
    parser.add_argument("--topics", type=Path, default=TOPICS)
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--expected-cases", type=int, default=12)
    parser.add_argument("--evaluation-note")
    parser.add_argument("--request-description")
    parser.add_argument("--frozen-manifest", type=Path)
    parser.add_argument("--metrics", type=Path, default=CURATED / "qmsum_agent_holdout_metrics.json")
    parser.add_argument("--outputs", type=Path, default=CURATED / "qmsum_agent_holdout_outputs.jsonl")
    args = parser.parse_args()
    metrics, outputs = evaluate(args.method, args.model, args.device,
                                meetings=args.meetings, conversations=args.conversations,
                                units=args.units, relations=args.relations, topics=args.topics,
                                cases_path=args.cases, expected_cases=args.expected_cases,
                                evaluation_note=args.evaluation_note,
                                frozen_manifest=args.frozen_manifest, raw_dir=args.raw_dir,
                                request_description=args.request_description)
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.outputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
                            encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "details"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
