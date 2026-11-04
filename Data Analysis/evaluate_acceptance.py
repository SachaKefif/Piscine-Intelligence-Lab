"""Replay the eight specified end-to-end Atlas scenarios with inspectable evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from generate_response import validate_response
from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS, ROOT, read_jsonl
from relations_temporality import DEFAULT_RELATIONS
from search_semantic_units import DEFAULT_MODEL, E5_WEIGHT, METHODS, RRF_K
from semantic_agent import SemanticPipeline
from semantic_memory import DEFAULT_MEMORY_TOPICS
from simulate_actions import validate_draft


DEFAULT_CASES = ROOT / "Data" / "Curated" / "atlas_acceptance_cases.jsonl"
DEFAULT_METRICS = ROOT / "Data" / "Curated" / "atlas_acceptance_metrics.json"
DEFAULT_OUTPUTS = ROOT / "Data" / "Curated" / "atlas_acceptance_outputs.jsonl"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_case(case, output, pipeline):
    issues = []
    if output["intent"] != case["expected_intent"]:
        issues.append("intent")
    if output["status"] != case["expected_status"]:
        issues.append("status")
    retrieval = output.get("retrieval", {})
    if retrieval.get("method") != pipeline.method or not retrieval.get("results"):
        issues.append("retrieval_missing")
    for row in retrieval.get("results", []):
        unit = pipeline.engine.units.get(row["unit_id"])
        if (unit is None or row["evidence"] != unit["evidence"]
                or row["source_id"] != unit["source_id"]
                or not math.isfinite(row["score"])):
            issues.append("retrieval_provenance")
            break
    if "sections" in output:
        validate_response(output, pipeline.engine.units, pipeline.engine.relations)
        sections = output["sections"]
        if output["status"] == "clarify" and (output["references"] or any(sections.values())):
            issues.append("clarification_with_claims")
        cited = {uid for claims in sections.values() for claim in claims
                 for uid in claim["source_unit_ids"]}
        relations = {claim.get("relation") for claims in sections.values() for claim in claims
                     if claim["kind"] == "reviewed_relation"}
        current = {uid for section in ("answer", "decisions") for claim in sections[section]
                   if claim.get("role") == "current_decision" for uid in claim["source_unit_ids"]}
        for section in case.get("required_sections", []):
            if not sections.get(section):
                issues.append(f"section:{section}")
        if set(case.get("required_relations", [])) - relations:
            issues.append("reviewed_relations")
        if set(case.get("forbidden_current_unit_ids", [])) & current:
            issues.append("obsolete_as_current")
    else:
        validate_draft(output, pipeline.engine.units)
        cited = set(output["source_unit_ids"])
        if "expected_recipient" in case:
            recipient = output.get("recipient") or {}
            if recipient.get("name") != case["expected_recipient"]:
                issues.append("recipient_name")
            if recipient.get("email") != case["expected_email"]:
                issues.append("recipient_address")
            if recipient.get("status") != case["expected_recipient_status"]:
                issues.append("recipient_resolution")
        if "expected_executed" in case and output["executed"] != case["expected_executed"]:
            issues.append("external_effect")
    if set(case["required_unit_ids"]) - cited:
        issues.append("required_evidence")
    if "expected_tool_candidate" in case and output.get("tool_candidate") != case["expected_tool_candidate"]:
        issues.append("operation_candidate")
    if ("expected_clarification_contains" in case and
            case["expected_clarification_contains"] not in (output.get("message") or "")):
        issues.append("clarification_message")
    return {"id": case["id"], "description": case["description"],
            "passed": not issues, "issues": issues,
            "intent": output["intent"], "status": output["status"],
            "cited_unit_ids": sorted(cited),
            "retrieved_unit_ids": [row["unit_id"] for row in retrieval.get("results", [])]}


def evaluate(pipeline, cases):
    if len(cases) != 8 or len({row["id"] for row in cases}) != 8:
        raise ValueError("La recette attend huit scénarios distincts")
    details = []
    outputs = []
    for case in cases:
        output = pipeline.run(case["request"])
        details.append(inspect_case(case, output, pipeline))
        outputs.append({"id": case["id"], "response": output})
    return details, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cases = list(read_jsonl(args.cases))
    pipeline = SemanticPipeline(method=args.method, model_path=args.model, device=args.device)
    details, outputs = evaluate(pipeline, cases)
    report = {
        "fixture": str(args.cases.relative_to(ROOT)) if args.cases.is_relative_to(ROOT) else str(args.cases),
        "fixture_sha256": sha256(args.cases),
        "method": args.method, "cases": len(details),
        "passed": sum(row["passed"] for row in details),
        "pass_rate": round(sum(row["passed"] for row in details) / len(details), 4),
        "inputs_sha256": {
            "conversations": sha256(DEFAULT_CONVERSATIONS), "units": sha256(DEFAULT_UNITS),
            "relations": sha256(DEFAULT_RELATIONS), "topics": sha256(DEFAULT_MEMORY_TOPICS),
            "model_weights": sha256(args.model / "model.safetensors") if args.method != "bm25" else None,
        },
        "retrieval_config": {"rrf_k": RRF_K, "e5_weight": E5_WEIGHT,
                             "top_k_display": pipeline.top_k},
        "note": "Recette sur conversations Atlas synthétiques ; ne mesure pas la généralisation externe.",
        "details": details,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.outputs.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outputs),
                            encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
