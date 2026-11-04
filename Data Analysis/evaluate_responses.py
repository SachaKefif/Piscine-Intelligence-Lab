"""Evaluate EF-07 on manually specified Atlas scenarios (not an independent corpus)."""

import argparse
import json
from pathlib import Path

from generate_response import ResponseEngine, validate_claim, validate_response
from ingest_semantic_units import ROOT, read_jsonl


DEFAULT_CASES = ROOT / "Data" / "Curated" / "response_eval_cases.jsonl"


def evaluate(engine, cases):
    if not cases:
        raise ValueError("Aucun scénario EF-07")
    seen = set()
    rows = []
    expected_total = covered_total = claims_total = invalid_claims = 0
    for case in cases:
        identifier = case.get("id")
        if not identifier or identifier in seen:
            raise ValueError(f"Identifiant de scénario absent ou dupliqué : {identifier!r}")
        seen.add(identifier)
        expected = set(case["expected_unit_ids"])
        if not expected <= engine.units.keys():
            raise ValueError(f"{identifier} : unités attendues absentes")
        output = engine.generate(case["request"])
        try:
            validate_response(output, engine.units, engine.relations)
            provenance_ok = True
        except ValueError:
            provenance_ok = False
        factual_claims = [claim for section in output["sections"].values() for claim in section]
        claims_total += len(factual_claims)
        for claim in factual_claims:
            try:
                validate_claim(claim, engine.units, engine.relations)
            except ValueError:
                invalid_claims += 1
        cited = {uid for claim in factual_claims for uid in claim["source_unit_ids"]}
        found = expected & cited
        expected_total += len(expected)
        covered_total += len(found)
        relations = {claim.get("relation") for claim in factual_claims
                     if claim["kind"] == "reviewed_relation"}
        expected_relations = set(case.get("expected_relations", []))
        rows.append({
            "id": identifier, "request": case["request"],
            "expected_status": case["expected_status"], "actual_status": output["status"],
            "status_correct": output["status"] == case["expected_status"],
            "expected_unit_ids": sorted(expected), "covered_unit_ids": sorted(found),
            "missing_unit_ids": sorted(expected - found),
            "expected_relations": sorted(expected_relations),
            "missing_relations": sorted(expected_relations - relations),
            "provenance_ok": provenance_ok,
        })
    return {
        "note": "Scénarios synthétiques Atlas annotés pour le développement ; aucune mesure de généralisation externe.",
        "cases": len(rows), "expected_facts": expected_total,
        "fact_coverage": round(covered_total / expected_total, 4) if expected_total else None,
        "status_accuracy": round(sum(row["status_correct"] for row in rows) / len(rows), 4),
        "relation_coverage": round(sum(not row["missing_relations"] for row in rows) / len(rows), 4),
        "citation_integrity": round(sum(row["provenance_ok"] for row in rows) / len(rows), 4),
        "unsupported_claims_structural": invalid_claims,
        "claim_count": claims_total,
        "details": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(ResponseEngine(), list(read_jsonl(args.cases)))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "details"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
