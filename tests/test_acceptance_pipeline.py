"""Reproduce all eight acceptance scenarios through the complete CLI router."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from evaluate_acceptance import DEFAULT_CASES, evaluate
from ingest_semantic_units import read_jsonl
from semantic_agent import SemanticPipeline


class AcceptancePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = SemanticPipeline(method="bm25")
        cls.cases = list(read_jsonl(DEFAULT_CASES))
        cls.details, cls.outputs = evaluate(cls.pipeline, cls.cases)
        cls.by_id = {row["id"]: row["response"] for row in cls.outputs}

    def test_eight_frozen_scenarios_pass_with_verified_retrieval(self):
        self.assertEqual(len(self.details), 8)
        self.assertTrue(all(row["passed"] for row in self.details),
                        [row for row in self.details if not row["passed"]])
        self.assertTrue(all(output["retrieval"]["method"] == "bm25"
                            for output in self.by_id.values()))

    def test_dossier_commitment_and_budget_recipient_have_exact_sources(self):
        commitment = self.by_id["scenario04"]["sections"]["answer"][0]
        self.assertEqual(commitment["owner"], "Marc")
        self.assertEqual(commitment["due_date"], "2026-11-10")
        dossier = self.by_id["scenario05"]["sections"]
        self.assertEqual(dossier["context"][0]["source_unit_ids"],
                         ["Synthetic:Atlas-02:8-8:0"])
        self.assertTrue(dossier["risks"])
        mail = self.by_id["scenario06"]
        self.assertEqual(mail["recipient"]["name"], "Camille")
        self.assertEqual(mail["recipient"]["source_unit_ids"],
                         ["Synthetic:Atlas-02:9-9:0"])
        self.assertIsNone(mail["parameters"]["to_email"])
        self.assertFalse(mail["executed"])

    def test_ambiguous_and_hypothetical_requests_have_no_operation(self):
        for identifier in ("scenario07", "scenario08"):
            with self.subTest(identifier=identifier):
                result = self.by_id[identifier]
                self.assertEqual(result["status"], "clarify")
                self.assertIsNone(result.get("tool_candidate"))


if __name__ == "__main__":
    unittest.main()
