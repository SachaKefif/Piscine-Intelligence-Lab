"""Checks for provenance and cross-conversation unit retrieval."""

import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS, SCHEMA_PATH, load_conversations, read_jsonl, validate_units
from search_semantic_units import UnitSearch, rank, result_rows


class SemanticUnitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conversations = load_conversations(DEFAULT_CONVERSATIONS)
        cls.units = list(read_jsonl(DEFAULT_UNITS))
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_two_sources_and_exact_evidence(self):
        sources = validate_units(self.conversations, self.units, self.schema)
        self.assertEqual(sources, {"Synthetic:Atlas-01", "Synthetic:Atlas-02"})

    def test_rejects_invented_evidence(self):
        units = deepcopy(self.units)
        units[0]["evidence"] = "Le budget de 40 000 € a été approuvé."
        with self.assertRaisesRegex(ValueError, "preuve introuvable"):
            validate_units(self.conversations, units, self.schema)

    def test_rejects_wrong_date(self):
        units = deepcopy(self.units)
        units[0]["source_date"] = "2026-10-21"
        with self.assertRaisesRegex(ValueError, "source_date différente"):
            validate_units(self.conversations, units, self.schema)

    def test_bm25_searches_both_conversations(self):
        search = UnitSearch(self.units, None, None, 16, dense=False)
        scores = search.scores("Quelles propositions et décisions concernent le budget Atlas ?")["bm25"]
        rows = result_rows(self.units, scores, top_k=5)
        sources = {row["source_id"] for row in rows}
        self.assertEqual(sources, {"Synthetic:Atlas-01", "Synthetic:Atlas-02"})
        self.assertTrue(all(row["evidence"] and row["unit_id"] for row in rows))
        self.assertEqual(len(rank(scores)), len(self.units))


if __name__ == "__main__":
    unittest.main()
