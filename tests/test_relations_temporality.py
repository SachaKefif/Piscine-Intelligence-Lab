"""Relation evidence, uncertainty and temporal decision checks."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS, load_conversations, read_jsonl
from relations_temporality import (
    DEFAULT_RELATIONS,
    DEFAULT_TOPICS,
    classify_pair,
    current_decision,
    validate_relations,
    validate_topics,
)


class RelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conversations = load_conversations(DEFAULT_CONVERSATIONS)
        cls.units = {unit["unit_id"]: unit for unit in read_jsonl(DEFAULT_UNITS)}
        cls.relations = list(read_jsonl(DEFAULT_RELATIONS))
        cls.topics = validate_topics(cls.units, list(read_jsonl(DEFAULT_TOPICS)))

    def test_all_six_labels_have_source_evidence(self):
        labels = validate_relations(self.conversations, self.units, self.relations)
        self.assertEqual(labels, {"supports", "contradicts", "supersedes", "creates_action", "answers", "unrelated"})

    def test_unreviewed_pair_is_unknown_not_unrelated(self):
        result = classify_pair("Synthetic:Atlas-01:0-0:0", "Synthetic:Atlas-02:3-3:0", self.relations)
        self.assertIsNone(result["relation"])
        self.assertFalse(result["reviewed"])

    def test_conflict_is_symmetric(self):
        result = classify_pair("Synthetic:Atlas-02:4-4:0", "Synthetic:Atlas-02:7-7:0", self.relations)
        self.assertEqual(result["relation"], "contradicts")
        self.assertTrue(result["annotation_direction_reversed"])

    def test_rejects_unsupported_link(self):
        relations = deepcopy(self.relations)
        relations[0]["evidence"] = "Une citation inventée."
        with self.assertRaisesRegex(ValueError, "preuve du lien introuvable"):
            validate_relations(self.conversations, self.units, relations)

    def test_rejects_reversed_supersession(self):
        relations = deepcopy(self.relations)
        replacement = next(row for row in relations if row["relation"] == "supersedes")
        replacement["from_id"], replacement["to_id"] = replacement["to_id"], replacement["from_id"]
        with self.assertRaisesRegex(ValueError, "unité confirmée"):
            validate_relations(self.conversations, self.units, relations)

    def test_rejects_supersession_without_dates(self):
        units = deepcopy(self.units)
        units["Synthetic:Atlas-01:1-1:0"]["source_date"] = None
        with self.assertRaisesRegex(ValueError, "sans les dates"):
            validate_relations(self.conversations, units, self.relations)

    def test_current_budget_and_delivery_use_confirmed_evidence(self):
        budget = current_decision(self.topics["atlas_budget"], self.units, self.relations)
        delivery = current_decision(self.topics["atlas_delivery"], self.units, self.relations)
        self.assertEqual(budget["state"], "current")
        self.assertEqual(budget["decision"]["source_id"], "Synthetic:Atlas-02")
        self.assertEqual(delivery["state"], "current")
        self.assertIn("19 novembre", delivery["decision"]["evidence"])
        self.assertTrue(any(row["relation"] == "supersedes" for row in delivery["relations"]))

    def test_proposal_alone_is_not_a_current_decision(self):
        topic = {"topic_id": "proposal_only", "project_id": "Atlas",
                 "unit_ids": ["Synthetic:Atlas-01:1-1:0"]}
        result = current_decision(topic, self.units, self.relations)
        self.assertEqual(result["state"], "no_confirmed_decision")

    def test_later_date_alone_does_not_choose_between_decisions(self):
        units = deepcopy(self.units)
        second = deepcopy(units["Synthetic:Atlas-02:0-0:0"])
        second["unit_id"] = "Synthetic:Atlas-03:0-0:0"
        second["source_date"] = "2026-11-04"
        second["statement"] = "Un autre budget a été approuvé."
        units[second["unit_id"]] = second
        topic = deepcopy(self.topics["atlas_budget"])
        topic["unit_ids"].append(second["unit_id"])
        result = current_decision(topic, units, self.relations)
        self.assertEqual(result["state"], "ambiguous")
        self.assertIsNone(result["decision"])

    def test_contradiction_prevents_current_decision(self):
        relations = deepcopy(self.relations)
        relations.append({"from_id": "Synthetic:Atlas-02:7-7:0",
                          "to_id": "Synthetic:Atlas-02:0-0:0", "relation": "contradicts",
                          "evidence_source_id": "Synthetic:Atlas-02", "evidence": "Nova n'a pas confirmé le prix de 8 000 €."})
        result = current_decision(self.topics["atlas_budget"], self.units, relations)
        self.assertEqual(result["state"], "ambiguous")


if __name__ == "__main__":
    unittest.main()
