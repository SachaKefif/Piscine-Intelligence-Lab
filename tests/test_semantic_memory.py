"""EF-05 timeline, evidence and uncertainty checks."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS
from relations_temporality import DEFAULT_RELATIONS
from semantic_memory import DEFAULT_MEMORY_TOPICS, chronological_key, load_memory, topic_memory


class SemanticMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.relations, cls.topics = load_memory(
            DEFAULT_CONVERSATIONS, DEFAULT_UNITS, DEFAULT_RELATIONS, DEFAULT_MEMORY_TOPICS
        )

    def test_every_demo_topic_spans_two_conversations(self):
        self.assertEqual(len(self.topics), 4)
        for topic in self.topics.values():
            self.assertEqual(
                len({self.units[unit_id]["source_id"] for unit_id in topic["unit_ids"]}), 2
            )

    def test_delivery_marks_only_proven_replacement_obsolete(self):
        memory = topic_memory(self.topics["atlas_delivery"], self.units, self.relations)
        old, new = memory["history"]
        self.assertEqual(old["memory_state"], "obsolete")
        self.assertEqual(old["superseded_by"], [new["unit_id"]])
        self.assertEqual(new["memory_state"], "not_superseded")
        self.assertEqual(memory["current_decision"]["decision"]["unit_id"], new["unit_id"])
        self.assertEqual(memory["relations"][0]["relation"], "supersedes")

    def test_price_conflict_retains_both_sourced_claims(self):
        memory = topic_memory(self.topics["atlas_sensor_price"], self.units, self.relations)
        self.assertEqual(len(memory["contradictions"]), 1)
        self.assertEqual(memory["contradictions"][0]["state"], "unresolved")
        self.assertEqual(memory["obsolete_unit_ids"], [])
        claims = [item for item in memory["history"] if item["contradicts"]]
        self.assertEqual(len(claims), 2)
        self.assertTrue(all(item["evidence"] and item["source_id"] for item in claims))

    def test_later_date_alone_does_not_make_budget_proposal_obsolete(self):
        memory = topic_memory(self.topics["atlas_budget"], self.units, self.relations)
        self.assertEqual(memory["obsolete_unit_ids"], [])
        self.assertEqual(memory["history"][0]["status"], "proposed")
        self.assertEqual(memory["current_decision"]["state"], "current")

    def test_unknown_date_does_not_precede_known_date(self):
        known = self.units["Synthetic:Atlas-01:1-1:0"]
        undated = deepcopy(known)
        undated["source_date"] = None
        self.assertLess(chronological_key(known), chronological_key(undated))

    def test_unreviewed_relation_is_not_invented(self):
        memory = topic_memory(self.topics["atlas_procurement"], self.units, self.relations)
        self.assertEqual(memory["obsolete_unit_ids"], [])
        self.assertEqual(memory["contradictions"], [])
        self.assertFalse(any(row["relation"] == "supersedes" for row in memory["relations"]))


if __name__ == "__main__":
    unittest.main()
