"""Held-out relation metrics and conservative inference checks."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from evaluate_relations import DEFAULT_FIXTURE, LABELS, compute_metrics, evaluate, load_fixture
from ingest_semantic_units import DEFAULT_UNITS, read_jsonl
from predict_relations import predict_pair


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.units, cls.pairs, _, _ = load_fixture(DEFAULT_FIXTURE)

    def test_holdout_has_two_independent_projects_and_six_labels(self):
        self.assertEqual({u["project_id"] for u in self.units.values()}, {"Boréal", "Cèdre"})
        self.assertEqual({p["gold"] for p in self.pairs}, set(LABELS))
        self.assertEqual(len(self.pairs), 12)

    def test_predicted_evidence_is_exact_source_quote_and_unreviewed(self):
        report = evaluate(DEFAULT_FIXTURE)
        self.assertEqual(report["metrics"]["count"], 12)
        for pair in self.pairs:
            prediction = predict_pair(self.units[pair["from"]], self.units[pair["to"]])
            self.assertFalse(prediction["reviewed"])
            if prediction["relation"] not in {None, "unrelated"}:
                proof = next(u for u in self.units.values() if u["source_id"] == prediction["evidence_source_id"]
                             and u["evidence"] == prediction["evidence"])
                self.assertIsNotNone(proof)

    def test_ambiguous_same_topic_abstains(self):
        decision = self.units["Synthetic:Boreal-02:0"]
        action = deepcopy(self.units["Synthetic:Boreal-02:4"])
        action["evidence"] = "Commander les batteries après étude."
        result = predict_pair(decision, action)
        self.assertIsNone(result["relation"])

    def test_different_amount_does_not_support_or_contradict(self):
        answer = self.units["Synthetic:Boreal-02:2"]
        decision = deepcopy(self.units["Synthetic:Boreal-02:0"])
        decision["evidence"] = "Le financement de 40 000 € pour Boréal est approuvé."
        self.assertIsNone(predict_pair(answer, decision)["relation"])
        negative = self.units["Synthetic:Boreal-02:3"]
        positive = deepcopy(self.units["Synthetic:Boreal-01:2"])
        positive["evidence"] = "Solis confirme le tarif de 6 000 € pour les batteries."
        self.assertIsNone(predict_pair(negative, positive)["relation"])

    def test_action_must_name_the_actual_cause(self):
        units = {row["unit_id"]: row for row in read_jsonl(DEFAULT_UNITS)}
        commitment = units["Synthetic:Atlas-02:2-2:0"]
        action = units["Synthetic:Atlas-02:6-6:0"]
        self.assertIsNone(predict_pair(commitment, action)["relation"])

    def test_metrics_include_abstention_in_confusion_matrix(self):
        metrics = compute_metrics(list(LABELS), [None, *LABELS[1:]])
        self.assertEqual(metrics["confusion_matrix"]["supports"]["abstain"], 1)
        self.assertEqual(metrics["by_class"]["supports"]["recall"], 0)
        self.assertAlmostEqual(metrics["coverage"], 5 / 6)


if __name__ == "__main__":
    unittest.main()
