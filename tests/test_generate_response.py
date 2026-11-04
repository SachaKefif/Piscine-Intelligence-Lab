"""EF-07: factual provenance, temporal status and missing-information checks."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from generate_response import ResponseEngine, render_markdown, validate_response


class ResponseEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ResponseEngine()

    @staticmethod
    def parsed(intent, subject=None, reason=None):
        return {"intent": intent, "subject": subject, "clarification_reason": reason}

    def test_current_budget_decision_is_sourced(self):
        result = self.engine.generate(
            "Quelle est la décision actuelle sur le budget Atlas ?",
            parsed=self.parsed("answer", "budget Atlas"))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["topic_ids"], ["atlas_budget"])
        self.assertEqual(result["sections"]["answer"][0]["role"], "current_decision")
        self.assertEqual(result["sections"]["answer"][0]["source_unit_ids"],
                         ["Synthetic:Atlas-02:0-0:0"])
        self.assertTrue(validate_response(result, self.engine.units, self.engine.relations))

    def test_proposed_delivery_is_history_not_current_decision(self):
        result = self.engine.generate(
            "Prépare un dossier sur la livraison Atlas.",
            parsed=self.parsed("build_dossier", "livraison Atlas"))
        chronology = result["sections"]["chronology"]
        self.assertEqual([row["status"] for row in chronology], ["proposed", "confirmed"])
        self.assertEqual(chronology[0]["role"], "obsolete")
        self.assertEqual([row["source_unit_ids"][0] for row in result["sections"]["decisions"]],
                         ["Synthetic:Atlas-02:1-1:0"])
        self.assertEqual([row["relation"] for row in result["sections"]["changes"]],
                         ["supersedes"])

    def test_unresolved_price_conflict_is_not_presented_as_fact(self):
        result = self.engine.generate(
            "Quel est le prix des capteurs Nova pour Atlas ?",
            parsed=self.parsed("answer", "capteurs Atlas"))
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(len(result["sections"]["answer"]), 2)
        self.assertEqual(result["sections"]["open_points"][0]["relation"], "contradicts")
        dossier = self.engine.generate(
            "Prépare un dossier sur le prix des capteurs Nova pour Atlas.",
            parsed=self.parsed("build_dossier", "capteurs Atlas"))
        self.assertEqual(dossier["sections"]["facts"], [])
        self.assertTrue(any(row.get("relation") == "contradicts"
                            for row in dossier["sections"]["open_points"]))

    def test_asserted_fact_needs_corroboration_before_entering_facts(self):
        dossier = self.engine.generate(
            "Construis un dossier Atlas.",
            parsed=self.parsed("clarify", None, "missing_subject"))
        fact_ids = {uid for row in dossier["sections"]["facts"]
                    for uid in row["source_unit_ids"]}
        self.assertEqual(fact_ids, {"Synthetic:Atlas-02:5-5:0", "Synthetic:Atlas-02:8-8:0"})

    def test_historical_question_can_use_obsolete_proposal(self):
        result = self.engine.generate(
            "Quelle date avait été proposée pour la livraison Atlas ?",
            parsed=self.parsed("answer", "livraison Atlas"))
        self.assertEqual(result["sections"]["answer"][0]["role"], "historical_proposal")
        self.assertEqual(result["sections"]["answer"][0]["source_unit_ids"],
                         ["Synthetic:Atlas-01:1-1:0"])

    def test_integrated_routing_keeps_explicit_information_question(self):
        result = self.engine.generate("Quel est le prix confirmé des capteurs Nova pour Atlas ?")
        self.assertEqual(result["intent"], "answer")
        self.assertEqual(result["status"], "uncertain")
        result = self.engine.generate("Qui doit envoyer la commande des capteurs Atlas et quand ?")
        self.assertEqual(result["intent"], "answer")
        self.assertEqual(result["status"], "ok")

    def test_project_dossier_covers_every_topic_and_named_owner(self):
        result = self.engine.generate(
            "Construis un dossier Atlas.",
            parsed=self.parsed("clarify", None, "missing_subject"))
        self.assertEqual(result["intent"], "build_dossier")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["topic_ids"]), 4)
        self.assertEqual(len(result["sections"]["chronology"]), 16)
        self.assertEqual(result["sections"]["context"][0]["source_unit_ids"],
                         ["Synthetic:Atlas-02:8-8:0"])
        self.assertTrue(result["sections"]["risks"])
        self.assertEqual({row["owner"] for row in result["sections"]["responsibles"]},
                         {"Camille", "Julie", "Marc"})
        self.assertTrue(all(row["source_unit_ids"] for section in result["sections"].values()
                            for row in section))

    def test_comparison_only_reports_reviewed_change(self):
        result = self.engine.generate(
            "Compare les deux réunions sur la livraison Atlas.",
            parsed=self.parsed("compare", "livraison Atlas"))
        self.assertEqual(result["intent"], "compare")
        self.assertEqual([row["relation"] for row in result["sections"]["changes"]],
                         ["supersedes"])
        self.assertEqual({row["source_id"] for row in result["references"]},
                         {"Synthetic:Atlas-01", "Synthetic:Atlas-02"})

    def test_unknown_project_does_not_fall_back_to_atlas(self):
        result = self.engine.generate(
            "Quelle est la décision sur le budget Orion ?",
            parsed=self.parsed("answer", "budget Orion"))
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["references"], [])

    def test_unknown_responsible_does_not_become_a_speaker(self):
        result = self.engine.generate(
            "Qui est responsable du prix des capteurs Nova pour Atlas ?",
            parsed=self.parsed("answer", "capteurs Atlas"))
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(result["clarification_reason"], "missing_owner")
        self.assertEqual(result["references"], [])

    def test_project_only_answer_requires_subject(self):
        result = self.engine.generate(
            "Que sait-on sur Atlas ?",
            parsed=self.parsed("answer"))
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["clarification_reason"], "ambiguous_subject")

    def test_missing_subject_without_project_remains_clarification(self):
        result = self.engine.generate(
            "Quel est le budget prévu ?",
            parsed=self.parsed("clarify", None, "missing_subject"))
        self.assertEqual(result["status"], "clarify")

    def test_action_intent_is_not_executed(self):
        result = self.engine.generate(
            "Rédige un mail à Camille sur le budget Atlas.",
            parsed=self.parsed("draft_email", "budget Atlas"))
        self.assertEqual(result["status"], "unsupported_intent")
        self.assertEqual(result["references"], [])

    def test_markdown_exposes_citation_and_exact_excerpt(self):
        result = self.engine.generate(
            "Quelle est la décision actuelle sur le budget Atlas ?",
            parsed=self.parsed("answer", "budget Atlas"))
        rendered = render_markdown(result)
        self.assertIn("Synthetic:Atlas-02:0-0:0", rendered)
        self.assertIn(result["references"][0]["evidence"], rendered)

    def test_tampered_claim_or_reference_is_rejected(self):
        original = self.engine.generate(
            "Quelle est la décision actuelle sur le budget Atlas ?",
            parsed=self.parsed("answer", "budget Atlas"))
        bad_claim = deepcopy(original)
        bad_claim["sections"]["answer"][0]["text"] = "Budget de 90 000 € approuvé."
        with self.assertRaises(ValueError):
            validate_response(bad_claim, self.engine.units, self.engine.relations)
        bad_reference = deepcopy(original)
        bad_reference["references"][0]["evidence"] = "preuve inventée"
        with self.assertRaises(ValueError):
            validate_response(bad_reference, self.engine.units, self.engine.relations)
        comparison = self.engine.generate(
            "Compare les deux réunions sur la livraison Atlas.",
            parsed=self.parsed("compare", "livraison Atlas"))
        invented_relation = deepcopy(comparison)
        invented_relation["sections"]["changes"][0]["relation"] = "contradicts"
        with self.assertRaises(ValueError):
            validate_response(invented_relation, self.engine.units, self.engine.relations)


if __name__ == "__main__":
    unittest.main()
