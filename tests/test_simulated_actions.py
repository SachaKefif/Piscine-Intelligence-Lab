"""EF-08/09: safe drafts and evidence-based recipient resolution."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from generate_response import ResponseEngine
from evaluate_actions import (BOREAL_ACTION_CASES, BOREAL_CONVERSATIONS, BOREAL_RECIPIENT_CASES,
                              BOREAL_RELATIONS, BOREAL_TOPICS, BOREAL_UNITS,
                              evaluate_drafts, evaluate_recipients)
from ingest_semantic_units import read_jsonl
from resolve_recipient import resolve_recipient, validate_resolution
from semantic_agent import handle
from simulate_actions import ActionSimulator, validate_draft


class RecipientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ResponseEngine()

    def test_explicit_addressee_is_user_choice_without_invented_address(self):
        result = resolve_recipient("Rédige un mail à Camille sur le budget Atlas.",
                                   engine=self.engine)
        self.assertEqual(result["status"], "explicit_user")
        self.assertEqual(result["name"], "Camille")
        self.assertIsNone(result["email"])
        self.assertEqual(result["source_unit_ids"], [])

    def test_recent_role_owner_is_sourced(self):
        request = "Prépare un mail à la responsable de la commande des capteurs Atlas."
        result = resolve_recipient(request, engine=self.engine)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["name"], "Julie")
        self.assertEqual(len(result["source_unit_ids"]), 2)
        self.assertTrue(all(row["owner"] == "Julie" for row in result["references"]))
        self.assertTrue(validate_resolution(result, engine=self.engine, request=request))
        forged = deepcopy(result)
        forged["references"][0]["evidence"] = "citation inventée"
        with self.assertRaises(ValueError):
            validate_resolution(forged, engine=self.engine, request=request)

    def test_devis_owner_is_not_order_owner(self):
        result = resolve_recipient(
            "Prépare un mail à la responsable du devis du prototype Atlas.",
            engine=self.engine)
        self.assertEqual(result["name"], "Camille")
        self.assertEqual(result["source_unit_ids"], ["Synthetic:Atlas-01:2-2:0"])

    def test_budget_approver_and_mentioned_person_are_not_owner(self):
        result = resolve_recipient(
            "Rédige un mail à la responsable du budget Atlas. Camille connaît le dossier.",
            engine=self.engine)
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["reason"], "no_explicit_owner")
        self.assertIsNone(result["name"])

    def test_unspecified_role_stays_ambiguous(self):
        result = resolve_recipient("Prépare un mail à la responsable sur le budget Atlas.",
                                   engine=self.engine)
        self.assertEqual(result["reason"], "role_not_specific")

    def test_stale_role_abstains(self):
        result = resolve_recipient(
            "Prépare un mail à la responsable de la commande des capteurs Atlas.",
            engine=self.engine, as_of="2027-04-01")
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["reason"], "stale_attribution")

    def test_competing_owners_abstain_even_when_one_is_newer(self):
        engine = deepcopy(self.engine)
        other = deepcopy(engine.units["Synthetic:Atlas-02:6-6:0"])
        other["unit_id"] = "Synthetic:Atlas-02:8-8:0"
        other["owner"] = "Marc"
        other["statement"] = "Marc doit envoyer la commande des capteurs."
        other["start_turn"] = other["end_turn"] = 8
        engine.units[other["unit_id"]] = other
        engine.topics["atlas_procurement"]["unit_ids"].append(other["unit_id"])
        result = resolve_recipient(
            "Prépare un mail à la responsable de la commande des capteurs Atlas.", engine=engine)
        self.assertEqual(result["reason"], "competing_owners")
        self.assertEqual(result["candidates"], ["Julie", "Marc"])
        engine.relations.append({"relation": "supersedes", "from_id": other["unit_id"],
                                 "to_id": "Synthetic:Atlas-02:6-6:0"})
        # The older Julie commitment is still separate, so competition remains.
        self.assertEqual(resolve_recipient(
            "Prépare un mail à la responsable de la commande des capteurs Atlas.",
            engine=engine)["reason"], "competing_owners")


class ActionDraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simulator = ActionSimulator()

    @staticmethod
    def parsed(intent, subject=None, action=None, due_date=None, reason=None):
        return {"intent": intent, "subject": subject, "action": action,
                "due_date": due_date, "clarification_reason": reason}

    def test_email_draft_has_content_source_and_no_execution(self):
        result = self.simulator.prepare("Rédige un mail à Camille sur le budget Atlas.")
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["intent"], "draft_email")
        self.assertEqual(result["recipient"]["name"], "Camille")
        self.assertIsNone(result["parameters"]["to_email"])
        self.assertTrue(result["needs_confirmation"])
        self.assertFalse(result["executed"])
        self.assertTrue(validate_draft(result, self.simulator.engine.units))

    def test_role_based_email_uses_recipient_evidence(self):
        result = self.simulator.prepare(
            "Prépare un mail à la responsable de la commande des capteurs Atlas.")
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["recipient"]["name"], "Julie")
        self.assertTrue(set(result["recipient"]["source_unit_ids"]) <= set(result["source_unit_ids"]))

    def test_unowned_budget_role_clarifies_without_draft(self):
        result = self.simulator.prepare(
            "Prépare un mail à la responsable du budget Atlas.")
        self.assertEqual(result["status"], "clarify")
        self.assertEqual(result["clarification_reason"], "no_explicit_owner")
        self.assertIsNone(result["tool_candidate"])
        self.assertFalse(result["executed"])

    def test_task_draft_preserves_user_action_not_assignee_inference(self):
        result = self.simulator.prepare(
            "Crée une tâche : envoyer la commande des capteurs Atlas avant le 2026-11-06.",
            parsed=self.parsed("create_task", "capteurs Atlas", "envoyer la commande des capteurs atlas",
                               "2026-11-06"))
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["parameters"]["due_date"], "2026-11-06")
        self.assertIsNone(result["parameters"]["assignee_name"])
        self.assertEqual(result["tool_candidate"], "tasks.create_draft")
        self.assertFalse(result["executed"])

    def test_event_draft_keeps_unknown_time_and_attendees_empty(self):
        result = self.simulator.prepare(
            "Planifie une réunion sur la livraison Atlas le 2026-11-19.",
            parsed=self.parsed("schedule_event", "livraison Atlas", due_date="2026-11-19"))
        self.assertEqual(result["status"], "draft")
        self.assertEqual(result["parameters"]["date"], "2026-11-19")
        self.assertIsNone(result["parameters"]["time"])
        self.assertEqual(result["parameters"]["attendees"], [])
        self.assertFalse(result["executed"])

    def test_missing_date_action_and_hypothetical_abstain(self):
        cases = [
            ("Crée une tâche sur le budget Atlas.", self.parsed("create_task", "budget Atlas"), "missing_action"),
            ("Planifie une réunion sur le budget Atlas demain.", self.parsed("schedule_event", "budget Atlas"), "missing_due_date"),
            ("On pourrait peut-être revoir le budget Atlas.", self.parsed("clarify", "budget Atlas", reason="hypothetical_not_request"), "hypothetical_not_request"),
        ]
        for request, parsed, reason in cases:
            with self.subTest(request=request):
                result = self.simulator.prepare(request, parsed=parsed)
                self.assertEqual(result["status"], "clarify")
                self.assertEqual(result["clarification_reason"], reason)
                self.assertIsNone(result["tool_candidate"])
                self.assertFalse(result["executed"])

    def test_conflicting_context_abstains(self):
        result = self.simulator.prepare(
            "Rédige un mail à Camille sur le prix des capteurs Nova pour Atlas.",
            parsed=self.parsed("draft_email", "capteurs Nova"))
        self.assertEqual(result["clarification_reason"], "conflicting_evidence")

    def test_validator_rejects_executed_and_modified_quote(self):
        original = self.simulator.prepare("Rédige un mail à Camille sur le budget Atlas.")
        executed = deepcopy(original)
        executed["executed"] = True
        with self.assertRaises(ValueError):
            validate_draft(executed, self.simulator.engine.units)
        forged = deepcopy(original)
        forged["content"] = "Bonjour Camille, budget approuvé."
        with self.assertRaises(ValueError):
            validate_draft(forged, self.simulator.engine.units)

    def test_agent_routes_answers_and_actions_end_to_end(self):
        answer = handle("Quelle est la décision actuelle sur le budget Atlas ?",
                        engine=self.simulator.engine)
        self.assertEqual(answer["intent"], "answer")
        self.assertEqual(answer["status"], "ok")
        draft = handle("Rédige un mail à Camille sur le budget Atlas.",
                       engine=self.simulator.engine)
        self.assertEqual(draft["intent"], "draft_email")
        self.assertEqual(draft["status"], "draft")
        self.assertFalse(draft["executed"])
        task = handle("Crée une tâche pour envoyer la commande des capteurs Atlas avant le 2026-11-06.",
                      engine=self.simulator.engine)
        self.assertEqual(task["intent"], "create_task")
        self.assertEqual(task["status"], "draft")
        event = handle("Planifie une réunion sur la livraison Atlas le 2026-11-19.",
                       engine=self.simulator.engine)
        self.assertEqual(event["intent"], "schedule_event")
        self.assertEqual(event["status"], "draft")
        ambiguous = handle("Prépare un mail à la responsable du budget Atlas.",
                           engine=self.simulator.engine)
        self.assertEqual(ambiguous["status"], "clarify")


class BorealIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ResponseEngine(BOREAL_CONVERSATIONS, BOREAL_UNITS,
                                    BOREAL_RELATIONS, BOREAL_TOPICS)

    def test_reviewed_reassignment_replaces_old_role_owner(self):
        result = resolve_recipient("Rédige un mail à la responsable du rapport Boréal.",
                                   engine=self.engine)
        self.assertEqual(result["name"], "Omar")
        self.assertEqual(result["source_unit_ids"], ["Synthetic:Boreal-02:0-0:0"])
        self.assertEqual(result["references"][0]["owner"], "Omar")

    def test_new_project_cases_are_reproducible(self):
        recipients = evaluate_recipients(
            self.engine, list(read_jsonl(BOREAL_RECIPIENT_CASES)))
        drafts = evaluate_drafts(
            ActionSimulator(self.engine), list(read_jsonl(BOREAL_ACTION_CASES)))
        self.assertEqual(recipients["false_recipients"], 0)
        self.assertEqual(recipients["justified_abstention_rate"], 1.0)
        self.assertEqual(drafts["safe_output_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
