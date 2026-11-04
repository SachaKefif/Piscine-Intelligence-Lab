"""EF-06 routing, nullable slots and conservative clarification checks."""

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from parse_intent import INTENTS, folded, load_classifier, matches_source_hash, parse_intent, route_with_safety


class IntentParserTests(unittest.TestCase):
    def test_model_source_hash_accepts_only_line_ending_equivalence(self):
        lf = b'{"request":"test"}\n'
        crlf = lf.replace(b"\n", b"\r\n")
        expected = hashlib.sha256(crlf).hexdigest()
        self.assertTrue(matches_source_hash(lf, expected))
        self.assertTrue(matches_source_hash(crlf, expected))
        self.assertFalse(matches_source_hash(b'{"request":"changed"}\n', expected))

    def test_development_examples_cover_all_intents_and_slots(self):
        path = ROOT / "Data" / "Curated" / "intent_seed_dev.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["intent"] for row in rows}, set(INTENTS))
        for row in rows:
            with self.subTest(id=row["id"]):
                actual = parse_intent(row["request"], use_rules=True)
                for key in ("intent", "subject", "person", "due_date", "action"):
                    self.assertEqual(actual[key], row[key])

    def test_named_person_is_not_resolved_as_recipient(self):
        result = parse_intent("Rédige un mail à Camille sur le budget Atlas.")
        self.assertEqual(result["intent"], "draft_email")
        self.assertEqual(result["person"], "Camille")
        self.assertNotIn("recipient", result)

    def test_unnamed_recipient_requires_clarification(self):
        result = parse_intent("Prépare un mail à la responsable sur le budget Atlas.")
        self.assertEqual(result["intent"], "clarify")
        self.assertIsNone(result["person"])
        self.assertEqual(result["clarification_reason"], "missing_named_recipient")

    def test_other_named_person_is_not_recipient(self):
        result = parse_intent("Rédige un mail à la responsable du budget Atlas. Camille connaît le dossier.")
        self.assertEqual(result["person"], "Camille")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(result["clarification_reason"], "missing_named_recipient")

    def test_hypothetical_statement_does_not_trigger_task(self):
        result = parse_intent("On pourrait peut-être revoir le budget Atlas.")
        self.assertEqual(result["intent"], "clarify")
        self.assertEqual(result["clarification_reason"], "hypothetical_not_request")

    def test_unanchored_relative_date_stays_null(self):
        result = parse_intent("Planifie une réunion avec Camille sur le budget Atlas demain.")
        self.assertIsNone(result["due_date"])
        self.assertEqual(result["intent"], "clarify")

    def test_lowercase_adjective_is_not_a_project(self):
        result = parse_intent("Quel est le budget prévu ?")
        self.assertIsNone(result["subject"])
        self.assertEqual(result["intent"], "clarify")

    def test_subject_with_french_preposition(self):
        result = parse_intent("Quelle est la décision sur le budget de Mistral ?")
        self.assertEqual(result["subject"], "budget Mistral")
        self.assertIsNone(result["person"])

    def test_event_noun_is_not_a_scheduling_verb(self):
        result = parse_intent("Quel créneau est prévu pour le budget Atlas ?")
        self.assertEqual(result["intent"], "answer")

    def test_invalid_calendar_date_stays_null(self):
        result = parse_intent("Planifie une réunion sur le budget Atlas le 31 février 2026.")
        self.assertIsNone(result["due_date"])
        self.assertEqual(result["intent"], "clarify")

    def test_empty_request_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_intent("  ")

    def test_model_is_available_and_routes_a_complete_request(self):
        self.assertTrue(hasattr(load_classifier(), "predict"))
        result = parse_intent("Rédige un mail à Camille sur le budget Atlas.")
        self.assertEqual(result["intent"], "draft_email")
        self.assertIsNone(result["clarification_reason"])

    def test_safety_policy_overrides_a_predicted_action(self):
        cases = (
            ("On pourrait peut-être confirmer le budget Atlas.", "create_task", "budget Atlas", "confirmer le budget atlas", None, "hypothetical_not_request"),
            ("Rédige un mail à la responsable sur le budget Atlas.", "draft_email", "budget Atlas", None, None, "missing_named_recipient"),
            ("Compare les deux décisions.", "compare", None, None, None, "missing_subject"),
            ("Crée une tâche sur le budget Atlas.", "create_task", "budget Atlas", None, None, "missing_action"),
            ("Planifie une réunion sur le budget Atlas demain.", "schedule_event", "budget Atlas", None, None, "missing_due_date"),
        )
        for request, predicted, subject, action, due_date, reason in cases:
            with self.subTest(request=request):
                self.assertEqual(route_with_safety(request, predicted, subject, action, due_date),
                                 ("clarify", reason))

    def test_new_holdout_has_new_projects_and_balanced_labels(self):
        data = ROOT / "Data" / "Curated"
        fresh = [json.loads(line) for line in (data / "intent_fresh_holdout.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(fresh), 56)
        self.assertEqual({label: sum(row["intent"] == label for row in fresh) for label in INTENTS},
                         {label: 8 for label in INTENTS})
        earlier_projects = set()
        for name in ("train", "validation", "eval_holdout"):
            rows = [json.loads(line) for line in (data / f"intent_{name}.jsonl").read_text(encoding="utf-8").splitlines()]
            earlier_projects.update(row["subject"].split()[-1] for row in rows if row["subject"])
        self.assertFalse(earlier_projects & {row["subject"].split()[-1] for row in fresh if row["subject"]})

    def test_expanded_splits_are_unique_balanced_and_project_disjoint(self):
        minimums = {"train": 400, "validation": 120, "eval_holdout": 120}
        project_sets = []
        all_requests = set()
        for name, minimum in minimums.items():
            path = ROOT / "Data" / "Curated" / f"intent_{name}.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), minimum)
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
            for label in INTENTS:
                self.assertGreaterEqual(sum(row["intent"] == label for row in rows), 15)
            projects = {row["subject"].split()[-1] for row in rows if row["subject"]}
            for prior in project_sets:
                self.assertFalse(prior & projects)
            project_sets.append(projects)
            for row in rows:
                key = folded(row["request"])
                self.assertNotIn(key, all_requests)
                all_requests.add(key)


if __name__ == "__main__":
    unittest.main()
