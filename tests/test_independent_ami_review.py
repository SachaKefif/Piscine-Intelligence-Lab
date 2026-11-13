"""Reviewer gates; the dummy labels in this test are not research gold."""

import csv
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from evaluate_independent_ami_review import review_state, write_adjudication_templates
from predict_owner import predict_owner


class IndependentAmiReviewTests(unittest.TestCase):
    def test_reviewer_packets_are_separate_and_blind(self):
        source = ROOT / "Data/Curated/independent_ami_review"
        for reviewer, other in (("a", "b"), ("b", "a")):
            with ZipFile(source / f"reviewer_{reviewer}_packet.zip") as archive:
                names = set(archive.namelist())
            self.assertIn(f"reviewer_{reviewer}_pairs.csv", names)
            self.assertIn(f"reviewer_{reviewer}_owners.csv", names)
            self.assertIn("source_packet.md", names)
            self.assertNotIn("model_predictions.jsonl", names)
            self.assertNotIn(f"reviewer_{other}_pairs.csv", names)
            self.assertNotIn(f"reviewer_{other}_owners.csv", names)

    def test_owner_proposals_keep_vague_assignments_unresolved(self):
        self.assertEqual(predict_owner("The project manager will find out about walls."),
                         "project manager")
        self.assertIsNone(predict_owner("One team member will ask about preferences."))

    def test_review_remains_pending_until_two_complete_matching_forms(self):
        source = ROOT / "Data/Curated/independent_ami_review"
        self.assertEqual(review_state(source)["status"], "pending_human_review")
        self.assertIsNone(review_state(source)["pair_agreement_rate"])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "fixture"
            shutil.copytree(source, directory)
            for reviewer in ("a", "b"):
                for kind in ("pairs", "owners"):
                    path = directory / f"reviewer_{reviewer}_{kind}.csv"
                    with path.open(encoding="utf-8", newline="") as stream:
                        records = list(csv.DictReader(stream))
                        fields = list(records[0])
                    for record in records:
                        if kind == "pairs":
                            record["semantic_relation"] = "unrelated"
                            record["temporal_relation"] = "not_supersedes"
                        else:
                            record["owner"] = "none"
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(records)
            scored = review_state(directory)
            self.assertEqual(scored["status"], "scored")
            self.assertIsNone(scored["contradicts"]["recall"])
            self.assertIsNone(scored["supersedes"]["recall"])
            path = directory / "reviewer_b_pairs.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
                fields = list(records[0])
            records[0]["semantic_relation"] = "contradicts"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            self.assertEqual(review_state(directory)["status"], "pending_human_review")
            self.assertEqual(len(review_state(directory)["pending_adjudication"]), 1)
            templates = write_adjudication_templates(directory, review_state(directory),
                                                     directory / "adjudication")
            self.assertEqual(len(templates), 1)
            with Path(templates[0]).open(encoding="utf-8", newline="") as stream:
                template_rows = list(csv.DictReader(stream))
            self.assertEqual(len(template_rows), 1)
            self.assertEqual(template_rows[0]["item_id"], records[0]["item_id"])
            self.assertEqual(template_rows[0]["semantic_relation"], "")


if __name__ == "__main__":
    unittest.main()
