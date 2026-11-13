"""New-source ingestion preserves evidence without promoting unreviewed claims."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Data Analysis"))

from bootstrap_memory import bootstrap
from generate_response import ResponseEngine
from ingest_semantic_units import read_jsonl
from semantic_agent import SemanticPipeline


class BootstrapMemoryTests(unittest.TestCase):
    def test_text_transcripts_are_searchable_but_decisions_remain_unreviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            transcripts = base / "transcripts"
            transcripts.mkdir()
            (transcripts / "meeting1.txt").write_text(
                "Camille: Je propose un budget de 40 000 € pour Atlas.\n"
                "Marc: Qui valide le budget Atlas ?\n", encoding="utf-8")
            (transcripts / "meeting2.txt").write_text(
                "Camille: Le budget Atlas de 40 000 € est approuvé.\n"
                "Julie: Je prends en charge le dossier Atlas.\n", encoding="utf-8")
            output = base / "memory"
            result = bootstrap(text_dir=transcripts, project="Atlas", source_date="2026-11-03",
                               output_dir=output)
            self.assertEqual((result["conversations"], result["units"]), (2, 4))
            units = list(read_jsonl(output / "units.jsonl"))
            self.assertEqual(len({row["source_id"] for row in units}), 2)
            self.assertTrue(all(row["statement"] == row["evidence"] for row in units))
            self.assertFalse(any(row["status"] == "confirmed" or row["owner"] for row in units))
            self.assertEqual(units[-1]["type"], "commitment")
            self.assertEqual(list(read_jsonl(output / "relations.jsonl")), [])
            engine = ResponseEngine(output / "conversations.jsonl", output / "units.jsonl",
                                    output / "relations.jsonl", output / "topics.jsonl")
            answer = SemanticPipeline(engine=engine, method="bm25").run("Quel est le budget Atlas ?")
            self.assertEqual(answer["status"], "uncertain")
            self.assertEqual(answer["sections"]["answer"][0]["role"], "unreviewed_source_statement")
            self.assertIn(answer["references"][0]["evidence"],
                          (transcripts / "meeting2.txt").read_text(encoding="utf-8"))

    def test_refuses_existing_output_and_single_source(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            transcripts = base / "transcripts"
            transcripts.mkdir()
            (transcripts / "only.txt").write_text("Une phrase.\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                bootstrap(text_dir=transcripts, project="Atlas", output_dir=base / "memory")


if __name__ == "__main__":
    unittest.main()
