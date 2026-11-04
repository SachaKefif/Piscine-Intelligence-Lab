"""Build a small, provenance-checked agent fixture from two held-out QMSum meetings.

The raw test transcripts are committed under Data/Datasets/QMSum. Their dates are
unknown; this script does not invent them. No training script reads this fixture.
"""

import argparse
import json
from pathlib import Path

from ingest_semantic_units import ROOT


MEETINGS = ("ES2004a", "ES2004b")
RAW = ROOT / "Data/Datasets/QMSum/data/Product/test"
OUTPUT = ROOT / "Data/Processed/qmsum_agent_holdout_conversations.jsonl"
UNITS = ROOT / "Data/Curated/qmsum_agent_holdout_units.jsonl"


def prepare(output=OUTPUT, *, meetings=MEETINGS, units=UNITS, raw_dir=RAW):
    raw_dir = raw_dir.resolve()
    rows = []
    for meeting_id in meetings:
        raw_path = raw_dir / f"{meeting_id}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        rows.append({
            "dataset": "QMSum", "meeting_id": meeting_id, "split": "test",
            "source_date": None, "granularity": "speaker_turn",
            "source_files": [str(raw_path.relative_to(ROOT)).replace("\\", "/")],
            "segments": [
                {"segment_id": f"{meeting_id}:{index}",
                 "speaker": turn["speaker"], "text": turn["content"]}
                for index, turn in enumerate(raw["meeting_transcripts"])
            ],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                      encoding="utf-8")
    from ingest_semantic_units import ingest
    # Validates every evidence span and the exact source/turn metadata.
    count, sources = ingest(output, units,
                            output.with_name(output.stem + "_units_index.jsonl"))
    return {"meetings": len(rows), "units": count, "sources": sources,
            "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--units", type=Path, default=UNITS)
    parser.add_argument("--meetings", nargs="+", default=MEETINGS)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, meetings=args.meetings, units=args.units,
                             raw_dir=args.raw_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
