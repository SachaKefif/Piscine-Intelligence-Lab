"""Create one JSONL record per 10 consecutive QMSum speaker turns.

Run from any directory: python Data/Processing/create_qmsum_10_passages.py
Reads Data/Processed/qmsum.jsonl and writes
Data/Processed/qmsum_10_passages.jsonl without changing the source file.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "Data" / "Processed" / "qmsum.jsonl"
OUTPUT = ROOT / "Data" / "Processed" / "qmsum_10_passages.jsonl"
TURNS_PER_PASSAGE = 10


def main():
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    meetings = 0
    passages = 0

    with INPUT.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            meeting = json.loads(line)
            meeting_id = meeting["meeting_id"]
            segments = meeting["segments"]
            meetings += 1

            for start in range(0, len(segments), TURNS_PER_PASSAGE):
                group = segments[start : start + TURNS_PER_PASSAGE]
                end = start + len(group) - 1
                passage = {
                    "passage_id": f"{meeting_id}:{start}-{end}",
                    "dataset": meeting["dataset"],
                    "meeting_id": meeting_id,
                    "split": meeting["split"],
                    "source_date": meeting.get("source_date"),
                    "domain": meeting.get("domain"),
                    "start_turn": start,
                    "end_turn": end,
                    "text": "\n".join(
                        f"{segment['speaker']}: {segment['text']}"
                        if segment.get("speaker")
                        else segment["text"]
                        for segment in group
                    ),
                    "segments": group,
                    "source_files": meeting.get("source_files", []),
                }
                destination.write(json.dumps(passage, ensure_ascii=False) + "\n")
                passages += 1

    temporary.replace(OUTPUT)
    print(f"Created {OUTPUT.relative_to(ROOT)}: {passages} passages from {meetings} meetings")


if __name__ == "__main__":
    main()
