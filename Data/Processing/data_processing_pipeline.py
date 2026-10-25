"""Normalize AMI, ICSI, MeetingBank and QMSum into four meeting-level JSONL files.

Run: python Data/Processing/data_processing_pipeline.py
Original dataset files are left untouched.
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "Data" / "Datasets"
OUTPUT = ROOT / "Data" / "Processed"
NITE_ID = "{http://nite.sourceforge.net/}id"
WORD_REF = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def segment_text(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    return re.sub(r"([({\[])\s+", r"\1", text).strip()


def parse_nite_segment(segment, words, positions):
    tokens = []
    for child in segment:
        if local_name(child.tag) != "child":
            continue
        match = WORD_REF.search(child.get("href", ""))
        if not match:
            continue
        start_id, end_id = match.group(1), match.group(2) or match.group(1)
        if start_id not in positions or end_id not in positions:
            raise ValueError(f"Unresolved word reference: {child.get('href')}")
        start, end = positions[start_id], positions[end_id]
        if end < start:
            raise ValueError(f"Reversed word range: {child.get('href')}")
        tokens.extend(word.text.strip() for word in words[start : end + 1]
                      if local_name(word.tag) == "w" and word.text and word.text.strip())
    return segment_text(tokens)


def load_nite_corpus(name, base, segment_dir, segment_suffix):
    """Read AMI/ICSI NITE XML with speaker, timing and provenance."""
    meetings = defaultdict(lambda: {"segments": [], "source_files": []})
    segment_files = sorted((base / segment_dir).glob(f"*{segment_suffix}"))
    if not segment_files:
        raise FileNotFoundError(f"No {name} segments in {base / segment_dir}")

    for segment_file in segment_files:
        stem = segment_file.name.removesuffix(segment_suffix)
        meeting_id, speaker_code = stem.rsplit(".", 1)
        word_dir = "words" if name == "AMI" else "Words"
        words_file = base / word_dir / f"{stem}.words.xml"
        words = list(ET.parse(words_file).getroot())
        positions = {word.get(NITE_ID): index for index, word in enumerate(words)}
        source_files = meetings[meeting_id]["source_files"]
        source_files.extend([str(segment_file.relative_to(ROOT)), str(words_file.relative_to(ROOT))])

        for segment in ET.parse(segment_file).getroot():
            if local_name(segment.tag) != "segment":
                continue
            content = parse_nite_segment(segment, words, positions)
            if not content:  # Non-speech event.
                continue
            start = segment.get("starttime") or segment.get("transcriber_start")
            end = segment.get("endtime") or segment.get("transcriber_end")
            meetings[meeting_id]["segments"].append({
                "segment_id": segment.get(NITE_ID),
                "speaker": segment.get("participant") or speaker_code,
                "text": content,
                "start_seconds": float(start) if start else None,
                "end_seconds": float(end) if end else None,
            })

    records = []
    for meeting_id, data in sorted(meetings.items()):
        data["segments"].sort(key=lambda s: (s["start_seconds"] is None,
                                              s["start_seconds"] or 0, s["segment_id"] or ""))
        records.append({
            "dataset": name, "meeting_id": meeting_id, "split": None,
            "source_date": None,
            "granularity": "speech_segment", "segments": data["segments"],
            "summary": None, "queries": None, "topics": None,
            "source_files": data["source_files"],
        })
    return pd.DataFrame.from_records(records)


def load_meetingbank(base):
    records = []
    for split in ("train", "validation", "test"):
        path = base / f"{split}.jsonl"
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                item = json.loads(line)
                meeting_id = item.get("uid")
                if not meeting_id:
                    raise ValueError(f"Missing MeetingBank uid in {path}:{line_number}")
                records.append({
                    "dataset": "MeetingBank", "meeting_id": str(meeting_id),
                    "split": split, "source_date": None,
                    "source_record_id": item.get("id"),
                    "granularity": "full_transcript",
                    "segments": [{"segment_id": f"{meeting_id}:transcript", "speaker": None,
                                  "text": item["transcript"], "start_seconds": None,
                                  "end_seconds": None}],
                    "summary": item.get("summary"), "queries": None, "topics": None,
                    "source_files": [str(path.relative_to(ROOT))],
                })
    return pd.DataFrame.from_records(records)


def load_qmsum(base):
    records = []
    # `all` and `jsonl` are duplicate exports of the split files.
    for domain in ("Academic", "Committee", "Product"):
        for directory, split in (("train", "train"), ("val", "validation"), ("test", "test")):
            for path in sorted((base / "data" / domain / directory).glob("*.json")):
                item = json.loads(path.read_text(encoding="utf-8"))
                meeting_id = path.stem
                records.append({
                    "dataset": "QMSum", "meeting_id": meeting_id, "split": split,
                    "source_date": None,
                    "domain": domain, "granularity": "speaker_turn",
                    "segments": [
                        {"segment_id": f"{meeting_id}:{index}",
                         "speaker": turn.get("speaker"), "text": turn["content"],
                         "start_seconds": None, "end_seconds": None}
                        for index, turn in enumerate(item["meeting_transcripts"])
                    ],
                    "summary": None,
                    "queries": {"general": item.get("general_query_list", []),
                                "specific": item.get("specific_query_list", [])},
                    "topics": item.get("topic_list", []),
                    "source_files": [str(path.relative_to(ROOT))],
                })
    if not records:
        raise FileNotFoundError(f"No QMSum meeting JSON files in {base / 'data'}")
    return pd.DataFrame.from_records(records)


def save_jsonl(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_json(temporary, orient="records", lines=True, force_ascii=False)
    temporary.replace(path)
    print(f"{path.relative_to(ROOT)}: {len(frame)} meetings")


def main():
    # Load the 4 datasets from their actual local formats.
    df_ami = load_nite_corpus("AMI", DATASETS / "AMI", "segments", ".segments.xml")
    df_icsi = load_nite_corpus("ICSI", DATASETS / "ICSI" / "ICSIplus", "Segments", ".segs.xml")
    df_meetingbank = load_meetingbank(DATASETS / "MeetingBank")
    df_qmsum = load_qmsum(DATASETS / "QMSum")

    # Convert them all to JSONL and save them in Data/Processed.
    for name, frame in (("ami", df_ami), ("icsi", df_icsi),
                        ("meetingbank", df_meetingbank), ("qmsum", df_qmsum)):
        save_jsonl(frame, OUTPUT / f"{name}.jsonl")

    # Stream the four exports into one JSONL file without loading them again.
    merged_path = OUTPUT / "meetings_merged.jsonl"
    temporary = merged_path.with_suffix(".jsonl.tmp")
    total = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as merged:
        for name in ("ami", "icsi", "meetingbank", "qmsum"):
            with (OUTPUT / f"{name}.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        merged.write(line.rstrip("\r\n") + "\n")
                        total += 1
    temporary.replace(merged_path)
    print(f"{merged_path.relative_to(ROOT)}: {total} meetings")


if __name__ == "__main__":
    main()
