"""Freeze blind AMI review packets for two independent human annotators.

The four selected AMI meetings have human-authored decision/action summaries
and are absent from all local QMSum splits. This script creates candidates,
blank reviewer forms and separate model predictions. It creates no gold labels.
"""

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from ingest_semantic_units import ROOT
from predict_owner import predict_owner
from predict_relations import predict_pair


AMI = ROOT / "Data/Datasets/AMI"
MEETING_PAIRS = (("IB4003", "IB4005"), ("IB4010", "IB4011"))
MEETING_DATES = {"IB4003": "2004-11-02", "IB4005": "2004-11-02",
                 "IB4010": "2005-04-08", "IB4011": "2005-04-08"}
DATE_SOURCE = "https://groups.inf.ed.ac.uk/ami/corpus/signals.shtml"
SEQUENCE_SOURCE = "https://groups.inf.ed.ac.uk/ami/corpus/meetingids.shtml"
DEFAULT_OUTPUT = ROOT / "Data/Curated/independent_ami_review"
NITE_ID = "{http://nite.sourceforge.net/}id"
PAIR_FIELDS = ("item_id", "meeting_a", "meeting_b", "source_a", "source_b",
               "sentence_id_a", "sentence_id_b", "text_a", "text_b",
               "semantic_relation", "temporal_relation", "chronology_evidence", "notes")
OWNER_FIELDS = ("item_id", "meeting", "source", "sentence_id", "text",
                "owner", "owner_evidence", "notes")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha256(path):
    """Hash AMI XML after canonicalizing checkout-dependent line endings."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def qmsum_ids():
    return {path.stem for path in (ROOT / "Data/Datasets/QMSum/data").glob("*/*/*.json")}


def sections(meeting):
    path = AMI / "abstractive" / f"{meeting}.abssumm.xml"
    root = ET.parse(path).getroot()
    output = {}
    for section in ("abstract", "actions", "decisions", "problems"):
        parent = root.find(section)
        output[section] = [{"sentence_id": node.attrib[NITE_ID],
                            "text": " ".join("".join(node.itertext()).split())}
                           for node in parent.findall("sentence")]
    return path, output


def transcript(meeting):
    """Render the local manual AMI word/segment layers as timed speaker turns."""
    rendered = []
    for path in sorted((AMI / "segments").glob(f"{meeting}.*.segments.xml")):
        speaker = path.name.split(".")[1]
        word_path = AMI / "words" / f"{meeting}.{speaker}.words.xml"
        word_nodes = ET.parse(word_path).getroot()
        words = {node.attrib[NITE_ID]: "".join(node.itertext()) for node in word_nodes
                 if node.tag == "w"}
        for segment in ET.parse(path).getroot().findall("segment"):
            chunks = []
            for child in segment.findall("{http://nite.sourceforge.net/}child"):
                ids = re.findall(r"id\(([^)]+)\)", child.attrib["href"])
                if not ids:
                    continue
                start = int(re.search(r"words(\d+)$", ids[0]).group(1))
                end = int(re.search(r"words(\d+)$", ids[-1]).group(1))
                for index in range(start, end + 1):
                    token = words.get(f"{meeting}.{speaker}.words{index}")
                    if token:
                        chunks.append(token)
            content = " ".join(chunks)
            content = re.sub(r"\s+([,.!?;:])", r"\1", content)
            if content:
                rendered.append((float(segment.attrib.get("transcriber_start", 0)), speaker, content))
    rendered.sort(key=lambda row: (row[0], row[1]))
    return "\n".join(f"[{time:.1f}s] {speaker}: {content}" for time, speaker, content in rendered) + "\n"


def candidates(old, new, old_name, new_name):
    left = old["decisions"]
    right = new["decisions"]
    if not left or not right:
        return []
    rows = []
    for index, (i, j) in enumerate(((i, j) for i in range(len(left))
                                    for j in range(len(right))), 1):
        rows.append({"item_id": f"PAIR-{old_name}-{new_name}-{index:03d}",
                     "meeting_a": old_name, "meeting_b": new_name,
                     "source_a": f"Data/Datasets/AMI/abstractive/{old_name}.abssumm.xml",
                     "source_b": f"Data/Datasets/AMI/abstractive/{new_name}.abssumm.xml",
                     "sentence_id_a": left[i]["sentence_id"],
                     "sentence_id_b": right[j]["sentence_id"],
                     "text_a": left[i]["text"], "text_b": right[j]["text"]})
    return rows


def owner_candidates(section, meeting):
    return [{"item_id": f"OWNER-{meeting}-{index:03d}", "meeting": meeting,
             "source": f"Data/Datasets/AMI/abstractive/{meeting}.abssumm.xml",
             "sentence_id": row["sentence_id"], "text": row["text"]}
            for index, row in enumerate(section["actions"], 1)]


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(output=DEFAULT_OUTPUT, *, replace_unannotated=False):
    meetings = {meeting for pair in MEETING_PAIRS for meeting in pair}
    overlap = meetings & qmsum_ids()
    if overlap:
        raise ValueError(f"Réunions déjà présentes dans QMSum : {sorted(overlap)}")
    if output.exists() and any(output.iterdir()):
        if not replace_unannotated:
            raise ValueError(f"Dossier de sortie déjà rempli : {output}")
        for path in output.glob("reviewer_*.csv"):
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if any(row.get(field, "").strip() for field in
                           ("semantic_relation", "temporal_relation", "owner")):
                        raise ValueError(f"Annotations déjà remplies, remplacement interdit : {path}")
    output.mkdir(parents=True, exist_ok=True)
    data = {}
    source_hashes = {}
    for meeting in sorted(meetings):
        path, section = sections(meeting)
        data[meeting] = section
        source_hashes[meeting] = source_sha256(path)
        (output / f"transcript_{meeting}.txt").write_text(transcript(meeting), encoding="utf-8")
    pair_rows = [row for old, new in MEETING_PAIRS
                 for row in candidates(data[old], data[new], old, new)]
    owner_rows = [row for meeting in sorted(meetings)
                  for row in owner_candidates(data[meeting], meeting)]
    pair_path = output / "pair_items.jsonl"
    owner_path = output / "owner_items.jsonl"
    for path, rows in ((pair_path, pair_rows), (owner_path, owner_rows)):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    for reviewer in ("a", "b"):
        write_csv(output / f"reviewer_{reviewer}_pairs.csv", PAIR_FIELDS, pair_rows)
        write_csv(output / f"reviewer_{reviewer}_owners.csv", OWNER_FIELDS, owner_rows)
    predictions = []
    for row in pair_rows:
        old_unit = {"unit_id": row["sentence_id_a"], "source_id": row["meeting_a"],
                    "source_date": None, "project_id": row["meeting_a"][:-1],
                    "type": "decision", "status": "confirmed", "start_turn": 0,
                    "end_turn": 0, "evidence": row["text_a"], "statement": row["text_a"]}
        new_unit = {**old_unit, "unit_id": row["sentence_id_b"],
                    "source_id": row["meeting_b"], "evidence": row["text_b"],
                    "statement": row["text_b"]}
        prediction = predict_pair(new_unit, old_unit)
        predictions.append({"item_id": row["item_id"], "task": "pair",
                            "prediction": prediction["relation"], "reason": prediction["reason"]})
    predictions += [{"item_id": row["item_id"], "task": "owner",
                     "prediction": predict_owner(row["text"])} for row in owner_rows]
    (output / "model_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8")
    packet = ["# AMI source packet", "", "AMI manual annotations v1.6.2, CC BY 4.0.",
              f"The official [signals table]({DATE_SOURCE}) gives the calendar dates below.",
              "Each comparison pair took place on the same date; the date alone does not prove which decision came later or replaced another.", ""]
    packet += [f"The official [meeting ID description]({SEQUENCE_SOURCE}) states that IB4005 retained participants from the earlier office-move groups, including IB4003. This supports meeting order for that pair, but does not establish replacement of any particular decision. It gives no order for IB4010/IB4011.", ""]
    for meeting in sorted(meetings):
        packet += [f"## {meeting} — {MEETING_DATES[meeting]}", "",
                   f"Manual transcript: [transcript_{meeting}.txt](transcript_{meeting}.txt)", ""]
        for name, rows in data[meeting].items():
            packet += [f"### {name}", ""]
            packet += [f"- {row['sentence_id']}: {row['text']}" for row in rows]
            packet.append("")
    (output / "source_packet.md").write_text("\n".join(packet), encoding="utf-8")
    for reviewer in ("a", "b"):
        with ZipFile(output / f"reviewer_{reviewer}_packet.zip", "w", ZIP_DEFLATED) as archive:
            for name in ("source_packet.md", *(f"transcript_{meeting}.txt" for meeting in sorted(meetings)),
                         f"reviewer_{reviewer}_pairs.csv", f"reviewer_{reviewer}_owners.csv"):
                entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.compress_type = ZIP_DEFLATED
                archive.writestr(entry, (output / name).read_bytes())
    manifest = {"source_sha256": source_hashes,
                "source_hash_canonicalization": "CRLF to LF before SHA-256",
                "meeting_dates": MEETING_DATES, "meeting_date_source": DATE_SOURCE,
                "meeting_sequence_source": SEQUENCE_SOURCE,
                "transcript_sha256": {meeting: sha256(output / f"transcript_{meeting}.txt")
                                      for meeting in sorted(meetings)},
                "pair_items_sha256": sha256(pair_path),
                "owner_items_sha256": sha256(owner_path),
                "predictions_sha256": sha256(output / "model_predictions.jsonl"),
                "reviewer_packet_sha256": {reviewer: sha256(output / f"reviewer_{reviewer}_packet.zip")
                                           for reviewer in ("a", "b")},
                "pair_items": len(pair_rows), "owner_items": len(owner_rows),
                "meetings": sorted(meetings), "gold_status": "awaiting_two_independent_humans"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replace-unannotated", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, replace_unannotated=args.replace_unannotated),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
