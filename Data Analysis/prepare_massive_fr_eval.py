"""Freeze a French recipient stress test from public MASSIVE test utterances."""

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data/Processed/massive-fr-test.parquet"
CASES = ROOT / "Data/Curated/massive_fr_recipient_cases.jsonl"
MANIFEST = ROOT / "Data/Curated/massive_fr_recipient_manifest.json"
REVISION = "2c02f1bc446b202987536ab9b2014579b1e7af3e"
URL = ("https://huggingface.co/datasets/AmazonScience/massive/resolve/"
       f"{REVISION}/fr-FR/massive-test.parquet")
SOURCE_SHA256 = "aacd57aa2bcf6e1d037a350a41b70cb756bbf51199d2023f40a3ebb9d38a1981"
DIRECT = re.compile(r"\b(?:e-?mail|mail)\s+(?:à|pour)\s+\[person : ([^\]]+)\]", re.I)
PERSON = re.compile(r"\[person : ([^\]]+)\]")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source():
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as response, SOURCE.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
    if sha256(SOURCE) != SOURCE_SHA256:
        raise ValueError("Unexpected MASSIVE source checksum")
    return SOURCE


def quality(row):
    judgments = row.judgments
    return (len(judgments["intent_score"]) > 0
            and all(value == 1 for value in judgments["intent_score"])
            and all(value == 1 for value in judgments["slots_score"])
            and min(judgments["grammar_score"]) >= 3)


def prepare():
    path = source()
    schema = pq.read_schema(path)
    intents = json.loads(schema.metadata[b"huggingface"])["info"]["features"]["intent"]["names"]
    rows = pd.read_parquet(path)
    cases = []
    for row in rows.itertuples(index=False):
        if not quality(row):
            continue
        intent = intents[row.intent]
        people = PERSON.findall(row.annot_utt)
        direct = DIRECT.search(row.annot_utt)
        if (intent == "email_sendemail" and direct and len(people) == 1
                and "[relation :" not in row.annot_utt):
            category, person = "direct_named", direct.group(1)
        elif intent == "email_sendemail" and "[relation :" in row.annot_utt and not people:
            category, person = "relation_without_identity", None
        elif intent in {"email_query", "email_querycontact"} and people:
            category, person = "person_mentioned_in_query", None
        else:
            continue
        cases.append({
            "id": str(row.id), "category": category, "request": row.utt,
            "source_intent": intent, "annotated_utterance": row.annot_utt,
            "expected_person": person,
        })
    CASES.parent.mkdir(parents=True, exist_ok=True)
    CASES.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cases),
                     encoding="utf-8", newline="\n")
    counts = {category: sum(row["category"] == category for row in cases)
              for category in ("direct_named", "relation_without_identity",
                               "person_mentioned_in_query")}
    manifest = {
        "source_url": URL, "source_sha256": SOURCE_SHA256,
        "source_split": "fr-FR/test", "source_license": "CC-BY-4.0",
        "cases_sha256": sha256(CASES), "cases": len(cases), "categories": counts,
        "selection": "All rows with unanimous correct intent/slots and grammar >=3, then fixed slot/syntax categories",
        "limitation": "MASSIVE annotates person and relation slots, not responsibility or email contact addresses.",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
