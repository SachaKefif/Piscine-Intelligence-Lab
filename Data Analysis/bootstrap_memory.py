"""Build a cautious, searchable draft memory from new conversation transcripts.

Input is either normalized conversation JSONL or a directory of UTF-8 text files.
Each nonempty text line is a turn, optionally prefixed with ``Speaker: ``.
The output is usable by semantic_agent.py, but all inferred decisions, owners and
relations remain unconfirmed until a person reviews their evidence.
"""

import argparse
import json
import re
from pathlib import Path

from ingest_semantic_units import SCHEMA_PATH, read_jsonl, source_id, validate_units
from predict_owner import predict_owner
from predict_relations import overlap, predict_pair


def text_conversations(directory, project, source_date):
    files = sorted(directory.glob("*.txt"))
    if len(files) < 2:
        raise ValueError("Fournir au moins deux transcriptions .txt")
    rows = []
    for path in files:
        segments = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            match = re.match(r"^([^:]{2,50}):\s+(.+)$", line)
            speaker, content = (match.group(1), match.group(2)) if match else (None, line)
            segments.append({"segment_id": f"{path.stem}:{number}", "speaker": speaker, "text": content})
        if not segments:
            raise ValueError(f"Transcription vide : {path}")
        rows.append({"dataset": "Local", "meeting_id": path.stem, "source_date": source_date,
                     "project_id": project, "segments": segments})
    return rows


def candidate_kind(text):
    plain = text.casefold()
    if text.rstrip().endswith("?"):
        return "question", "unknown"
    if re.search(r"\b(?:pourrait|peut-être|peut etre|éventuellement|envisageons)\b", plain):
        return "idea", "hypothetical"
    if re.search(r"\b(?:je|nous|on)\s+propos\w*\b|\bproposition\b", plain):
        return "idea", "proposed"
    if re.search(r"\b(?:j'ouvre l'action|je crée une tâche|une action est créée)\b", plain):
        return "action", "unknown"
    if re.search(r"\b(?:je m'engage|je prends en charge|je transmettrai|j'enverrai)\b", plain):
        return "commitment", "unknown"
    if re.search(r"\b(?:décidé|décision|approuvé|approuvée|validé|validée|confirmé|confirmée|adopté|adoptée)\b", plain):
        return "decision", "unknown"
    if re.search(r"\b(?:risque|bloqué|bloquée|retard|incertain|incertaine)\b", plain):
        return "risk", "asserted"
    return "fact", "asserted"


def build(conversations, project):
    units = []
    projects = {}
    for conversation in conversations:
        sid = source_id(conversation)
        project_id = conversation.get("project_id") or project
        if not isinstance(project_id, str) or not project_id.strip():
            raise ValueError(f"{sid} : project_id absent ; fournir --project")
        projects.setdefault(project_id, [])
        for index, segment in enumerate(conversation["segments"]):
            evidence = segment["text"].strip()
            if not evidence:
                continue
            kind, status = candidate_kind(evidence)
            unit = {"unit_id": f"{sid}:{index}-{index}:0", "source_id": sid,
                    "source_date": conversation.get("source_date"), "type": kind,
                    "statement": evidence, "evidence": evidence, "start_turn": index,
                    "end_turn": index, "status": status, "speaker": segment.get("speaker"),
                    "owner": None, "due_date": None, "project_id": project_id}
            units.append(unit)
            projects[project_id].append(unit["unit_id"])
    topics = []
    for index, (project_id, ids) in enumerate(sorted(projects.items()), 1):
        if not ids:
            continue
        aliases = [project_id]
        for word in ("budget", "livraison", "planning", "risque", "dossier", "date"):
            if any(word in unit["statement"].casefold() for unit in units if unit["project_id"] == project_id):
                aliases.append(word)
        topics.append({"topic_id": f"draft_{index}", "project_id": project_id,
                       "aliases": aliases, "unit_ids": ids})
    suggestions = []
    for index, newer in enumerate(units):
        for older in units[max(0, index - 40):index]:
            if newer["project_id"] != older["project_id"] or overlap(newer["evidence"], older["evidence"], newer["project_id"]) < 0.2:
                continue
            proposal = predict_pair(newer, older)
            if proposal["relation"] not in (None, "unrelated"):
                suggestions.append(proposal)
    return units, topics, suggestions


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap(*, conversations_path=None, text_dir=None, project=None, source_date=None, output_dir):
    if (conversations_path is None) == (text_dir is None):
        raise ValueError("Choisir exactement --conversations ou --text-dir")
    if text_dir:
        conversations = text_conversations(text_dir, project, source_date)
    else:
        conversations = list(read_jsonl(conversations_path))
    if len(conversations) < 2:
        raise ValueError("Fournir au moins deux conversations")
    if len({source_id(row) for row in conversations}) != len(conversations):
        raise ValueError("Identifiants de conversation dupliqués")
    for row in conversations:
        if not isinstance(row.get("segments"), list) or not row["segments"]:
            raise ValueError(f"Segments absents : {source_id(row)}")
    units, topics, suggestions = build(conversations, project)
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_units({source_id(row): row for row in conversations}, units, schema)
    if not topics:
        raise ValueError("Aucun sujet produit")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Dossier de sortie non vide : {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "conversations.jsonl", conversations)
    write_jsonl(output_dir / "units.jsonl", units)
    write_jsonl(output_dir / "topics.jsonl", topics)
    write_jsonl(output_dir / "relations.jsonl", [])
    write_jsonl(output_dir / "relation_suggestions.jsonl", suggestions)
    owner_suggestions = [{"unit_id": row["unit_id"], "owner_candidate": candidate,
                          "reviewed": False, "evidence": row["evidence"]}
                         for row in units if row["type"] in {"action", "commitment"}
                         if (candidate := predict_owner(row["statement"]))]
    write_jsonl(output_dir / "owner_suggestions.jsonl", owner_suggestions)
    return {"conversations": len(conversations), "units": len(units), "topics": len(topics),
            "relation_suggestions": len(suggestions),
            "owner_suggestions": len(owner_suggestions), "output_dir": str(output_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--conversations", type=Path)
    source.add_argument("--text-dir", type=Path)
    parser.add_argument("--project", help="Nom du projet si absent des conversations")
    parser.add_argument("--source-date", help="Date ISO appliquée aux fichiers texte uniquement")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = bootstrap(conversations_path=args.conversations, text_dir=args.text_dir,
                           project=args.project, source_date=args.source_date,
                           output_dir=args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
