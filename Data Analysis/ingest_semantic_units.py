"""Validate curated semantic units against normalized conversations, then ingest them.

The transcript provides provenance; the curated file provides semantic judgments.
QMSum relevant spans must not be converted automatically into decisions or statuses.
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATIONS = ROOT / "Data" / "Synthetic" / "atlas_conversations.jsonl"
DEFAULT_UNITS = ROOT / "Data" / "Curated" / "atlas_semantic_units.jsonl"
DEFAULT_OUTPUT = ROOT / "Data" / "Processed" / "semantic_units.jsonl"
SCHEMA_PATH = ROOT / "docs" / "schemas" / "semantic_unit.schema.json"


def read_jsonl(path):
    if not path.is_file():
        raise ValueError(f"Fichier introuvable : {path}")
    with path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number} : JSON invalide : {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number} : chaque ligne doit être un objet JSON")
            yield row


def source_id(conversation):
    dataset = conversation.get("dataset")
    meeting_id = conversation.get("meeting_id")
    if not isinstance(dataset, str) or not dataset or not isinstance(meeting_id, str) or not meeting_id:
        raise ValueError("Conversation sans dataset ou meeting_id valide")
    return f"{dataset}:{meeting_id}"


def load_conversations(path):
    conversations = {}
    for conversation in read_jsonl(path):
        identifier = source_id(conversation)
        if identifier in conversations:
            raise ValueError(f"Source dupliquée : {identifier}")
        segments = conversation.get("segments")
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{identifier} : liste de segments vide ou absente")
        for number, segment in enumerate(segments):
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise ValueError(f"{identifier} : segment {number} invalide")
        conversations[identifier] = conversation
    if len(conversations) < 2:
        raise ValueError("EF-01 demande au moins deux conversations distinctes")
    return conversations


def check_schema(row, schema):
    identifier = row.get("unit_id") or row.get("from_id") or "<sans identifiant>"
    required = set(schema["required"])
    properties = schema["properties"]
    missing = required - row.keys()
    extra = row.keys() - properties.keys()
    if missing or extra:
        raise ValueError(f"{identifier} : champs manquants {sorted(missing)}, inattendus {sorted(extra)}")
    for key, specification in properties.items():
        value = row[key]
        types = specification.get("type")
        if isinstance(types, str):
            types = [types]
        if types:
            valid = any(
                (kind == "string" and isinstance(value, str))
                or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
                or (kind == "null" and value is None)
                for kind in types
            )
            if not valid:
                raise ValueError(f"{identifier} : type invalide pour {key}")
        if "enum" in specification and value not in specification["enum"]:
            raise ValueError(f"{identifier} : valeur invalide pour {key} : {value!r}")
        if isinstance(value, str) and len(value) < specification.get("minLength", 0):
            raise ValueError(f"{identifier} : {key} ne peut pas être vide")
        if isinstance(value, int) and value < specification.get("minimum", float("-inf")):
            raise ValueError(f"{identifier} : {key} hors limites")
        if specification.get("format") == "date" and value is not None:
            try:
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) or date.fromisoformat(value).isoformat() != value:
                    raise ValueError
            except ValueError as error:
                raise ValueError(f"{identifier} : date ISO invalide pour {key} : {value!r}") from error


def validate_units(conversations, units, schema):
    if not units:
        raise ValueError("Aucune unité sémantique")
    seen = set()
    covered_sources = set()
    for row in units:
        check_schema(row, schema)
        identifier = row["unit_id"]
        if identifier in seen:
            raise ValueError(f"Identifiant d'unité dupliqué : {identifier}")
        seen.add(identifier)
        source = row["source_id"]
        if source not in conversations:
            raise ValueError(f"{identifier} : source absente des conversations : {source}")
        conversation = conversations[source]
        if row["source_date"] != conversation.get("source_date"):
            raise ValueError(f"{identifier} : source_date différente de la conversation")
        if row["start_turn"] > row["end_turn"] or row["end_turn"] >= len(conversation["segments"]):
            raise ValueError(f"{identifier} : bornes de tours invalides")
        id_pattern = rf"{re.escape(source)}:{row['start_turn']}-{row['end_turn']}:\d+"
        if not re.fullmatch(id_pattern, identifier):
            raise ValueError(f"{identifier} : identifiant incohérent avec la source et les tours")
        segments = conversation["segments"][row["start_turn"] : row["end_turn"] + 1]
        evidence_text = "\n".join(segment["text"] for segment in segments)
        if row["evidence"] not in evidence_text:
            raise ValueError(f"{identifier} : preuve introuvable dans les tours indiqués")
        speakers = {segment.get("speaker") for segment in segments}
        if row["speaker"] is not None and row["speaker"] not in speakers:
            raise ValueError(f"{identifier} : locuteur absent des tours indiqués")
        covered_sources.add(source)
    if len(covered_sources) < 2:
        raise ValueError("Les unités doivent provenir d'au moins deux conversations distinctes")
    return covered_sources


def ingest(conversations_path, units_path, output_path):
    conversations = load_conversations(conversations_path)
    units = list(read_jsonl(units_path))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    covered_sources = validate_units(conversations, units, schema)
    if output_path.resolve() in {units_path.resolve(), conversations_path.resolve()}:
        raise ValueError("Le fichier de sortie ne doit pas écraser les sources")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in units:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    return len(units), len(covered_sources)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        unit_count, source_count = ingest(args.conversations, args.units, args.output)
    except ValueError as error:
        parser.error(str(error))
    print(f"Validé {unit_count} unités de {source_count} conversations ; index : {args.output}")
    print("La fidélité des reformulations, statuts et responsables exige aussi une revue humaine.")


if __name__ == "__main__":
    main()
