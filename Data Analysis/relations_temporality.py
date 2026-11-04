"""Validate reviewed relations and determine the current supported decision.

Relations are curated judgments with exact source evidence. Unreviewed pairs stay
unknown; semantic similarity alone never creates a relation or a current decision.
"""

import argparse
import json
from pathlib import Path

from ingest_semantic_units import (
    DEFAULT_CONVERSATIONS,
    DEFAULT_OUTPUT,
    ROOT,
    SCHEMA_PATH as UNIT_SCHEMA_PATH,
    check_schema,
    load_conversations,
    read_jsonl,
    validate_units,
)
from predict_relations import predict_pair


DEFAULT_RELATIONS = ROOT / "Data" / "Curated" / "atlas_relations.jsonl"
DEFAULT_TOPICS = ROOT / "Data" / "Curated" / "atlas_decision_topics.jsonl"
RELATION_SCHEMA_PATH = ROOT / "docs" / "schemas" / "relation.schema.json"
SYMMETRIC = {"contradicts", "unrelated"}


def load_and_validate_units(conversations_path, units_path):
    conversations = load_conversations(conversations_path)
    units = list(read_jsonl(units_path))
    schema = json.loads(UNIT_SCHEMA_PATH.read_text(encoding="utf-8"))
    validate_units(conversations, units, schema)
    return conversations, {unit["unit_id"]: unit for unit in units}


def validate_relations(conversations, units_by_id, relations):
    schema = json.loads(RELATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    seen = set()
    for relation in relations:
        check_schema(relation, schema)
        source_id = relation["evidence_source_id"]
        pair = (relation["from_id"], relation["to_id"])
        if pair in seen:
            raise ValueError(f"Relation dupliquée pour {pair}")
        seen.add(pair)
        if pair[0] == pair[1]:
            raise ValueError(f"Relation réflexive interdite : {pair[0]}")
        if pair[0] not in units_by_id or pair[1] not in units_by_id:
            raise ValueError(f"Unité inconnue dans la relation {pair}")
        if source_id not in conversations:
            raise ValueError(f"Source de relation inconnue : {source_id}")
        kind = relation["relation"]
        evidence = relation["evidence"]
        if kind == "unrelated":
            if evidence is not None:
                raise ValueError(f"{pair} : unrelated doit avoir evidence=null")
        else:
            if not evidence or not any(
                evidence in segment["text"] for segment in conversations[source_id]["segments"]
            ):
                raise ValueError(f"{pair} : preuve du lien introuvable dans {source_id}")
        older = units_by_id[pair[1]]
        newer = units_by_id[pair[0]]
        if kind == "supersedes":
            if newer["status"] != "confirmed":
                raise ValueError(f"{pair} : un remplacement doit venir d'une unité confirmée")
            if not newer["source_date"] or not older["source_date"]:
                raise ValueError(f"{pair} : impossible de prouver l'ordre sans les dates")
            if newer["source_date"] < older["source_date"]:
                raise ValueError(f"{pair} : remplacement antérieur à l'information remplacée")
            if newer["source_date"] == older["source_date"]:
                if newer["source_id"] != older["source_id"] or newer["start_turn"] <= older["end_turn"]:
                    raise ValueError(f"{pair} : ordre non prouvé à date égale")
        if kind == "creates_action" and (newer["type"] not in {"decision", "commitment"} or older["type"] != "action"):
            raise ValueError(f"{pair} : creates_action attend décision/engagement → action")
        if kind == "answers" and (newer["type"] != "answer" or older["type"] != "question"):
            raise ValueError(f"{pair} : answers attend réponse → question")
    return {relation["relation"] for relation in relations}


def validate_topics(units_by_id, topics):
    seen = set()
    for topic in topics:
        identifier = topic.get("topic_id")
        ids = topic.get("unit_ids")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError(f"topic_id absent ou dupliqué : {identifier!r}")
        seen.add(identifier)
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(unit_id, str) or not unit_id for unit_id in ids)
                or len(ids) != len(set(ids))):
            raise ValueError(f"{identifier} : unit_ids absents ou dupliqués")
        if not set(ids) <= units_by_id.keys():
            raise ValueError(f"{identifier} : unité inconnue dans le sujet")
        project_id = topic.get("project_id")
        if not isinstance(project_id, str) or not project_id:
            raise ValueError(f"{identifier} : project_id absent")
        if any(units_by_id[unit_id]["project_id"] != project_id for unit_id in ids):
            raise ValueError(f"{identifier} : unités de projets différents")
    return {topic["topic_id"]: topic for topic in topics}


def classify_pair(from_id, to_id, relations):
    for relation in relations:
        if relation["from_id"] == from_id and relation["to_id"] == to_id:
            return {"from_id": from_id, "to_id": to_id, "relation": relation["relation"],
                    "evidence_source_id": relation["evidence_source_id"], "evidence": relation["evidence"],
                    "reviewed": True}
        if relation["from_id"] == to_id and relation["to_id"] == from_id and relation["relation"] in SYMMETRIC:
            return {"from_id": from_id, "to_id": to_id, "relation": relation["relation"],
                    "evidence_source_id": relation["evidence_source_id"], "evidence": relation["evidence"],
                    "reviewed": True, "annotation_direction_reversed": True}
    return {"from_id": from_id, "to_id": to_id, "relation": None,
            "reviewed": False, "reason": "Paire non annotée : revue nécessaire."}


def unit_summary(unit):
    return {key: unit[key] for key in (
        "unit_id", "statement", "type", "status", "source_id", "source_date", "evidence"
    )}


def current_decision(topic, units_by_id, relations):
    topic_ids = set(topic["unit_ids"])
    history = [unit_summary(units_by_id[unit_id]) for unit_id in topic["unit_ids"]]
    confirmed = {
        unit_id for unit_id in topic_ids
        if units_by_id[unit_id]["type"] == "decision"
        and units_by_id[unit_id]["status"] == "confirmed"
        and units_by_id[unit_id]["evidence"]
    }
    superseded = {
        relation["to_id"] for relation in relations
        if relation["relation"] == "supersedes"
        and relation["to_id"] in confirmed
        and units_by_id[relation["from_id"]]["project_id"] == topic["project_id"]
    }
    active = confirmed - superseded
    disputed = {
        decision_id for relation in relations
        if relation["relation"] == "contradicts"
        for decision_id in (relation["from_id"], relation["to_id"])
        if decision_id in active
    }
    response = {"topic_id": topic["topic_id"], "project_id": topic["project_id"], "history": history}
    if not active:
        response.update({"state": "no_confirmed_decision", "decision": None,
                         "reason": "Aucune décision confirmée et non remplacée avec preuve."})
    elif len(active) > 1 or disputed:
        response.update({"state": "ambiguous", "decision": None,
                         "candidates": [unit_summary(units_by_id[unit_id]) for unit_id in topic["unit_ids"] if unit_id in active],
                         "reason": "Décisions concurrentes ou contradiction non résolue par un remplacement prouvé."})
    else:
        selected_id = next(iter(active))
        if units_by_id[selected_id]["source_date"] is None:
            response.update({"state": "undated", "decision": unit_summary(units_by_id[selected_id]),
                             "reason": "Décision confirmée et sourcée, mais date de source inconnue : état courant non établi."})
        else:
            response.update({"state": "current", "decision": unit_summary(units_by_id[selected_id]),
                             "reason": "Décision confirmée, sourcée et non remplacée par une autre décision prouvée."})
    response["relations"] = [relation for relation in relations
                             if (relation["from_id"] in topic_ids and relation["to_id"] in topic_ids)
                             or (relation["relation"] == "contradicts" and
                                 (relation["from_id"] in active or relation["to_id"] in active))]
    return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    pair_parser = commands.add_parser("classify")
    pair_parser.add_argument("--from-id", required=True)
    pair_parser.add_argument("--to-id", required=True)
    predict_parser = commands.add_parser("predict", help="Proposer un lien pour une paire non annotée")
    predict_parser.add_argument("--from-id", required=True)
    predict_parser.add_argument("--to-id", required=True)
    current_parser = commands.add_parser("current")
    current_parser.add_argument("topic_id")
    args = parser.parse_args()
    try:
        conversations, units_by_id = load_and_validate_units(args.conversations, args.units)
        relations = list(read_jsonl(args.relations))
        labels = validate_relations(conversations, units_by_id, relations)
        topics = validate_topics(units_by_id, list(read_jsonl(args.topics)))
        if args.command == "validate":
            response = {"units": len(units_by_id), "relations": len(relations), "relation_types": sorted(labels),
                        "topics": len(topics), "status": "validated"}
        elif args.command in {"classify", "predict"}:
            if args.from_id not in units_by_id or args.to_id not in units_by_id:
                raise ValueError("Unité inconnue dans la paire demandée")
            reviewed = classify_pair(args.from_id, args.to_id, relations)
            response = (reviewed if args.command == "classify" or reviewed["reviewed"] else
                        predict_pair(units_by_id[args.from_id], units_by_id[args.to_id]))
        else:
            if args.topic_id not in topics:
                raise ValueError(f"Sujet inconnu : {args.topic_id}")
            response = current_decision(topics[args.topic_id], units_by_id, relations)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
