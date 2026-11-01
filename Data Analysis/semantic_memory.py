"""Consult a reviewed, evidence-backed memory across conversations.

Topic membership is curated. Only validated relations can mark a unit obsolete or
contradictory; dates and semantic similarity alone never settle those judgments.
"""

import argparse
import json
from pathlib import Path

from ingest_semantic_units import ROOT, read_jsonl
from relations_temporality import (
    DEFAULT_RELATIONS,
    current_decision,
    load_and_validate_units,
    unit_summary,
    validate_relations,
    validate_topics,
)
from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_OUTPUT


DEFAULT_MEMORY_TOPICS = ROOT / "Data" / "Curated" / "atlas_memory_topics.jsonl"


def load_memory(conversations_path, units_path, relations_path, topics_path):
    """Load the four inputs and reject invalid evidence or topic references."""
    conversations, units = load_and_validate_units(conversations_path, units_path)
    relations = list(read_jsonl(relations_path))
    validate_relations(conversations, units, relations)
    topics = validate_topics(units, list(read_jsonl(topics_path)))
    for relation in relations:
        if relation["relation"] in {"supersedes", "contradicts"}:
            left = units[relation["from_id"]]["project_id"]
            right = units[relation["to_id"]]["project_id"]
            if left != right:
                raise ValueError("Un remplacement ou une contradiction relie deux projets distincts")
    return units, relations, topics


def chronological_key(unit):
    """Unknown dates sort last and are never interpreted as later events."""
    return (
        unit["source_date"] is None,
        unit["source_date"] or "",
        unit["source_id"],
        unit["start_turn"],
        unit["end_turn"],
        unit["unit_id"],
    )


def topic_memory(topic, units, relations):
    """Build a sourced timeline and explicit obsolescence/conflict signals."""
    ids = set(topic["unit_ids"])
    relevant = [
        row for row in relations
        if (row["from_id"] in ids or row["to_id"] in ids)
        and (row["relation"] != "unrelated"
             or (row["from_id"] in ids and row["to_id"] in ids))
    ]
    superseded_by = {unit_id: [] for unit_id in ids}
    contradictions = []
    for row in relevant:
        if row["relation"] == "supersedes" and row["to_id"] in ids:
            superseded_by[row["to_id"]].append(row["from_id"])
        if row["relation"] == "contradicts":
            contradictions.append({
                "from_id": row["from_id"], "to_id": row["to_id"],
                "evidence_source_id": row["evidence_source_id"],
                "evidence": row["evidence"], "state": "unresolved",
            })

    history = []
    for unit_id in sorted(ids, key=lambda identifier: chronological_key(units[identifier])):
        unit = units[unit_id]
        item = unit_summary(unit)
        item.update({
            "speaker": unit["speaker"], "owner": unit["owner"],
            "due_date": unit["due_date"],
            "start_turn": unit["start_turn"], "end_turn": unit["end_turn"],
            "memory_state": "obsolete" if superseded_by[unit_id] else "not_superseded",
            "superseded_by": sorted(superseded_by[unit_id]),
            "contradicts": sorted({
                row["to_id"] if row["from_id"] == unit_id else row["from_id"]
                for row in relevant if row["relation"] == "contradicts"
                and unit_id in (row["from_id"], row["to_id"])
            }),
        })
        history.append(item)

    # A relation can point outside this curated topic. Preserve that evidence
    # rather than silently dropping a contradiction or a replacement.
    external_ids = {
        identifier for row in relevant for identifier in (row["from_id"], row["to_id"])
        if identifier not in ids
    }
    result = {
        "topic_id": topic["topic_id"], "project_id": topic["project_id"],
        "source_ids": sorted({units[unit_id]["source_id"] for unit_id in ids}),
        "history": history,
        "relations": relevant,
        "related_units_outside_topic": [unit_summary(units[unit_id]) for unit_id in sorted(external_ids)],
        "obsolete_unit_ids": sorted(unit_id for unit_id, newer in superseded_by.items() if newer),
        "contradictions": contradictions,
        "review_needed_unit_ids": sorted(
            unit_id for unit_id in ids
            if units[unit_id]["status"] == "superseded" and not superseded_by[unit_id]
        ),
    }
    decision = current_decision(topic, units, relations)
    result["current_decision"] = {
        key: decision[key] for key in ("state", "decision", "reason")
    }
    if "candidates" in decision:
        result["current_decision"]["candidates"] = decision["candidates"]
    return result


def list_topics(topics, units):
    return [
        {
            "topic_id": topic["topic_id"], "project_id": topic["project_id"],
            "unit_count": len(topic["unit_ids"]),
            "source_ids": sorted({units[unit_id]["source_id"] for unit_id in topic["unit_ids"]}),
        }
        for topic in sorted(topics.values(), key=lambda row: row["topic_id"])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--topics", type=Path, default=DEFAULT_MEMORY_TOPICS)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate", help="Valider les sources, unités, liens et sujets")
    commands.add_parser("list", help="Lister les sujets de la mémoire")
    show = commands.add_parser("show", help="Afficher l'historique d'un sujet")
    show.add_argument("topic_id")
    args = parser.parse_args()
    try:
        units, relations, topics = load_memory(
            args.conversations, args.units, args.relations, args.topics
        )
        if args.command == "validate":
            response = {
                "status": "validated", "units": len(units),
                "relations": len(relations), "topics": len(topics),
                "cross_conversation_topics": sum(
                    len({units[unit_id]["source_id"] for unit_id in topic["unit_ids"]}) > 1
                    for topic in topics.values()
                ),
            }
        elif args.command == "list":
            response = list_topics(topics, units)
        else:
            if args.topic_id not in topics:
                raise ValueError(f"Sujet inconnu : {args.topic_id}")
            response = topic_memory(topics[args.topic_id], units, relations)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
