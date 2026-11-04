"""EF-07: evidence-backed answers, comparisons and dossiers from reviewed memory.

Every factual claim is copied from a semantic unit or derived from a reviewed
relation. No unreviewed similarity, proposed date or unresolved conflict becomes
a current fact. The output is structured JSON; no external action is performed.
"""

import argparse
import json
import re
from pathlib import Path

from bm25_score_qmsum import bm25_scores
from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS
from parse_intent import classify, folded, parse_intent
from relations_temporality import DEFAULT_RELATIONS
from semantic_memory import DEFAULT_MEMORY_TOPICS, chronological_key, load_memory, topic_memory


ANSWER_INTENTS = {"answer", "compare", "build_dossier"}
SECTIONS = ("answer", "context", "facts", "chronology", "decisions", "risks",
            "open_points", "responsibles", "changes")
RELATION_TEXT = {
    "supersedes": "Une information confirmée remplace l'information antérieure.",
    "contradicts": "Deux informations se contredisent ; la contradiction reste à résoudre.",
}


def normalized(text):
    return re.sub(r"[^\w]+", " ", folded(text)).strip()


def contains_phrase(text, phrase):
    return bool(re.search(rf"(?<!\w){re.escape(normalized(phrase))}(?!\w)", text))


def empty_response(request, intent, status, message, *, reason=None):
    return {
        "request": request, "intent": intent, "status": status, "message": message,
        "clarification_reason": reason, "project_id": None, "topic_ids": [],
        "sections": {name: [] for name in SECTIONS}, "references": [], "missing": [],
    }


def unit_claim(unit, *, role=None):
    claim = {"text": unit["statement"], "source_unit_ids": [unit["unit_id"]],
             "kind": "unit", "status": unit["status"]}
    if role:
        claim["role"] = role
    if unit.get("owner"):
        claim["owner"] = unit["owner"]
    if unit.get("due_date"):
        claim["due_date"] = unit["due_date"]
    return claim


def relation_claim(relation):
    return {
        "text": RELATION_TEXT[relation["relation"]],
        "source_unit_ids": [relation["from_id"], relation["to_id"]],
        "kind": "reviewed_relation", "relation": relation["relation"],
        "evidence_source_id": relation["evidence_source_id"],
    }


def reference(unit):
    return {key: unit[key] for key in (
        "unit_id", "source_id", "source_date", "start_turn", "end_turn",
        "statement", "evidence", "type", "status", "speaker", "owner", "due_date",
    )}


def select_scope(request, subject, topics):
    """Match reviewed topic aliases; a project alone is a dossier/compare scope."""
    query = normalized(request)
    projects = {topic["project_id"] for topic in topics.values()}
    mentioned = {project for project in projects if contains_phrase(query, project)}
    if len(mentioned) > 1:
        return [], "ambiguous_project"
    project = next(iter(mentioned), None)
    selected = []
    for topic in topics.values():
        if project and topic["project_id"] != project:
            continue
        aliases = topic.get("aliases", [])
        if any(contains_phrase(query, alias) for alias in aliases):
            selected.append(topic)
    # EF-06 can identify an unknown project in e.g. "budget Orion". Do not
    # silently answer from Atlas merely because the word "budget" matched.
    if subject:
        final_word = normalized(subject).split()[-1]
        if final_word not in {normalized(p) for p in projects} and not any(
            contains_phrase(normalized(alias), final_word)
            for topic in selected for alias in topic.get("aliases", [])
        ):
            return [], "unknown_project"
    if selected and len({topic["project_id"] for topic in selected}) > 1:
        return [], "ambiguous_project"
    if selected:
        return selected, None
    if project:
        return [topic for topic in topics.values() if topic["project_id"] == project], "project_only"
    return [], "unknown_subject"


def relevant_relations(relations, unit_ids):
    return [row for row in relations if row["relation"] in RELATION_TEXT
            and row["from_id"] in unit_ids and row["to_id"] in unit_ids]


def unique_claims(claims):
    seen = set()
    result = []
    for claim in claims:
        key = (claim["kind"], tuple(claim["source_unit_ids"]), claim.get("role"))
        if key not in seen:
            result.append(claim)
            seen.add(key)
    return result


def answer_claims(request, selected, memories, units, relations, retrieved_ids=None):
    ids = {unit_id for topic in selected for unit_id in topic["unit_ids"]}
    conflicts = [row for row in relevant_relations(relations, ids)
                 if row["relation"] == "contradicts"]
    if conflicts:
        involved = {identifier for row in conflicts for identifier in (row["from_id"], row["to_id"])}
        return ([unit_claim(units[unit_id], role="conflicting_statement")
                 for unit_id in sorted(involved, key=lambda uid: chronological_key(units[uid]))],
                [relation_claim(row) for row in conflicts], "uncertain")

    ambiguous = [memory for memory in memories if memory["current_decision"]["state"] == "ambiguous"]
    if ambiguous:
        candidates = [unit_claim(units[row["unit_id"]], role="decision_candidate")
                      for memory in ambiguous for row in memory["current_decision"].get("candidates", [])]
        return unique_claims(candidates), [], "uncertain"

    eligible = [units[unit_id] for unit_id in ids
                if units[unit_id]["type"] != "question"
                and units[unit_id]["status"] != "unknown"
                and not any(unit_id in memory["obsolete_unit_ids"] for memory in memories)]
    eligible.sort(key=chronological_key)
    query = normalized(request)
    if re.search(r"\b(propos\w*|envisag\w*|initial\w*)\b", query):
        proposals = [units[unit_id] for unit_id in ids if units[unit_id]["status"] == "proposed"]
        if proposals:
            if len(proposals) == 1:
                return [unit_claim(proposals[0], role="historical_proposal")], [], "ok"
            proposed_scores = bm25_scores([row["statement"] for row in proposals], request)
            best = proposals[max(range(len(proposals)), key=lambda index: proposed_scores[index])]
            if max(proposed_scores) > 0:
                return [unit_claim(best, role="historical_proposal")], [], "ok"
    if not eligible:
        return [], [], "not_found"
    current = [units[memory["current_decision"]["decision"]["unit_id"]]
               for memory in memories if memory["current_decision"]["state"] == "current"]
    if current and re.search(r"\b(decision|actuel|actuelle|retenu|retenue|fixee|fixe)\b", query):
        return [unit_claim(row, role="current_decision") for row in current], [], "ok"
    if current and "date" in query and "propos" not in query and "envisag" not in query:
        return [unit_claim(row, role="current_decision") for row in current], [], "ok"
    scores = bm25_scores([row["statement"] for row in eligible], request)
    retrieval_position = {uid: index for index, uid in enumerate(retrieved_ids or [])}
    ranked = sorted(range(len(eligible)), key=lambda index: (
        retrieval_position.get(eligible[index]["unit_id"], len(retrieval_position)),
        -scores[index], eligible[index]["unit_id"]
    )) if retrieved_ids else sorted(range(len(eligible)),
                                    key=lambda index: (-scores[index], eligible[index]["unit_id"]))
    if not ranked or scores[ranked[0]] <= 0:
        return [], [], "not_found"
    best = eligible[ranked[0]]
    # A bare proposal is not an answer to a question about the current state.
    if best["status"] == "proposed" and not re.search(r"\b(propos|envisag|initial|avant)\w*\b", query):
        confirmed = [row for row in eligible if row["status"] == "confirmed"]
        if confirmed:
            confirmed_scores = bm25_scores([row["statement"] for row in confirmed], request)
            best = confirmed[max(range(len(confirmed)), key=lambda index: confirmed_scores[index])]
    return [unit_claim(best)], [], "ok"


def build_sections(intent, request, selected, memories, units, relations, retrieved_ids=None):
    ids = {unit_id for topic in selected for unit_id in topic["unit_ids"]}
    ordered = sorted(ids, key=lambda uid: chronological_key(units[uid]))
    obsolete = {unit_id for memory in memories for unit_id in memory["obsolete_unit_ids"]}
    links = relevant_relations(relations, ids)
    conflicts = [row for row in links if row["relation"] == "contradicts"]
    conflicted = {uid for row in conflicts for uid in (row["from_id"], row["to_id"])}
    corroborated = {
        uid for row in relations if row["relation"] == "supports"
        for uid, other in ((row["from_id"], row["to_id"]), (row["to_id"], row["from_id"]))
        if uid in ids and other in ids and units[other]["status"] == "confirmed"
    }
    sections = {name: [] for name in SECTIONS}
    if intent == "answer":
        sections["answer"], sections["open_points"], status = answer_claims(
            request, selected, memories, units, relations, retrieved_ids)
        return sections, status

    context_ids = {uid for topic in selected for uid in topic.get("context_unit_ids", [])}
    sections["context"] = [unit_claim(units[uid], role="project_context")
                           for uid in ordered if uid in context_ids]
    sections["chronology"] = [unit_claim(units[uid], role="obsolete" if uid in obsolete else "history")
                              for uid in ordered]
    sections["changes"] = [relation_claim(row) for row in links]
    sections["facts"] = [unit_claim(units[uid]) for uid in ordered
                         if units[uid]["type"] in {"fact", "answer"}
                         and (units[uid]["status"] == "confirmed"
                              or (units[uid]["status"] == "asserted" and uid in corroborated))
                         and uid not in obsolete | conflicted]
    sections["responsibles"] = [unit_claim(units[uid]) for uid in ordered
                                if units[uid]["owner"] and units[uid]["type"] in {"action", "commitment"}]
    sections["risks"] = [unit_claim(units[uid]) for uid in ordered
                         if units[uid]["type"] == "risk" and uid not in obsolete]
    sections["open_points"] = [unit_claim(units[uid]) for uid in ordered
                               if units[uid]["type"] == "risk"
                               or (units[uid]["type"] == "question" and not any(
                                   row["relation"] == "answers" and row["to_id"] == uid
                                   for row in relations))]
    sections["open_points"].extend(relation_claim(row) for row in conflicts)
    for memory in memories:
        decision = memory["current_decision"]
        if decision["state"] == "current":
            sections["decisions"].append(unit_claim(units[decision["decision"]["unit_id"]],
                                                    role="current_decision"))
        elif decision["state"] == "ambiguous":
            sections["open_points"].extend(
                unit_claim(units[row["unit_id"]], role="decision_candidate")
                for row in decision.get("candidates", []))
    for key in SECTIONS:
        sections[key] = unique_claims(sections[key])
    if intent == "compare":
        # The timeline is grouped by source in the consumer using reference.source_id;
        # only reviewed links state that something changed between meetings.
        return sections, "ok"
    return sections, "ok"


def validate_claim(claim, units, relations):
    """Check a claim against one exact unit or one reviewed relation."""
    reviewed = {(row["relation"], row["from_id"], row["to_id"], row["evidence_source_id"])
                for row in relations}
    ids = claim.get("source_unit_ids")
    if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
            or any(uid not in units for uid in ids)):
        raise ValueError("Affirmation sans unités de preuve valides")
    if claim.get("kind") == "unit":
        if len(ids) != 1 or claim.get("text") != units[ids[0]]["statement"]:
            raise ValueError("Affirmation différente de l'unité citée")
    elif claim.get("kind") == "reviewed_relation":
        if len(ids) != 2:
            raise ValueError("Lien sans les deux unités requises")
        key = (claim.get("relation"), ids[0], ids[1], claim.get("evidence_source_id"))
        if key not in reviewed or claim.get("text") != RELATION_TEXT.get(key[0]):
            raise ValueError("Lien affirmé sans relation revue et prouvée")
    else:
        raise ValueError("Type d'affirmation inconnu")
    return True


def validate_response(response, units, relations):
    """Check the provenance contract for every generated factual claim."""
    cited = set()
    for section in response["sections"].values():
        for claim in section:
            validate_claim(claim, units, relations)
            cited.update(claim["source_unit_ids"])
    if {row["unit_id"] for row in response["references"]} != cited:
        raise ValueError("Références incomplètes ou inutilisées")
    for row in response["references"]:
        if row["unit_id"] not in units or row != reference(units[row["unit_id"]]):
            raise ValueError("Référence modifiée après validation")
    return True


class ResponseEngine:
    def __init__(self, conversations=DEFAULT_CONVERSATIONS, units_path=DEFAULT_UNITS,
                 relations_path=DEFAULT_RELATIONS, topics_path=DEFAULT_MEMORY_TOPICS):
        self.units, self.relations, self.topics = load_memory(
            conversations, units_path, relations_path, topics_path)
        for topic in self.topics.values():
            aliases = topic.get("aliases")
            if not isinstance(aliases, list) or not aliases or any(
                not isinstance(alias, str) or not normalized(alias) for alias in aliases
            ):
                raise ValueError(f"Alias de sujet absents ou invalides : {topic['topic_id']}")
            context_ids = topic.get("context_unit_ids", [])
            if (not isinstance(context_ids, list) or len(context_ids) != len(set(context_ids))
                    or any(uid not in topic["unit_ids"] or self.units[uid]["type"] != "fact"
                           or self.units[uid]["status"] != "confirmed" for uid in context_ids)):
                raise ValueError(f"Contexte de sujet invalide : {topic['topic_id']}")

    def generate(self, request, *, parsed=None, retrieved_ids=None):
        parsed = parsed or parse_intent(request)
        intent = parsed["intent"]
        selected, scope_reason = select_scope(request, parsed["subject"], self.topics)
        project_named = any(contains_phrase(normalized(request), project)
                            for project in {topic["project_id"] for topic in self.topics.values()})
        rules_intent = classify(request)[0]
        question = bool(re.match(r"^(qui|quel\w*|quand|ou|comment|pourquoi|est ce)\b",
                                 normalized(request)))
        if (intent == "clarify" and parsed["clarification_reason"] in
                {"missing_subject", "intent_uncertain_or_ambiguous"}
                and selected and project_named):
            # EF-06 cannot extract a bare project or "livraison du prototype Atlas".
            # A reviewed topic or an explicit project resolves that missing slot.
            if rules_intent in ANSWER_INTENTS and (
                parsed["clarification_reason"] == "missing_subject" or question
            ):
                intent = rules_intent
        elif intent not in ANSWER_INTENTS and question and project_named and selected:
            # An explicit information question can be mistaken for an action by
            # EF-06 because it mentions verbs such as "envoyer".
            if rules_intent == "answer":
                intent = "answer"
        if intent == "clarify":
            message = "La demande doit être précisée."
            if parsed["clarification_reason"] == "missing_subject" and rules_intent == "compare":
                message = "Quelles décisions ou quel sujet souhaitez-vous comparer ?"
            elif parsed["clarification_reason"] == "hypothetical_not_request":
                message = "Souhaitez-vous formuler une demande explicite ?"
            return empty_response(request, intent, "clarify", message,
                                  reason=parsed["clarification_reason"])
        if intent not in ANSWER_INTENTS:
            return empty_response(request, intent, "unsupported_intent",
                                  "Cette intention relève des brouillons d'action, hors EF-07.")
        if scope_reason == "ambiguous_project":
            return empty_response(request, "clarify", "clarify",
                                  "Plusieurs projets correspondent à la demande.", reason="ambiguous_project")
        if not selected:
            result = empty_response(request, intent, "not_found",
                                    "Aucune information correspondante dans la mémoire chargée.")
            result["missing"] = [parsed["subject"] or "sujet demandé"]
            return result
        if scope_reason == "project_only" and intent == "answer":
            return empty_response(request, "clarify", "clarify",
                                  "Quel sujet du projet faut-il examiner ?", reason="ambiguous_subject")
        if scope_reason == "project_only" and intent not in {"compare", "build_dossier"}:
            return empty_response(request, "clarify", "clarify",
                                  "Quel sujet du projet faut-il examiner ?", reason="ambiguous_subject")
        memories = [topic_memory(topic, self.units, self.relations) for topic in selected]
        sections, status = build_sections(intent, request, selected, memories,
                                          self.units, self.relations, retrieved_ids)
        result = empty_response(request, intent, status, None)
        result["project_id"] = selected[0]["project_id"]
        result["topic_ids"] = [topic["topic_id"] for topic in selected]
        result["sections"] = sections
        cited = {uid for section in sections.values() for claim in section
                 for uid in claim["source_unit_ids"]}
        result["references"] = [reference(self.units[uid]) for uid in sorted(cited)]
        if status == "not_found":
            result["message"] = "Information introuvable dans les unités du sujet."
            result["missing"] = [parsed["subject"] or "information demandée"]
        elif status == "uncertain":
            result["message"] = "Les preuves disponibles ne permettent pas de trancher."
        elif intent == "build_dossier":
            result["missing"] = [name for name in ("context", "facts", "decisions", "risks",
                                                    "open_points", "responsibles")
                                 if not sections[name]]
        validate_response(result, self.units, self.relations)
        return result


def render_markdown(response):
    """Present the already validated JSON without adding factual statements."""
    headings = {
        "answer": "Réponse", "context": "Contexte", "facts": "Faits établis",
        "chronology": "Chronologie", "decisions": "Décisions", "risks": "Risques",
        "open_points": "Points ouverts",
        "responsibles": "Responsables", "changes": "Changements vérifiés",
    }
    lines = [f"# {response['intent']} — {response['status']}"]
    if response["message"]:
        lines.extend(("", response["message"]))
    for section_name, heading in headings.items():
        claims = response["sections"][section_name]
        if not claims or (response["intent"] == "answer" and section_name not in {"answer", "open_points"}):
            continue
        lines.extend(("", f"## {heading}", ""))
        for claim in claims:
            ids = ", ".join(claim["source_unit_ids"])
            lines.append(f"- {claim['text']} [{ids}]")
    if response["missing"]:
        lines.extend(("", "## Informations introuvables", ""))
        lines.extend(f"- {item}" for item in response["missing"])
    if response["references"]:
        lines.extend(("", "## Références", ""))
        for row in response["references"]:
            date = row["source_date"] or "date inconnue"
            lines.append(
                f"- {row['unit_id']} — {row['source_id']}, {date}, "
                f"tours {row['start_turn']}–{row['end_turn']} : « {row['evidence']} »"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Question, comparaison ou demande de dossier")
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--topics", type=Path, default=DEFAULT_MEMORY_TOPICS)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()
    try:
        engine = ResponseEngine(args.conversations, args.units, args.relations, args.topics)
        response = engine.generate(args.request)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.format == "markdown":
        print(render_markdown(response), end="")
    else:
        print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
