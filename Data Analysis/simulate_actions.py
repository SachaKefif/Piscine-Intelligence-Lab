"""EF-08: prepare evidence-backed email, task and event drafts without side effects."""

import argparse
import json

from bm25_score_qmsum import bm25_scores
from generate_response import ResponseEngine, contains_phrase, normalized, reference, select_scope
from parse_intent import classify, parse_intent
from resolve_recipient import resolve_recipient, validate_resolution
from semantic_memory import chronological_key, topic_memory


TOOLS = {"draft_email": "email.compose_draft", "create_task": "tasks.create_draft",
         "schedule_event": "calendar.create_draft"}


def clarification(request, reason, *, intent="clarify", recipient=None):
    return {
        "request": request, "intent": intent, "status": "clarify",
        "clarification_reason": reason, "tool_candidate": None, "parameters": None,
        "content": None, "recipient": recipient, "source_unit_ids": [], "references": [],
        "needs_confirmation": True, "executed": False,
    }


def supported_context(topic, engine, request, retrieved_ids=None):
    """Choose a current decision or a confirmed active action, never a disputed fact."""
    memory = topic_memory(topic, engine.units, engine.relations)
    if memory["contradictions"] or memory["current_decision"]["state"] == "ambiguous":
        return None, "conflicting_evidence"
    decision = memory["current_decision"]
    if decision["state"] == "current":
        return engine.units[decision["decision"]["unit_id"]], None
    excluded = set(memory["obsolete_unit_ids"])
    candidates = [engine.units[uid] for uid in topic["unit_ids"]
                  if uid not in excluded and engine.units[uid]["status"] == "confirmed"
                  and engine.units[uid]["type"] in {"action", "commitment"}]
    if candidates:
        scores = bm25_scores([row["statement"] for row in candidates], request)
        positions = {uid: rank for rank, uid in enumerate(retrieved_ids or [])}
        index = min(range(len(candidates)), key=lambda i: (
            positions.get(candidates[i]["unit_id"], len(positions)),
            -scores[i], chronological_key(candidates[i])
        )) if retrieved_ids else max(range(len(candidates)),
                                     key=lambda i: (scores[i], chronological_key(candidates[i])))
        return candidates[index], None
    return None, "no_confirmed_context"


def validate_draft(draft, units):
    """Reject any executable output, missing source, or modified evidence quote."""
    if draft.get("executed") is not False or draft.get("needs_confirmation") is not True:
        raise ValueError("Les actions externes sont interdites")
    if draft["status"] == "clarify":
        if (draft["tool_candidate"] is not None or draft["content"] is not None
                or draft["parameters"] is not None or draft["source_unit_ids"]
                or draft["references"]):
            raise ValueError("Une clarification ne doit pas proposer d'opération")
        return True
    if draft["status"] != "draft" or draft["intent"] not in TOOLS:
        raise ValueError("Type de brouillon invalide")
    if draft["tool_candidate"] != TOOLS[draft["intent"]]:
        raise ValueError("Opération candidate incohérente")
    ids = draft["source_unit_ids"]
    if not ids or len(ids) != len(set(ids)) or any(uid not in units for uid in ids):
        raise ValueError("Brouillon sans unités sources valides")
    if {row["unit_id"] for row in draft["references"]} != set(ids):
        raise ValueError("Références du brouillon incomplètes")
    if any(row["unit_id"] not in units or row != reference(units[row["unit_id"]])
           for row in draft["references"]):
        raise ValueError("Extrait de preuve modifié")
    context_id = draft["context_unit_id"]
    if context_id not in ids or units[context_id]["statement"] not in draft["content"]:
        raise ValueError("Contexte cité absent du contenu")
    if f"[{context_id}]" not in draft["content"]:
        raise ValueError("Citation absente du contenu")
    if draft["intent"] == "draft_email":
        recipient = draft["recipient"]
        if not recipient or recipient["status"] not in {"resolved", "explicit_user"}:
            raise ValueError("Destinataire non résolu")
        if recipient["email"] is not None:
            raise ValueError("Adresse non attestée")
        if not set(recipient["source_unit_ids"]) <= set(ids):
            raise ValueError("Justification du destinataire manquante")
        if draft["parameters"]["to_name"] != recipient["name"]:
            raise ValueError("Nom du destinataire incohérent")
        if draft["parameters"]["body"] != draft["content"] or draft["parameters"]["to_email"] is not None:
            raise ValueError("Contenu ou adresse du mail incohérent")
    elif draft["intent"] == "create_task" and not draft["parameters"]["action"]:
        raise ValueError("Action de tâche manquante")
    elif draft["intent"] == "schedule_event" and not draft["parameters"]["date"]:
        raise ValueError("Date d'événement manquante")
    return True


class ActionSimulator:
    def __init__(self, engine=None):
        self.engine = engine or ResponseEngine()

    def prepare(self, request, *, parsed=None, as_of=None, retrieved_ids=None):
        parsed = parsed or parse_intent(request)
        intent = parsed["intent"]
        if parsed["clarification_reason"] == "hypothetical_not_request":
            return clarification(request, "hypothetical_not_request")
        selected, scope_reason = select_scope(request, parsed["subject"], self.engine.topics)
        project_named = any(contains_phrase(normalized(request), project)
                            for project in {topic["project_id"] for topic in self.engine.topics.values()})
        if (intent == "clarify" and parsed["clarification_reason"] in
                {"missing_named_recipient", "missing_subject"}
                and classify(request)[0] == "draft_email" and selected and project_named):
            # A role may become resolvable from EF-09 even when EF-06 could not
            # extract a named recipient. No other clarification is overridden.
            intent = "draft_email"
        if intent == "clarify":
            return clarification(request, parsed["clarification_reason"] or "uncertain_intent")
        if intent not in TOOLS:
            return clarification(request, "not_an_action_intent", intent=intent)
        if len(selected) != 1 or scope_reason in {
            "project_only", "ambiguous_project", "unknown_project", "unknown_subject"
        }:
            return clarification(request, "ambiguous_or_unknown_subject", intent=intent)
        if intent == "create_task" and not parsed["action"]:
            return clarification(request, "missing_action", intent=intent)
        if intent == "schedule_event" and not parsed["due_date"]:
            return clarification(request, "missing_due_date", intent=intent)

        topic = selected[0]
        context, context_error = supported_context(topic, self.engine, request, retrieved_ids)
        if context is None:
            return clarification(request, context_error, intent=intent)
        recipient = None
        if intent == "draft_email":
            recipient = resolve_recipient(request, engine=self.engine, selected=selected,
                                          subject=parsed["subject"], as_of=as_of)
            validate_resolution(recipient, engine=self.engine, request=request)
            if recipient["status"] == "clarify":
                return clarification(request, recipient["reason"], intent=intent,
                                     recipient=recipient)

        subject = parsed["subject"] or f"{topic['aliases'][0]} {topic['project_id']}"
        evidence_line = f"Contexte confirmé : {context['statement']} [{context['unit_id']}]"
        if intent == "draft_email":
            content = (f"Bonjour {recipient['name']},\n\nJe vous écris au sujet de {subject}.\n"
                       f"{evidence_line}\n\nCordialement,")
            parameters = {"to_name": recipient["name"], "to_email": None,
                          "subject": subject, "body": content}
        elif intent == "create_task":
            content = f"Tâche demandée : {parsed['action']}.\n{evidence_line}"
            parameters = {"action": parsed["action"], "due_date": parsed["due_date"],
                          "assignee_name": None}
        else:
            content = f"Réunion demandée sur {subject} le {parsed['due_date']}.\n{evidence_line}"
            parameters = {"title": f"Réunion — {subject}", "date": parsed["due_date"],
                          "time": None, "attendees": []}
        source_ids = {context["unit_id"]}
        if recipient:
            source_ids.update(recipient["source_unit_ids"])
        result = {
            "request": request, "intent": intent, "status": "draft",
            "clarification_reason": None, "tool_candidate": TOOLS[intent],
            "parameters": parameters, "content": content, "recipient": recipient,
            "context_unit_id": context["unit_id"],
            "source_unit_ids": sorted(source_ids),
            "references": [reference(self.engine.units[uid]) for uid in sorted(source_ids)],
            "needs_confirmation": True, "executed": False,
        }
        validate_draft(result, self.engine.units)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Demande de mail, tâche ou événement simulé")
    parser.add_argument("--as-of", help="Date ISO de référence pour EF-09")
    args = parser.parse_args()
    try:
        result = ActionSimulator().prepare(args.request, as_of=args.as_of)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
