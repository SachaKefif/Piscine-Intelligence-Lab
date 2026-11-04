"""EF-09: resolve an email addressee only from the user's choice or reviewed ownership.

The local demonstration has no contact directory. Unknown email addresses remain
null. A speaker, a person merely mentioned, and a budget approver are not owners.
"""

import argparse
import json
import re
from datetime import date, timedelta

from generate_response import ResponseEngine, normalized, reference, select_scope
from parse_intent import extract_named_recipient, extract_subject
from semantic_memory import chronological_key


MAX_AGE_DAYS = 90
ROLE_STOPWORDS = {"de", "du", "des", "la", "le", "les", "l", "un", "une", "au", "aux",
                  "pour", "sur", "a", "mail", "email", "courriel", "prototype"}


def role_terms(request, projects):
    """Return the role's object, not incidental names elsewhere in the request."""
    plain = normalized(request)
    match = re.search(r"\b(?:responsable|chargee?|referente?)\b", plain)
    if not match:
        return None
    detail = re.match(r"\s+(?:de|du|des|d)\b\s+(.+)", plain[match.end():])
    if not detail:
        return []
    phrase = re.split(r"\s+(?:sur|pour|avec|concernant)\b", detail.group(1), maxsplit=1)[0]
    stop = ROLE_STOPWORDS | {normalized(project) for project in projects}
    return [word for word in phrase.split() if word not in stop]


def as_date(value, units):
    if value is None:
        known = [row["source_date"] for row in units.values() if row["source_date"]]
        return date.fromisoformat(max(known)) if known else None
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def unresolved(reason, *, as_of=None, candidates=None):
    return {"status": "clarify", "reason": reason, "name": None, "email": None,
            "source_unit_ids": [], "references": [], "as_of": as_of,
            "candidates": candidates or []}


def validate_resolution(result, *, engine, request, max_age_days=MAX_AGE_DAYS):
    """Check that a chosen role really has current, exact ownership evidence."""
    if result["email"] is not None:
        raise ValueError("Adresse de contact non attestée")
    if result["status"] == "clarify":
        if result["name"] is not None or result["source_unit_ids"] or result["references"]:
            raise ValueError("Une abstention ne choisit personne")
        return True
    if result["status"] == "explicit_user":
        if (result["name"] != extract_named_recipient(request)
                or result["source_unit_ids"] or result["references"]):
            raise ValueError("Destinataire explicite modifié")
        return True
    if result["status"] != "resolved":
        raise ValueError("Statut de destinataire inconnu")
    ids = result["source_unit_ids"]
    if not ids or len(ids) != len(set(ids)) or any(uid not in engine.units for uid in ids):
        raise ValueError("Attribution sans unités valides")
    if {row["unit_id"] for row in result["references"]} != set(ids):
        raise ValueError("Références d'attribution incomplètes")
    anchor = date.fromisoformat(result["as_of"])
    obsolete = {row["to_id"] for row in engine.relations if row["relation"] == "supersedes"}
    for uid in ids:
        unit = engine.units[uid]
        if (unit["owner"] != result["name"] or unit["status"] != "confirmed"
                or unit["type"] not in {"action", "commitment"} or uid in obsolete
                or not unit["source_date"]):
            raise ValueError("Attribution non explicite ou remplacée")
        age = anchor - date.fromisoformat(unit["source_date"])
        if age < timedelta(0) or age > timedelta(days=max_age_days):
            raise ValueError("Attribution hors fenêtre de fraîcheur")
    if any(row["unit_id"] not in engine.units
           or row != reference(engine.units[row["unit_id"]]) for row in result["references"]):
        raise ValueError("Preuve d'attribution modifiée")
    return True


def resolve_recipient(request, *, engine, subject=None, selected=None,
                      as_of=None, max_age_days=MAX_AGE_DAYS):
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Demande vide")
    if max_age_days < 0:
        raise ValueError("La fenêtre de fraîcheur doit être positive")
    anchor = as_date(as_of, engine.units)
    anchor_text = anchor.isoformat() if anchor else None
    named = extract_named_recipient(request)
    if named:
        # This is the user's direct choice, not an inferred project role.
        return {"status": "explicit_user", "reason": "named_by_user", "name": named,
                "email": None, "source_unit_ids": [], "references": [],
                "as_of": anchor_text, "candidates": []}

    terms = role_terms(request, {topic["project_id"] for topic in engine.topics.values()})
    if terms is None:
        return unresolved("recipient_not_designated", as_of=anchor_text)
    if not terms:
        return unresolved("role_not_specific", as_of=anchor_text)
    if selected is None:
        selected, scope_reason = select_scope(request, subject or extract_subject(request), engine.topics)
        if scope_reason in {"project_only", "ambiguous_project", "unknown_project"}:
            return unresolved("ambiguous_or_unknown_subject", as_of=anchor_text)
    if not selected:
        return unresolved("unknown_subject", as_of=anchor_text)
    if anchor is None:
        return unresolved("undated_attribution", as_of=None)

    ids = {uid for topic in selected for uid in topic["unit_ids"]}
    obsolete = {row["to_id"] for row in engine.relations if row["relation"] == "supersedes"}
    matching = []
    stale = False
    for uid in ids:
        unit = engine.units[uid]
        if (not unit["owner"] or unit["type"] not in {"commitment", "action"}
                or unit["status"] != "confirmed" or uid in obsolete):
            continue
        words = set(normalized(unit["statement"]).split())
        if not all(word in words for word in terms):
            continue
        if not unit["source_date"]:
            stale = True
            continue
        age = anchor - date.fromisoformat(unit["source_date"])
        if age < timedelta(0) or age > timedelta(days=max_age_days):
            stale = True
            continue
        matching.append(unit)
    if not matching:
        return unresolved("stale_attribution" if stale else "no_explicit_owner",
                          as_of=anchor_text)
    owners = {unit["owner"] for unit in matching}
    if len(owners) != 1:
        return unresolved("competing_owners", as_of=anchor_text,
                          candidates=sorted(owners))
    chosen = sorted(matching, key=chronological_key)
    return {"status": "resolved", "reason": "recent_explicit_owner",
            "name": next(iter(owners)), "email": None,
            "source_unit_ids": [row["unit_id"] for row in chosen],
            "references": [reference(row) for row in chosen],
            "as_of": anchor_text, "candidates": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Demande de mail avec destinataire nommé ou rôle")
    parser.add_argument("--as-of", help="Date ISO de référence ; défaut : dernière source chargée")
    parser.add_argument("--max-age-days", type=int, default=MAX_AGE_DAYS)
    args = parser.parse_args()
    try:
        result = resolve_recipient(args.request, engine=ResponseEngine(),
                                   as_of=args.as_of, max_age_days=args.max_age_days)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
