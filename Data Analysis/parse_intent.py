"""EF-06: trained French intent routing with conservative slot guards.

This module only interprets the user's request. It does not resolve people against
the evidence base, retrieve facts, or execute an action.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from datetime import date
from functools import lru_cache
from pathlib import Path


INTENTS = ("answer", "compare", "build_dossier", "draft_email", "create_task", "schedule_event", "clarify")
MONTHS = {name: index for index, name in enumerate((
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre"), 1)}
TOPICS = ("budget", "livraison", "planning", "capteurs", "approvisionnement")
ACTION_VERBS = ("relancer", "envoyer", "confirmer", "préparer", "transmettre", "vérifier", "valider", "réviser")
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "Data" / "Curated" / "intent_classifier.joblib"
PROJECT_STOPWORDS = {"la", "le", "les", "un", "une", "du", "des", "de", "pour", "avant", "avec", "sur"}
PERSON_STOPWORDS = {"Prépare", "Rédige", "Écris", "Envoie", "Crée", "Ajoute", "Planifie", "Programme", "Organise", "Compare", "Quelle", "Quel", "Peux", "Pourrais", "Je", "Le", "La", "Les", "Un", "Une"}


def folded(value):
    """Accent-insensitive matching without changing extracted text."""
    return "".join(char for char in unicodedata.normalize("NFD", value.casefold())
                   if unicodedata.category(char) != "Mn")


def extract_subject(request):
    topics = "|".join(TOPICS)
    match = re.search(rf"\b(?i:({topics}))\b\s+(?:(?i:de|du|des)\s+|(?i:d['’]))?([A-ZÉÈÀ][\w-]*)", request)
    if match and folded(match.group(2)) not in PROJECT_STOPWORDS:
        return f"{folded(match.group(1))} {match.group(2)}"
    return None


def extract_person(request, subject):
    """Extract a named person; this is never a recipient resolution."""
    project = subject.split(" ", 1)[1] if subject else None
    for match in re.finditer(r"\b[A-ZÉÈÀ][a-zéèàêëïîôöûüç-]{2,}\b", request):
        name = match.group()
        if name in PERSON_STOPWORDS or name == project or match.start() == 0:
            continue
        return name
    return None


def extract_named_recipient(request):
    """Find an explicitly named mail addressee; other names do not qualify."""
    name = r"([A-ZÉÈÀ][a-zéèàêëïîôöûüç-]{2,})"
    mail = r"(?i:mail|e-mail|email|courriel)"
    patterns = (
        rf"\b{mail}\b(?:\s+(?:destiné|destinée))?\s+(?:à|pour)\s+{name}\b",
        rf"\b(?i:rédige|écris|prépare|compose)\s+(?:à|pour)\s+{name}\s+(?:un|une)\s+{mail}\b",
    )
    for pattern in patterns:
        match = re.search(pattern, request)
        if match:
            return match.group(1)
    return None


def extract_due_date(request):
    iso = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", request)
    if iso:
        try:
            return date(*map(int, iso.groups())).isoformat()
        except ValueError:
            return None
    months = "|".join(MONTHS)
    match = re.search(rf"\b(\d{{1,2}})\s+({months})\s+(\d{{4}})\b", request, re.I)
    if match:
        try:
            return date(int(match.group(3)), MONTHS[match.group(2).casefold()], int(match.group(1))).isoformat()
        except ValueError:
            return None
    return None


def extract_action(request):
    verbs = "|".join(ACTION_VERBS)
    match = re.search(rf"\b({verbs})\b([^,.;?!]*)", request, re.I)
    if not match:
        return None
    phrase = match.group().strip()
    phrase = re.split(r"\s+(?:avant(?:\s+le)?|d'ici(?:\s+le)?|pour le)\s+\d{1,4}\b", phrase, maxsplit=1, flags=re.I)[0]
    return folded(phrase.strip()) or None


def classify(request):
    plain = folded(request)
    if re.search(r"\b(on pourrait|il faudrait peut-etre|on devrait peut-etre)\b", plain):
        return "clarify", "hypothetical_not_request"
    if re.search(r"\b(mail|e-mail|email|courriel)\b", plain) and re.search(r"\b(prepar\w*|redig\w*|ecri\w*|envoi\w*|faire|fais)\b", plain):
        return "draft_email", None
    if re.search(r"\b(rendez-vous|reunion|evenement|creneau)\b", plain) and re.search(r"\b(planifi\w*|programm\w*|organis\w*|fix\w*|cree\w*|ajout\w*)\b", plain):
        return "schedule_event", None
    if re.search(r"\b(tache|todo|rappel)\b", plain) and re.search(r"\b(cree\w*|ajout\w*|not\w*|planifi\w*|fais|faire)\b", plain):
        return "create_task", None
    if re.search(r"\b(dossier|synthese|recapitulatif|recap|brief)\b", plain) and re.search(r"\b(construi\w*|prepar\w*|redig\w*|fais|faire|produi\w*|gener\w*)\b", plain):
        return "build_dossier", None
    if re.search(r"\b(compar\w*|difference\w*|evolution)\b", plain):
        return "compare", None
    if re.search(r"\b(qui|que|quoi|quel\w*|quand|comment|ou|pourquoi|est-ce|dis-moi|explique|retrouve)\b", plain):
        return "answer", None
    return "clarify", "unrecognized_request"


@lru_cache(maxsize=4)
def load_classifier(path=DEFAULT_MODEL):
    """Load the locally trained model once per process."""
    import joblib

    if not Path(path).is_file():
        raise FileNotFoundError(f"Modèle EF-06 absent : {path}. Exécuter train_intents.py.")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"Artefact EF-06 invalide : {path}")
    if Path(path).resolve() == DEFAULT_MODEL.resolve():
        for split, name in (("train", "intent_train.jsonl"), ("validation", "intent_validation.jsonl")):
            expected = bundle.get("metadata", {}).get(split, {}).get("sha256")
            actual = hashlib.sha256((DEFAULT_MODEL.parent / name).read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError(f"Modèle EF-06 périmé pour {name} : relancer train_intents.py")
    return bundle["estimator"]


def route_with_safety(request, predicted_intent, subject, action, due_date):
    """A predicted operation remains a proposal until required slots are present."""
    if predicted_intent not in INTENTS:
        raise ValueError(f"Intention inconnue : {predicted_intent}")
    plain = folded(request)
    if re.search(r"\b(on pourrait|il faudrait peut-etre|on devrait peut-etre)\b", plain):
        return "clarify", "hypothetical_not_request"
    candidate = classify(request)[0] if predicted_intent == "clarify" else predicted_intent
    if candidate == "draft_email" and extract_named_recipient(request) is None:
        return "clarify", "missing_named_recipient"
    if candidate in {"answer", "compare", "build_dossier", "draft_email"} and subject is None:
        return "clarify", "missing_subject"
    if candidate == "create_task" and action is None:
        return "clarify", "missing_action"
    if candidate == "schedule_event" and due_date is None:
        return "clarify", "missing_due_date"
    if predicted_intent == "clarify":
        return "clarify", "intent_uncertain_or_ambiguous"
    return predicted_intent, None


def parse_intent(request, *, classifier=None, use_rules=False):
    if not isinstance(request, str) or not request.strip():
        raise ValueError("La demande doit être une chaîne non vide")
    subject = extract_subject(request)
    person = extract_person(request, subject)
    due_date = extract_due_date(request)
    action = extract_action(request)
    predicted_intent = classify(request)[0] if use_rules else str((classifier or load_classifier()).predict([request])[0])
    intent, reason = route_with_safety(request, predicted_intent, subject, action, due_date)
    return {"intent": intent, "subject": subject, "person": person,
            "due_date": due_date, "action": action, "clarification_reason": reason}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Demande en français à analyser")
    parser.add_argument("--rules", action="store_true", help="Utiliser l'ancienne baseline à règles")
    args = parser.parse_args()
    print(json.dumps(parse_intent(args.request, use_rules=args.rules), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
