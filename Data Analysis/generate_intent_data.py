"""Rebuild larger synthetic EF-06 JSONL splits from reviewed seeds and templates.

The generated combinations are examples for a future trained classifier. They are
not independent human utterances; template families and projects stay split apart.
"""

import json
import unicodedata
from pathlib import Path

from ingest_semantic_units import ROOT
from parse_intent import INTENTS


DATA = ROOT / "Data" / "Curated"
TOPICS = ("budget", "livraison", "planning", "capteurs", "approvisionnement")
ARTICLES = {"budget": "le", "livraison": "la", "planning": "le",
            "capteurs": "les", "approvisionnement": "l'"}
ACTIONS = ("relancer le fournisseur", "envoyer le dossier", "confirmer les chiffres",
           "vérifier le calendrier", "transmettre la note", "préparer le compte rendu")
SPLITS = {
    "dev": {"projects": ("Mistral", "Delta", "Titan", "Kappa", "Calypso", "Cosmos"),
            "people": ("Camille", "Omar", "Nadia", "Samir", "Lucie", "Maya"),
            "dates": ((12, "décembre", 2026), (13, "décembre", 2026), (14, "décembre", 2026),
                      (15, "décembre", 2026), (16, "décembre", 2026), (17, "décembre", 2026))},
    "validation": {"projects": ("Cobalt", "Meridian", "Galaxie", "Boreal", "Zenith"),
                   "people": ("Aïcha", "Bastien", "Dina", "Farid", "Noémie"),
                   "dates": ((3, "janvier", 2027), (4, "janvier", 2027), (5, "janvier", 2027),
                             (6, "janvier", 2027), (7, "janvier", 2027))},
    "eval_holdout": {"projects": ("Solstice", "Pulsar", "Quasar", "Rivage", "Marina"),
                     "people": ("Iris", "Karim", "Louise", "Milan", "Zoé"),
                     "dates": ((8, "février", 2027), (9, "février", 2027), (10, "février", 2027),
                               (11, "février", 2027), (12, "février", 2027))},
}


PATTERNS = {
    "dev": {
        "answer": (
            "Quelle est la décision actuelle sur {subject_phrase} ?",
            "Que dit {person} sur {subject_phrase} ?",
            "Que sait-on sur {subject_phrase} ?",
            "Quel est le statut {of_subject_phrase} ?",
            "Peux-tu retrouver les faits concernant {subject_phrase} ?",
            "Explique ce qui a été décidé pour {subject_phrase}.",
            "À quelle date a-t-on confirmé {subject_phrase} ?",
            "Quels risques concernent {subject_phrase} ?",
            "Donne-moi le dernier état {of_subject_phrase}.",
            "Est-ce que {subject_phrase} a changé ?",
        ),
        "compare": (
            "Compare {subject_phrase} entre les deux réunions.",
            "Quelles différences pour {subject_phrase} entre hier et aujourd'hui ?",
            "Fais une comparaison {of_subject_phrase} dans les comptes rendus.",
            "Comment {subject_phrase} a-t-il évolué entre les deux échanges ?",
            "Compare ce que {person} dit sur {subject_phrase} dans les deux réunions.",
            "Quelle évolution observes-tu pour {subject_phrase} ?",
            "Explique ce qui change pour {subject_phrase} d'une réunion à l'autre.",
            "Compare les statuts {of_subject_phrase} selon les sources.",
            "Entre les deux réunions, qu'est-ce qui diffère pour {subject_phrase} ?",
            "Présente les écarts sur {subject_phrase}.",
        ),
        "build_dossier": (
            "Prépare un dossier sur {subject_phrase}.",
            "Construis un dossier concernant {subject_phrase}.",
            "Fais une synthèse sur {subject_phrase}.",
            "Rédige un récapitulatif pour {subject_phrase}.",
            "Produis un brief autour {of_subject_phrase}.",
            "Établis un dossier sur {subject_phrase}.",
            "Rassemble un dossier concernant {subject_phrase} pour {person}.",
            "Génère une synthèse sur {subject_phrase}.",
            "Prépare le récap {of_subject_phrase}.",
            "Peux-tu faire un dossier sur {subject_phrase} ?",
        ),
        "draft_email": (
            "Rédige un mail à {person} au sujet {of_subject_phrase}.",
            "Écris un courriel pour {person} concernant {subject_phrase}.",
            "Prépare un e-mail à {person} sur {subject_phrase}.",
            "Peux-tu envoyer un mail à {person} sur {subject_phrase} ?",
            "Rédige à {person} un mail sur {subject_phrase}.",
            "Fais un email pour {person} concernant {subject_phrase}.",
            "Prépare un courriel destiné à {person} sur {subject_phrase}.",
            "Écris à {person} concernant {subject_phrase} par mail.",
            "Rédige un message email à {person} sur {subject_phrase}.",
            "Fais un brouillon de mail à {person} sur {subject_phrase}.",
        ),
        "create_task": (
            "Pour {subject_phrase}, crée une tâche : {action_phrase}.",
            "Ajoute une tâche sur {subject_phrase} : {action_phrase}.",
            "Note un rappel pour {subject_phrase} : {action_phrase}.",
            "Planifie une tâche concernant {subject_phrase} : {action_phrase}.",
            "Mets une tâche sur {subject_phrase} : {action_phrase}.",
            "Crée une tâche liée à {subject_phrase} : {action_phrase}, avant le {date_fr}.",
            "Ajoute au suivi {of_subject_phrase} une tâche pour {person} : {action_phrase}.",
            "Inscris une tâche pour {subject_phrase} : {action_phrase}.",
            "Crée un todo sur {subject_phrase} : {action_phrase}.",
            "Fais un rappel sur {subject_phrase} : {action_phrase}.",
        ),
        "schedule_event": (
            "Planifie une réunion avec {person} sur {subject_phrase} le {date_fr}.",
            "Organise un rendez-vous avec {person} concernant {subject_phrase} le {date_iso}.",
            "Programme un événement sur {subject_phrase} le {date_fr}.",
            "Fixe un créneau avec {person} pour {subject_phrase} le {date_iso}.",
            "Bloque une réunion avec {person} sur {subject_phrase} le {date_fr}.",
            "Ajoute une réunion au calendrier sur {subject_phrase} le {date_iso}.",
            "Crée un rendez-vous sur {subject_phrase} le {date_fr}.",
            "Réserve un créneau avec {person} à propos {of_subject_phrase} le {date_iso}.",
            "Peux-tu planifier une réunion sur {subject_phrase} le {date_fr} ?",
            "Organise un événement avec {person} sur {subject_phrase} le {date_iso}.",
        ),
        "clarify": (
            "Prépare un mail à la responsable pour {subject_phrase}.",
            "Compare les deux décisions.",
            "On pourrait peut-être revoir {subject_phrase}.",
            "Planifie une réunion sur {subject_phrase}.",
            "Crée une tâche sur {subject_phrase}.",
            "Rédige un courriel au responsable pour {subject_phrase}.",
            "Quelle est la décision ?",
            "Il faudrait peut-être {action_phrase} pour {subject_phrase}.",
            "Organise un rendez-vous avec {person} sur {subject_phrase} demain.",
            "Prépare un mail à la responsable pour {subject_phrase}. {person} suit le dossier.",
        ),
    },
    "validation": {
        "answer": ("Peux-tu me dire où en est {subject_phrase} ?", "Retrouve la décision prise pour {subject_phrase}.", "D'après {person}, où en est {subject_phrase} ?"),
        "compare": ("Mets en parallèle les échanges sur {subject_phrase}.", "Qu'est-ce qui diffère sur {subject_phrase} ?", "Compare la première et la seconde version {of_subject_phrase}."),
        "build_dossier": ("Assemble un dossier consacré à {subject_phrase}.", "Prépare une fiche de synthèse sur {subject_phrase}.", "Rassemble les éléments pour un brief sur {subject_phrase}."),
        "draft_email": ("Prépare un message par mail pour {person} à propos {of_subject_phrase}.", "Écris à {person} un courriel concernant {subject_phrase}.", "Rédige un e-mail destiné à {person} sur {subject_phrase}."),
        "create_task": ("Pour {subject_phrase}, ajoute à ma liste une tâche : {action_phrase}.", "Inscris un rappel pour {person} sur {subject_phrase} : {action_phrase}.", "Crée une tâche pour {subject_phrase} : {action_phrase}, avant le {date_fr}."),
        "schedule_event": ("Réserve une réunion avec {person} sur {subject_phrase} le {date_fr}.", "Programme un rendez-vous pour {subject_phrase} le {date_iso}.", "Mets un événement avec {person} sur {subject_phrase} le {date_fr}."),
        "clarify": ("Écris un mail au responsable sur {subject_phrase}.", "Compare ces décisions.", "On devrait peut-être revoir {subject_phrase}."),
    },
    "eval_holdout": {
        "answer": ("Indique-moi la décision relative à {subject_phrase}.", "Qu'a confirmé {person} à propos {of_subject_phrase} ?", "Fais le point sur {subject_phrase}."),
        "compare": ("Distingue les versions successives {of_subject_phrase}.", "Quel changement entre les réunions pour {subject_phrase} ?", "Oppose les comptes rendus sur {subject_phrase}."),
        "build_dossier": ("Constitue un dossier complet sur {subject_phrase}.", "Donne-moi une note de synthèse concernant {subject_phrase}.", "Monte un brief sur {subject_phrase}."),
        "draft_email": ("Compose un mail à {person} au sujet {of_subject_phrase}.", "Écris pour {person} un e-mail sur {subject_phrase}.", "Prépare à {person} un courriel sur {subject_phrase}."),
        "create_task": ("Consigne une tâche sur {subject_phrase} : {action_phrase}.", "Ajoute à ma todo pour {person} sur {subject_phrase} : {action_phrase}.", "Crée une tâche liée à {subject_phrase} : {action_phrase}, pour le {date_fr}."),
        "schedule_event": ("Inscris une réunion avec {person} sur {subject_phrase} le {date_fr}.", "Réserve un rendez-vous pour {subject_phrase} le {date_iso}.", "Bloque un événement avec {person} autour {of_subject_phrase} le {date_fr}."),
        "clarify": ("Prépare un e-mail à la cheffe sur {subject_phrase}.", "Quelle décision faut-il comparer ?", "On pourrait éventuellement {action_phrase} pour {subject_phrase}."),
    },
}


def normalized(value):
    return "".join(character for character in unicodedata.normalize("NFD", value.casefold())
                   if unicodedata.category(character) != "Mn")


def seed_rows(split):
    path = DATA / f"intent_seed_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generated_rows(split):
    config = SPLITS[split]
    for intent in INTENTS:
        for template_index, template in enumerate(PATTERNS[split][intent]):
            for project_index, project in enumerate(config["projects"]):
                if project_index and not any(marker in template for marker in ("{subject_phrase}", "{of_subject_phrase}", "{person}", "{action_phrase}", "{date_fr}", "{date_iso}")):
                    continue
                topic = TOPICS[(template_index + project_index) % len(TOPICS)]
                day, month, year = config["dates"][project_index]
                action_phrase = ACTIONS[(template_index + project_index) % len(ACTIONS)]
                phrase = f"{ARTICLES[topic]} {topic} de {project}" if topic != "approvisionnement" else f"l'approvisionnement de {project}"
                contracted = {"budget": "du", "livraison": "de la", "planning": "du",
                              "capteurs": "des", "approvisionnement": "de l'"}[topic]
                of_phrase = (f"{contracted} {topic} de {project}" if topic != "approvisionnement"
                             else f"de l'approvisionnement de {project}")
                context = {"subject_phrase": phrase, "person": config["people"][project_index],
                           "of_subject_phrase": of_phrase,
                           "action_phrase": action_phrase, "date_fr": f"{day} {month} {year}",
                           "date_iso": f"{year}-{1 if month == 'janvier' else 2 if month == 'février' else 12:02d}-{day:02d}"}
                request = template.format_map(context)
                has_subject = "{subject_phrase}" in template or "{of_subject_phrase}" in template
                has_date = "{date_fr}" in template or "{date_iso}" in template
                yield {
                    "id": f"{split}-{intent}-{template_index + 1:02d}-{project_index + 1:02d}",
                    "request": request, "intent": intent,
                    "subject": f"{topic} {project}" if has_subject else None,
                    "person": context["person"] if "{person}" in template else None,
                    "due_date": context["date_iso"] if has_date else None,
                    "action": normalized(action_phrase) if "{action_phrase}" in template else None,
                    "template_id": f"{split}-{intent}-{template_index + 1:02d}",
                    "origin": "synthetic_template",
                }


def main():
    templates = [normalized(template) for families in PATTERNS.values()
                 for patterns in families.values() for template in patterns]
    if len(templates) != len(set(templates)):
        raise ValueError("Une famille de formulation est présente dans plusieurs classes ou splits")
    all_requests = set()
    project_sets = []
    for split in SPLITS:
        rows = seed_rows(split) + list(generated_rows(split))
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or {row["intent"] for row in rows} != set(INTENTS):
            raise ValueError(f"Identifiants ou classes invalides pour {split}")
        projects = {row["subject"].split()[-1] for row in rows if row["subject"]}
        if any(projects & earlier for earlier in project_sets):
            raise ValueError(f"Fuite de projets entre splits : {split}")
        project_sets.append(projects)
        unique_rows = []
        for row in rows:
            key = normalized(row["request"]).strip()
            if key in all_requests:
                if row.get("origin") == "synthetic_template":
                    continue
                raise ValueError(f"Demande dupliquée dans les graines : {row['request']}")
            all_requests.add(key)
            unique_rows.append(row)
        path = DATA / ("intent_train.jsonl" if split == "dev" else f"intent_{split}.jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in unique_rows), encoding="utf-8")
        print(json.dumps({"split": split, "count": len(unique_rows), "classes": {label: sum(row["intent"] == label for row in unique_rows) for label in INTENTS},
                          "projects": sorted(projects)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
