"""Conservative, evidence-grounded candidate labels for new semantic-unit pairs.

Predictions never become reviewed relations automatically. Ambiguous pairs abstain.
"""

import re
import unicodedata


STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "cette", "dans", "de", "des", "du", "elle", "en",
    "est", "et", "il", "je", "la", "le", "les", "leur", "lui", "mais", "ne", "nous", "on",
    "ou", "par", "pas", "pour", "que", "qui", "se", "son", "sur", "un", "une", "vous",
    "avoir", "etre", "fait", "faire", "bien", "encore", "avant", "apres", "date", "jour",
}


def normalize(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def words(text, project_id=None):
    excluded = set(re.findall(r"[a-z0-9]+", normalize(project_id or "")))
    return {w for w in re.findall(r"[a-z0-9]+", normalize(text))
            if len(w) > 2 and w not in STOPWORDS and w not in excluded and not w.isdigit()}


def overlap(a, b, project_id=None):
    left, right = words(a, project_id), words(b, project_id)
    return len(left & right) / max(1, min(len(left), len(right)))


def compatible_numbers(a, b):
    def extract(text):
        return {re.sub(r"\s+", "", match) for match in re.findall(r"\b\d[\d\s]*\b", text)}
    left, right = extract(a), extract(b)
    return not left or not right or left == right


def is_negated(text):
    text = normalize(text)
    return bool(re.search(r"\b(?:n['’ ]|ne\s+).{0,35}\bpas\b|\b(?:aucun|jamais|non|sans)\b", text))


def _date_order(newer, older):
    first, second = newer.get("source_date"), older.get("source_date")
    if not first or not second or first < second:
        return False
    if first == second:
        return newer["source_id"] == older["source_id"] and newer["start_turn"] > older["end_turn"]
    return True


def predict_pair(from_unit, to_unit):
    """Return a proposal with exact unit evidence, or abstain with no label."""
    from_id, to_id = from_unit["unit_id"], to_unit["unit_id"]
    result = {"from_id": from_id, "to_id": to_id, "relation": None,
              "reviewed": False, "method": "conservative_rules_v1", "evidence_source_id": None,
              "evidence": None, "reason": "Preuve de relation insuffisante : revue nécessaire."}
    if from_id == to_id or from_unit.get("project_id") != to_unit.get("project_id"):
        result["reason"] = "Paire identique ou projets distincts : revue nécessaire."
        return result
    project = from_unit.get("project_id")
    a = from_unit["evidence"]
    b = to_unit["evidence"]
    sim = overlap(a, b, project)
    a_norm = normalize(a)
    b_norm = normalize(b)
    relation = None
    proof_unit = from_unit
    reason = None

    if (from_unit["status"] == "confirmed" and _date_order(from_unit, to_unit)
            and re.search(r"\b(?:remplac\w*|desormais|plutot\s+qu|au\s+lieu\s+de|et\s+non)\b", a_norm)
            and sim >= 0.15):
        relation, reason = "supersedes", "Remplacement explicite, statut confirmé et ordre daté."
    elif (from_unit["type"] == "answer" and to_unit["type"] == "question"
          and compatible_numbers(a, b) and sim >= 0.2):
        relation, reason = "answers", "Réponse et question portent sur le même sujet."
    elif (from_unit["type"] in {"decision", "commitment"} and to_unit["type"] == "action"
          and re.search(r"\b(?:puisque|suite\s+a|apres|en\s+application\s+de|parce\s+que)\b", b_norm)
          and overlap(a, b.split(",", 1)[0], project) >= 0.2):
        relation, proof_unit, reason = "creates_action", to_unit, "L'action cite explicitement sa cause."
    elif (sim >= 0.4 and compatible_numbers(a, b) and is_negated(a) != is_negated(b)
          and from_unit["type"] not in {"question", "idea", "action"}
          and to_unit["type"] not in {"question", "idea", "action"}):
        relation, reason = "contradicts", "Même sujet avec polarité opposée."
    elif (sim >= 0.3 and compatible_numbers(a, b) and is_negated(a) == is_negated(b)
          and from_unit["status"] in {"asserted", "confirmed"}
          and to_unit["status"] in {"asserted", "confirmed"}
          and (re.search(r"\b(?:confirme\w*|valide\w*|approuv\w*|bien)\b", a_norm))):
        relation, reason = "supports", "Énoncés concordants sur le même sujet."
    elif (sim == 0 and not (from_unit["type"] == "answer" and to_unit["type"] == "question")
          and not (from_unit["type"] in {"decision", "commitment"} and to_unit["type"] == "action")):
        relation, reason = "unrelated", "Aucun terme de sujet ni nombre commun après retrait du projet."

    if relation:
        result.update({"relation": relation, "reason": reason})
        if relation != "unrelated":
            result.update({"evidence_source_id": proof_unit["source_id"],
                           "evidence": proof_unit["evidence"]})
    return result
