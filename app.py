"""Local Streamlit demonstration of the evidence-backed semantic agent."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Data Analysis"))

from generate_response import ResponseEngine  # noqa: E402
from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS  # noqa: E402
from relations_temporality import DEFAULT_RELATIONS  # noqa: E402
from search_semantic_units import DEFAULT_MODEL  # noqa: E402
from semantic_agent import SemanticPipeline  # noqa: E402
from semantic_memory import DEFAULT_MEMORY_TOPICS  # noqa: E402


CASES = [json.loads(line) for line in
         (ROOT / "Data/Curated/atlas_acceptance_cases.jsonl").read_text(encoding="utf-8").splitlines()]
SECTIONS = {
    "answer": "Réponse",
    "context": "Contexte",
    "facts": "Faits établis",
    "chronology": "Chronologie",
    "decisions": "Décisions",
    "risks": "Risques",
    "open_points": "Points ouverts",
    "responsibles": "Responsables",
    "changes": "Changements vérifiés",
}
STATUSES = {
    "confirmed": "Confirmé",
    "proposed": "Proposé",
    "asserted": "Déclaré",
    "unknown": "Inconnu",
}
ROLES = {
    "current_decision": "Décision courante",
    "obsolete": "Remplacé",
    "history": "Historique",
}
CLARIFICATIONS = {
    "hypothetical_not_request": "Formulez une demande explicite si vous souhaitez une réponse ou un brouillon.",
    "missing_subject": "Précisez le sujet ou les décisions à examiner.",
    "ambiguous_subject": "Précisez le sujet à examiner.",
    "ambiguous_or_unknown_subject": "Précisez le sujet de l'action.",
    "missing_action": "Précisez l'action à inscrire dans la tâche.",
    "missing_due_date": "Précisez la date de l'événement.",
    "conflicting_evidence": "Les preuves se contredisent ; il faut résoudre ce point avant de préparer l'action.",
    "no_confirmed_context": "Aucun contexte confirmé ne permet de préparer l'action.",
}


def _uploaded_data(files):
    """Return immutable contents so a changed upload always rebuilds the pipeline."""
    return tuple((name, uploaded.getvalue()) for name, uploaded in files.items())


def _signature(mode, method, data):
    digest = hashlib.sha256()
    digest.update(f"{mode}:{method}".encode())
    if mode == "Atlas":
        for path in (DEFAULT_CONVERSATIONS, DEFAULT_UNITS, DEFAULT_RELATIONS,
                     DEFAULT_MEMORY_TOPICS):
            digest.update(path.read_bytes())
    else:
        for name, content in data:
            digest.update(name.encode())
            digest.update(content)
    return digest.hexdigest()


def _pipeline(mode, method, data):
    if mode == "Atlas":
        engine = ResponseEngine(DEFAULT_CONVERSATIONS, DEFAULT_UNITS,
                                DEFAULT_RELATIONS, DEFAULT_MEMORY_TOPICS)
    else:
        with tempfile.TemporaryDirectory(prefix="semantic-agent-ui-") as folder:
            paths = {}
            for name, content in data:
                path = Path(folder) / f"{name}.jsonl"
                path.write_bytes(content)
                paths[name] = path
            engine = ResponseEngine(paths["conversations"], paths["units"],
                                    paths["relations"], paths["topics"])
    return SemanticPipeline(engine=engine, method=method, device="cpu")


def _choose_example():
    selected = st.session_state.get("example")
    if selected is not None:
        st.session_state["request"] = CASES[selected]["request"]


def _render_claims(sections):
    shown = False
    for name, heading in SECTIONS.items():
        claims = sections.get(name) or []
        if not claims:
            continue
        shown = True
        st.subheader(heading)
        for claim in claims:
            st.write(claim["text"])
            details = [STATUSES.get(claim.get("status"), claim.get("status")),
                       ROLES.get(claim.get("role"), claim.get("role"))]
            if claim.get("relation"):
                details.append(f"Relation : {claim['relation']}")
            details.extend(claim.get("source_unit_ids", []))
            st.caption(" · ".join(str(value) for value in details if value))
    if not shown:
        st.info("Aucune affirmation sourcée à afficher.")


def _render_draft(result):
    st.success("Brouillon préparé · aucune action exécutée")
    parameters = result.get("parameters") or {}
    if result["intent"] == "draft_email":
        st.write(f"**Destinataire :** {parameters.get('to_name') or 'à préciser'}")
        st.write(f"**Adresse :** {parameters.get('to_email') or 'non renseignée'}")
        st.write(f"**Objet :** {parameters.get('subject') or 'à préciser'}")
    elif result["intent"] == "create_task":
        st.write(f"**Tâche :** {parameters.get('action') or 'à préciser'}")
        st.write(f"**Échéance :** {parameters.get('due_date') or 'non renseignée'}")
    elif result["intent"] == "schedule_event":
        st.write(f"**Événement :** {parameters.get('title') or 'à préciser'}")
        st.write(f"**Date :** {parameters.get('date') or 'à préciser'}")
    if result.get("needs_confirmation"):
        st.caption("Une confirmation serait nécessaire avant toute action réelle.")
    st.code(result.get("content") or "", language=None)


def _render_references(references):
    if not references:
        st.info("Aucune preuve citée pour cette réponse.")
        return
    for index, row in enumerate(references, 1):
        with st.container(border=True):
            st.write(f"**{index}. {row['statement']}**")
            st.caption(" · ".join(filter(None, [
                row["source_id"], row.get("source_date") or "date inconnue",
                f"tours {row['start_turn']}–{row['end_turn']}",
                STATUSES.get(row.get("status"), row.get("status")),
            ])))
            st.write(f"« {row['evidence']} »")
            st.caption(row["unit_id"])


def _render_retrieval(retrieval):
    st.caption(f"Méthode : {retrieval['method']} · classement de recherche, pas validation des faits")
    rows = retrieval.get("results") or []
    if not rows:
        st.info("Aucun résultat de recherche.")
        return
    for index, row in enumerate(rows, 1):
        with st.expander(f"{index}. {row['statement']}"):
            st.caption(" · ".join(filter(None, [
                row["source_id"], row.get("source_date") or "date inconnue",
                STATUSES.get(row.get("status"), row.get("status")),
                f"score {row['score']:.4f}",
            ])))
            st.write(f"« {row['evidence']} »")
            st.caption(row["unit_id"])


def main():
    st.set_page_config(page_title="Atlas · Mémoire sémantique", page_icon="🔎",
                       layout="wide")

    with st.sidebar:
        st.header("Corpus et recherche")
        mode = st.radio("Corpus", ["Atlas", "Mes fichiers"],
                        captions=["Deux conversations synthétiques annotées",
                                  "Quatre fichiers JSONL déjà annotés"])
        files = {}
        if mode == "Mes fichiers":
            st.caption("Chargez les conversations, unités, relations et sujets conformes aux schémas du projet.")
            for name, label in (
                ("conversations", "Conversations"), ("units", "Unités"),
                ("relations", "Relations"), ("topics", "Sujets"),
            ):
                files[name] = st.file_uploader(label, type="jsonl", key=f"upload_{name}")
        methods = ["bm25"]
        if DEFAULT_MODEL.exists():
            methods.append("hybrid")
        method = st.selectbox("Recherche", methods,
                              index=1 if "hybrid" in methods else 0,
                              format_func=lambda value: "Hybride (BM25 + E5)" if value == "hybrid" else "BM25")
        if "hybrid" not in methods:
            st.caption("Checkpoint E5 absent : seul BM25 est disponible.")
        st.caption("Les brouillons restent des simulations. Aucun message ou événement n'est envoyé.")

    st.title("Atlas · Mémoire sémantique")
    st.write("Interrogez les conversations, comparez des décisions et préparez des brouillons avec leurs preuves.")

    if mode == "Atlas":
        left, right = st.columns([2, 1])
        with left:
            st.selectbox("Essayer un scénario de recette", range(len(CASES)),
                         format_func=lambda index: f"{index + 1}. {CASES[index]['description']}",
                         index=None, placeholder="Choisir un exemple", key="example",
                         on_change=_choose_example)
        with right:
            st.metric("Scénarios Atlas", len(CASES))
    else:
        st.caption("Posez une question sur les sujets annotés dans les fichiers chargés.")

    with st.form("ask"):
        request = st.text_area("Votre demande", key="request", height=95,
                               placeholder="Quelle est la décision actuelle sur le budget Atlas ?")
        submitted = st.form_submit_button("Interroger le mini-agent", type="primary")

    if mode == "Mes fichiers" and not all(files.values()):
        st.info("Chargez les quatre fichiers JSONL pour utiliser votre corpus.")
        if submitted:
            st.stop()

    if submitted:
        if not request.strip():
            st.warning("Saisissez une demande.")
            st.stop()
        data = _uploaded_data(files) if mode == "Mes fichiers" else ()
        signature = _signature(mode, method, data)
        try:
            if st.session_state.get("pipeline_signature") != signature:
                with st.spinner("Chargement du corpus et de l'index…"):
                    st.session_state["pipeline"] = _pipeline(mode, method, data)
                    st.session_state["pipeline_signature"] = signature
            with st.spinner("Analyse de la demande…"):
                st.session_state["result"] = st.session_state["pipeline"].run(request.strip())
            st.session_state["result_signature"] = signature
        except (ValueError, FileNotFoundError, UnicodeError, json.JSONDecodeError) as error:
            st.error(f"Impossible de traiter ce corpus ou cette demande : {error}")
            st.stop()

    result = st.session_state.get("result")
    if not result:
        st.stop()
    current_data = _uploaded_data(files) if mode == "Mes fichiers" and all(files.values()) else ()
    if mode == "Mes fichiers" and not all(files.values()):
        st.stop()
    if st.session_state.get("result_signature") != _signature(mode, method, current_data):
        st.info("Le corpus ou la méthode a changé. Relancez la demande pour actualiser le résultat.")
        st.stop()

    st.divider()
    if result["status"] == "clarify":
        st.warning(result.get("message") or CLARIFICATIONS.get(
            result.get("clarification_reason"), "Précisez votre demande pour continuer."))
    elif result["status"] == "uncertain":
        st.warning(result.get("message") or "Les preuves disponibles ne permettent pas de trancher.")
    elif result["status"] == "not_found":
        st.warning(result.get("message") or "Information introuvable.")
    elif result["status"] == "ok":
        st.success("Réponse construite à partir des sources")

    tab_result, tab_sources, tab_search = st.tabs(["Résultat", "Preuves citées", "Recherche"])
    with tab_result:
        if result["status"] == "draft":
            _render_draft(result)
        else:
            _render_claims(result.get("sections") or {})
            if result.get("missing"):
                st.caption("Informations manquantes : " + ", ".join(result["missing"]))
    with tab_sources:
        _render_references(result.get("references") or [])
    with tab_search:
        _render_retrieval(result["retrieval"])


if __name__ == "__main__":
    main()
