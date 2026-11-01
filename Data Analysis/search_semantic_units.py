"""Search semantic units across conversations with BM25, E5 or hybrid RRF.

No meeting filter is applied. The same complete unit pool is used by every method.
The --questions mode compares methods on manually labelled unit IDs.
"""

import argparse
import json
from pathlib import Path

from bm25_score_qmsum import bm25_scores
from ingest_semantic_units import DEFAULT_OUTPUT, ROOT, read_jsonl


DEFAULT_QUESTIONS = ROOT / "Data" / "Curated" / "atlas_unit_queries.jsonl"
DEFAULT_MODEL = ROOT / "Models" / "qmsum-e5-small" / "checkpoints" / "checkpoint-842"
RRF_K = 60
E5_WEIGHT = 0.75
METHODS = ("bm25", "e5", "hybrid")


def load_units(path):
    units = list(read_jsonl(path))
    if not units:
        raise ValueError(f"Aucune unité dans {path}. Lancez ingest_semantic_units.py d'abord.")
    ids = [row["unit_id"] for row in units]
    if len(ids) != len(set(ids)):
        raise ValueError("Identifiants d'unité dupliqués")
    if len({row["source_id"] for row in units}) < 2:
        raise ValueError("La recherche EF-02 requiert au moins deux conversations indexées")
    return units


def display_path(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def rank(scores):
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def rrf_scores(bm25_order, e5_order, count):
    bm25_rank = {index: position for position, index in enumerate(bm25_order, 1)}
    e5_rank = {index: position for position, index in enumerate(e5_order, 1)}
    return [
        (1 - E5_WEIGHT) / (RRF_K + bm25_rank[index])
        + E5_WEIGHT / (RRF_K + e5_rank[index])
        for index in range(count)
    ]


class UnitSearch:
    def __init__(self, units, model_path, device, batch_size, dense):
        self.units = units
        self.texts = [row["statement"] for row in units]
        self.model = None
        self.embeddings = None
        if dense:
            if not model_path.is_dir():
                raise ValueError(
                    f"Checkpoint E5 introuvable : {model_path}. "
                    "Utilisez --model avec un modèle disponible, ou --method bm25."
                )
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(str(model_path), device=device)
            self.embeddings = self.model.encode(
                ["passage: " + text for text in self.texts],
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

    def scores(self, query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Question vide")
        lexical = bm25_scores(self.texts, query)
        result = {"bm25": lexical}
        if self.model is not None:
            query_vector = self.model.encode(
                "query: " + query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            dense = self.embeddings @ query_vector
            result["e5"] = dense
            result["hybrid"] = rrf_scores(rank(lexical), rank(dense), len(self.units))
        return result


def result_rows(units, scores, top_k):
    rows = []
    for index in rank(scores)[:top_k]:
        unit = units[index]
        rows.append({
            "unit_id": unit["unit_id"],
            "score": round(float(scores[index]), 7),
            "statement": unit["statement"],
            "type": unit["type"],
            "status": unit["status"],
            "source_id": unit["source_id"],
            "source_date": unit["source_date"],
            "start_turn": unit["start_turn"],
            "end_turn": unit["end_turn"],
            "evidence": unit["evidence"],
            "speaker": unit["speaker"],
            "owner": unit["owner"],
            "due_date": unit["due_date"],
        })
    return rows


def evaluate(search, questions, top_k):
    known = {unit["unit_id"] for unit in search.units}
    if not questions:
        raise ValueError("Aucune question annotée")
    seen = set()
    per_method = {method: [] for method in METHODS}
    for question in questions:
        query_id = question.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in seen:
            raise ValueError(f"query_id absent ou dupliqué : {query_id!r}")
        seen.add(query_id)
        gold_list = question.get("relevant_unit_ids")
        if not isinstance(gold_list, list) or not gold_list or len(gold_list) != len(set(gold_list)):
            raise ValueError(f"{query_id} : relevant_unit_ids absents ou dupliqués")
        gold = set(gold_list)
        if not gold <= known:
            raise ValueError(f"{query_id} : unités pertinentes inconnues : {sorted(gold - known)}")
        scores = search.scores(question.get("query"))
        for method in METHODS:
            indices = rank(scores[method])[:top_k]
            hits = [search.units[index]["unit_id"] in gold for index in indices]
            found = sum(hits)
            first = next((position for position, hit in enumerate(hits, 1) if hit), None)
            per_method[method].append({
                "query_id": query_id,
                "hit_at_k": int(found > 0),
                "recall_at_k": found / len(gold),
                "mrr_at_k": 1 / first if first else 0.0,
                "top_unit_ids": [search.units[index]["unit_id"] for index in indices],
            })
    summary = {}
    for method, rows in per_method.items():
        summary[method] = {
            "hit_at_k": round(sum(row["hit_at_k"] for row in rows) / len(rows), 4),
            "recall_at_k": round(sum(row["recall_at_k"] for row in rows) / len(rows), 4),
            "mrr_at_k": round(sum(row["mrr_at_k"] for row in rows) / len(rows), 4),
        }
    return {"candidate_units": len(search.units), "questions": len(questions), "top_k": top_k,
            "metrics": summary, "details": per_method}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query", help="Question à poser à toutes les conversations indexées")
    mode.add_argument("--questions", type=Path, help="Questions annotées JSONL pour comparer les méthodes")
    parser.add_argument("--units", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--device", default=None, help="cpu, cuda ou périphérique SentenceTransformer")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Écrire le JSON de comparaison dans ce fichier")
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1:
        parser.error("--top-k et --batch-size doivent être positifs")
    if args.output and not args.questions:
        parser.error("--output s'utilise avec --questions")
    try:
        units = load_units(args.units)
        search = UnitSearch(units, args.model, args.device, args.batch_size,
                            dense=bool(args.questions) or args.method != "bm25")
        if args.query is not None:
            scores = search.scores(args.query)[args.method]
            response = {
                "query": args.query,
                "method": args.method,
                "score_type": {"bm25": "bm25", "e5": "cosine", "hybrid": "weighted_rrf"}[args.method],
                "candidate_units": len(units),
                "results": result_rows(units, scores, args.top_k),
            }
        else:
            response = evaluate(search, list(read_jsonl(args.questions)), args.top_k)
            response["unit_file"] = display_path(args.units)
            response["question_file"] = display_path(args.questions)
            response["note"] = "Diagnostic synthétique ; ne remplace pas un test indépendant sur unités réelles."
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    display = response
    if args.questions and args.output:
        display = {key: value for key, value in response.items() if key != "details"}
        display["details_file"] = display_path(args.output)
    print(json.dumps(display, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
