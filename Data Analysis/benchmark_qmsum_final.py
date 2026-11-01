"""Final QMSum retrieval benchmark: validation selection, then one frozen test.

Run validation first. It selects a BM25 + adapted-E5 reciprocal-rank-fusion
weight using MRR@5, with nDCG@5 and Hit@5 as tie breakers. The test command
requires the frozen validation config and refuses to overwrite test results.
All five methods search only passages from the question's own meeting.
"""

import argparse
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from bm25_score_qmsum import bm25_scores
from finetune_qmsum_e5 import MODEL_NAME, PROCESSED, ROOT, read_jsonl, relevant_passages


OUTPUT = ROOT / "Models" / "qmsum-retrieval-benchmark"
CHECKPOINT = ROOT / "Models" / "qmsum-e5-small" / "checkpoints" / "checkpoint-842"
RRF_K = 60
WEIGHTS = (0.25, 0.50, 0.75)
METHODS = ("tf_idf", "bm25", "e5_initial", "e5_adapte", "hybride")
TOP_K = 5


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints():
    paths = {
        "meetings": PROCESSED / "qmsum.jsonl",
        "passages": PROCESSED / "qmsum_10_tours.jsonl",
        "adapted_model": CHECKPOINT / "model.safetensors",
        "benchmark_code": Path(__file__),
    }
    return {key: sha256_file(path) for key, path in paths.items()}


def load_split(split):
    meetings = {
        row["meeting_id"]: row for row in read_jsonl(PROCESSED / "qmsum.jsonl")
        if row["split"] == split
    }
    passages = defaultdict(list)
    for row in read_jsonl(PROCESSED / "qmsum_10_tours.jsonl"):
        if row["split"] == split and row["text"].strip():
            passages[row["meeting_id"]].append(row)
    questions = []
    for meeting_id, meeting in meetings.items():
        candidates = passages[meeting_id]
        if not candidates:
            raise ValueError(f"No passages for {meeting_id}")
        ids = [row["passage_id"] for row in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate passage ID in {meeting_id}")
        for number, question in enumerate(meeting["queries"]["specific"]):
            gold = {row["passage_id"] for row in relevant_passages(
                candidates, question["relevant_text_span"]
            )}
            if not gold:
                raise ValueError(f"No gold passage for {meeting_id}:{number}")
            questions.append({
                "question_id": f"{meeting_id}:{number}",
                "meeting_id": meeting_id,
                "domain": meeting["domain"],
                "question": question["query"],
                "gold_ids": gold,
            })
    return meetings, passages, questions


def order_scores(scores):
    # Stable ties follow the source passage order, including zero BM25 scores.
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def measures(order, candidates, gold_ids):
    top = order[:TOP_K]
    hits = [candidates[index]["passage_id"] in gold_ids for index in top]
    found = sum(hits)
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(hits, 1) if hit)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), TOP_K) + 1))
    first_hit = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    return {
        "hit_at_5": float(found > 0),
        "recall_at_5": found / len(gold_ids),
        "mrr_at_5": 1.0 / first_hit if first_hit else 0.0,
        "ndcg_at_5": dcg / ideal,
        "top_5": [
            {"passage_id": candidates[index]["passage_id"], "relevant": hits[rank]}
            for rank, index in enumerate(top)
        ],
    }


def mean_metrics(rows):
    keys = ("hit_at_5", "recall_at_5", "mrr_at_5", "ndcg_at_5")
    return {key: round(sum(row[key] for row in rows) / len(rows), 4) for key in keys}


def summarize(results):
    domains = sorted({row["domain"] for row in results})
    summary = {}
    for method in METHODS:
        summary[method] = {
            "global": {"questions": len(results), **mean_metrics([row["methods"][method] for row in results])},
            "domains": {
                domain: {
                    "questions": sum(row["domain"] == domain for row in results),
                    **mean_metrics([row["methods"][method] for row in results if row["domain"] == domain]),
                }
                for domain in domains
            },
        }
    return summary


def rrf_order(bm25_order, e5_order, weight, size):
    bm25_rank = {index: rank for rank, index in enumerate(bm25_order, 1)}
    e5_rank = {index: rank for rank, index in enumerate(e5_order, 1)}
    scores = [
        (1.0 - weight) / (RRF_K + bm25_rank[index])
        + weight / (RRF_K + e5_rank[index])
        for index in range(size)
    ]
    return order_scores(scores)


def add_lexical_scores(passages, questions):
    methods = defaultdict(dict)
    seconds = {"tf_idf": 0.0, "bm25": 0.0}
    grouped = defaultdict(list)
    for row in questions:
        grouped[row["meeting_id"]].append(row)
    for meeting_id, meeting_questions in grouped.items():
        texts = [row["text"] for row in passages[meeting_id]]
        t0 = time.perf_counter()
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(texts)
        for question in meeting_questions:
            query_vector = vectorizer.transform([question["question"]])
            scores = (document_vectors @ query_vector.T).toarray().ravel()
            methods[question["question_id"]]["tf_idf"] = order_scores(scores)
        seconds["tf_idf"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        for question in meeting_questions:
            scores = bm25_scores(texts, question["question"])
            methods[question["question_id"]]["bm25"] = order_scores(scores)
        seconds["bm25"] += time.perf_counter() - t0
    return methods, seconds


def add_dense_scores(passages, questions, methods, batch_size):
    import torch
    from sentence_transformers import SentenceTransformer

    seconds = {}
    memory = {}
    flat_passages = [row for meeting_id in dict.fromkeys(q["meeting_id"] for q in questions)
                     for row in passages[meeting_id]]
    passage_indices = defaultdict(list)
    for index, row in enumerate(flat_passages):
        passage_indices[row["meeting_id"]].append(index)
    question_texts = ["query: " + row["question"] for row in questions]
    passage_texts = ["passage: " + row["text"] for row in flat_passages]
    for method, name in (("e5_initial", MODEL_NAME), ("e5_adapte", str(CHECKPOINT))):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = SentenceTransformer(name)
        model.max_seq_length = 512
        query_vectors = model.encode(question_texts, batch_size=batch_size,
                                     normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=True)
        passage_vectors = model.encode(passage_texts, batch_size=batch_size,
                                       normalize_embeddings=True, convert_to_numpy=True,
                                       show_progress_bar=True)
        for question, query_vector in zip(questions, query_vectors):
            indices = passage_indices[question["meeting_id"]]
            scores = passage_vectors[indices] @ query_vector
            methods[question["question_id"]][method] = order_scores(scores)
        seconds[method] = round(time.perf_counter() - t0, 3)
        memory[method] = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        )
        del model, query_vectors, passage_vectors
    return seconds, memory


def write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_key(metrics):
    return (metrics["mrr_at_5"], metrics["ndcg_at_5"], metrics["hit_at_5"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not (CHECKPOINT / "modules.json").is_file():
        parser.error(f"Adapted E5 checkpoint missing: {CHECKPOINT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT / "frozen_config.json"
    summary_path = OUTPUT / f"{args.split}_summary.json"
    details_path = OUTPUT / f"{args.split}_details.jsonl"
    if summary_path.exists() or (args.split == "test" and details_path.exists()):
        parser.error(f"{args.split} benchmark already exists; results will not be overwritten")
    current_fingerprints = fingerprints()
    if args.split == "test":
        if not config_path.is_file() or not (OUTPUT / "validation_summary.json").is_file():
            parser.error("Run validation and freeze the fusion rule before test")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config["fingerprints"] != current_fingerprints:
            parser.error("Code, data, or adapted model changed since validation freeze")
        if config["batch_size"] != args.batch_size:
            parser.error("Use the same batch size as the frozen validation run")
        weight = config["hybrid_weight_e5"]
    else:
        if config_path.exists():
            parser.error("Fusion rule is already frozen; validation will not be overwritten")

    meetings, passages, questions = load_split(args.split)
    print(f"{args.split}: {len(meetings)} meetings, {len(questions)} questions, "
          f"{sum(len(rows) for rows in passages.values())} passages")
    methods, lexical_seconds = add_lexical_scores(passages, questions)
    dense_seconds, gpu_memory = add_dense_scores(passages, questions, methods, args.batch_size)

    if args.split == "validation":
        weight_candidates = []
        for candidate_weight in WEIGHTS:
            values = []
            for question in questions:
                question_id = question["question_id"]
                order = rrf_order(methods[question_id]["bm25"],
                                  methods[question_id]["e5_adapte"], candidate_weight,
                                  len(passages[question["meeting_id"]]))
                values.append(measures(order, passages[question["meeting_id"]], question["gold_ids"]))
            weight_candidates.append({"weight_e5": candidate_weight, **mean_metrics(values)})
        chosen = max(weight_candidates, key=lambda row: score_key(row))
        weight = chosen["weight_e5"]
        print(f"Validation RRF candidates: {weight_candidates}")
        print(f"Frozen hybrid weight for E5: {weight}")

    results = []
    for question in questions:
        question_id = question["question_id"]
        meeting_id = question["meeting_id"]
        candidates = passages[meeting_id]
        orders = methods[question_id]
        orders["hybride"] = rrf_order(orders["bm25"], orders["e5_adapte"], weight, len(candidates))
        results.append({
            "question_id": question_id,
            "meeting_id": meeting_id,
            "domain": question["domain"],
            "question": question["question"],
            "gold_ids": sorted(question["gold_ids"]),
            "methods": {method: measures(orders[method], candidates, question["gold_ids"])
                        for method in METHODS},
        })
    summary = summarize(results)
    selected = max(("e5_adapte", "hybride"),
                   key=lambda method: score_key(summary[method]["global"])) if args.split == "validation" else config["selected_retriever"]
    report = {
        "split": args.split,
        "meeting_count": len(meetings),
        "question_count": len(questions),
        "passage_count": sum(len(rows) for rows in passages.values()),
        "candidate_scope": "question meeting only",
        "gold_rule": "any inclusive turn overlap with QMSum relevant_text_span",
        "top_k": TOP_K,
        "hybrid": {"rule": "weighted reciprocal rank fusion", "rrf_k": RRF_K,
                   "weight_e5": weight, "weight_bm25": 1.0 - weight},
        "selected_retriever_on_validation": selected,
        "methods": summary,
        "timing_seconds": {**lexical_seconds, **dense_seconds},
        "peak_gpu_allocated_bytes": gpu_memory,
        "fingerprints": current_fingerprints,
    }
    if args.split == "test":
        initial = summary["e5_initial"]["global"]
        adapted = summary["e5_adapte"]["global"]
        bm25 = summary["bm25"]["global"]
        report["predeclared_objectives"] = {
            "adapted_mrr_gain_at_least_0_03": adapted["mrr_at_5"] - initial["mrr_at_5"] >= 0.03,
            "adapted_hit_at_least_bm25": adapted["hit_at_5"] >= bm25["hit_at_5"],
        }
    write_jsonl(details_path, results)
    write_json(summary_path, report)
    if args.split == "validation":
        config = {
            "hybrid_weight_e5": weight,
            "rrf_k": RRF_K,
            "candidate_weights": weight_candidates,
            "selection_metric_order": ["mrr_at_5", "ndcg_at_5", "hit_at_5"],
            "selected_retriever": selected,
            "batch_size": args.batch_size,
            "fingerprints": current_fingerprints,
        }
        write_json(config_path, config)
    print(f"Results: {summary_path}")
    for method in METHODS:
        print(method, summary[method]["global"])


if __name__ == "__main__":
    main()
