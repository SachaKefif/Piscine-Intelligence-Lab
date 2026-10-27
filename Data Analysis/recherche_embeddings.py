"""Search QMSum with pretrained E5 embeddings and evaluate on validation.

Evaluate all 237 validation questions:
    python "Data Analysis/recherche_ebeddings.py"

Inspect one question in one validation meeting:
    python "Data Analysis/recherche_ebeddings.py" --meeting-id Bed002 --question "What was discussed?"

The model is not trained here. Passage vectors are cached after the first run.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluate_qmsum_retrieval import gold_passage_ids, load_validation_data


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data" / "Processed"
BASELINE_RESULTS = PROCESSED / "qmsum_retrieval_validation_results.jsonl"
COMPARISON_RESULTS = PROCESSED / "qmsum_retrieval_validation_comparison.jsonl"
CACHE_PATH = PROCESSED / "qmsum_validation_e5_embeddings.npz"
MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 5


def flatten_passages(meetings, passages_by_meeting):
    passages = [
        passage
        for meeting_id in meetings
        for passage in passages_by_meeting[meeting_id]
    ]
    ids = [passage["passage_id"] for passage in passages]
    if len(ids) != len(set(ids)):
        raise ValueError("Passage IDs are not unique")
    return passages


def fingerprint_passages(passages):
    digest = hashlib.sha256()
    digest.update(MODEL_NAME.encode("utf-8"))
    for passage in passages:
        digest.update(json.dumps(
            [passage["passage_id"], passage["text"]], ensure_ascii=False
        ).encode("utf-8"))
    return digest.hexdigest()


def count_truncated_passages(model, passages):
    limit = model.max_seq_length
    if not limit:
        return None, 0
    count = 0
    for passage in passages:
        token_ids = model.tokenizer.encode(
            "passage: " + passage["text"],
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
        if len(token_ids) > limit:
            count += 1
    return limit, count


def passage_embeddings(model, passages, batch_size):
    fingerprint = fingerprint_passages(passages)
    passage_ids = [p["passage_id"] for p in passages]
    if CACHE_PATH.is_file():
        with np.load(CACHE_PATH, allow_pickle=False) as cached:
            if (
                str(cached["model_name"].item()) == MODEL_NAME
                and str(cached["fingerprint"].item()) == fingerprint
                and cached["passage_ids"].tolist() == passage_ids
            ):
                print(f"Loaded {len(passages)} cached passage embeddings.")
                return cached["embeddings"]

    print(f"Encoding {len(passages)} validation passages with {MODEL_NAME}...")
    embeddings = model.encode(
        ["passage: " + p["text"] for p in passages],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    temporary = CACHE_PATH.with_suffix(".npz.tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            embeddings=embeddings,
            passage_ids=np.asarray(passage_ids),
            model_name=MODEL_NAME,
            fingerprint=fingerprint,
        )
    temporary.replace(CACHE_PATH)
    print(f"Saved embeddings to {CACHE_PATH.relative_to(ROOT)}")
    return embeddings


def rank_passages(passages, scores, gold_ids=None):
    ranked = np.argsort(-scores)[:TOP_K]
    top = [
        {
            "passage_id": passages[int(i)]["passage_id"],
            "score": round(float(scores[i]), 6),
            "relevant": passages[int(i)]["passage_id"] in gold_ids
            if gold_ids is not None else None,
        }
        for i in ranked
    ]
    if gold_ids is None:
        return top
    found = sum(result["relevant"] for result in top)
    first_rank = next(
        (rank for rank, result in enumerate(top, 1) if result["relevant"]), None
    )
    return {
        "top_5": top,
        "hit_at_5": bool(found),
        "recall_at_5": found / len(gold_ids),
        "mrr_at_5": 1 / first_rank if first_rank else 0.0,
    }


def load_baseline_results():
    if not BASELINE_RESULTS.is_file():
        raise FileNotFoundError(
            f"Run 'python \"Data Analysis/evaluate_qmsum_retrieval.py\"' first: {BASELINE_RESULTS}"
        )
    with BASELINE_RESULTS.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return {row["question_id"]: row for row in rows}


def write_comparison(rows):
    temporary = COMPARISON_RESULTS.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(COMPARISON_RESULTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Search for one question instead of evaluating all questions")
    parser.add_argument("--meeting-id", help="Validation meeting for --question, e.g. Bed002")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if bool(args.question) != bool(args.meeting_id):
        parser.error("--question and --meeting-id must be supplied together")

    meetings, passages_by_meeting = load_validation_data()
    if args.meeting_id and args.meeting_id not in meetings:
        parser.error(f"Unknown validation meeting: {args.meeting_id}")
    passages = flatten_passages(meetings, passages_by_meeting)
    model = SentenceTransformer(MODEL_NAME)
    limit, truncated = count_truncated_passages(model, passages)
    print(f"Model: {MODEL_NAME}; maximum input: {limit} tokens; passages over limit: {truncated}/{len(passages)}")
    embeddings = passage_embeddings(model, passages, args.batch_size)
    index_by_id = {p["passage_id"]: index for index, p in enumerate(passages)}

    if args.question:
        meeting_passages = passages_by_meeting[args.meeting_id]
        passage_by_id = {p["passage_id"]: p for p in meeting_passages}
        indices = [index_by_id[p["passage_id"]] for p in meeting_passages]
        query_vector = model.encode(
            "query: " + args.question, normalize_embeddings=True, convert_to_numpy=True
        )
        scores = embeddings[indices] @ query_vector
        for rank, result in enumerate(rank_passages(meeting_passages, scores), 1):
            passage = passage_by_id[result["passage_id"]]
            print(f"{rank}. {result['passage_id']} | cosine={result['score']:.4f}")
            print(f"   {passage['text'][:600].replace(chr(10), ' ')}\n")
        return

    baselines = load_baseline_results()
    questions = [
        (meeting_id, number, question)
        for meeting_id, meeting in meetings.items()
        for number, question in enumerate(meeting["queries"]["specific"])
    ]
    query_vectors = model.encode(
        ["query: " + question["query"] for _, _, question in questions],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    rows = []
    for (meeting_id, number, question), query_vector in zip(questions, query_vectors):
        question_id = f"{meeting_id}:{number}"
        baseline = baselines.get(question_id)
        if baseline is None or baseline["question"] != question["query"]:
            raise ValueError(f"Baseline result missing or stale for {question_id}")
        meeting_passages = passages_by_meeting[meeting_id]
        gold_ids = gold_passage_ids(meeting_passages, question["relevant_text_span"])
        if sorted(gold_ids) != baseline["relevant_passage_ids"]:
            raise ValueError(f"Baseline relevance labels differ for {question_id}")
        indices = [index_by_id[p["passage_id"]] for p in meeting_passages]
        scores = embeddings[indices] @ query_vector
        row = dict(baseline)
        row["embeddings_pretrained"] = rank_passages(meeting_passages, scores, gold_ids)
        rows.append(row)

    if len(rows) != len(baselines):
        raise ValueError("Baseline and embedding evaluation have different question counts")
    write_comparison(rows)
    print(f"Evaluated {len(rows)} questions from {len(meetings)} validation meetings.")
    print("Method                Hit@5   Recall@5   MRR@5")
    for key, label in (
        ("tf_idf", "TF-IDF"),
        ("bm25", "BM25"),
        ("embeddings_pretrained", "Pretrained embeddings"),
    ):
        hit = sum(row[key]["hit_at_5"] for row in rows) / len(rows)
        recall = sum(row[key]["recall_at_5"] for row in rows) / len(rows)
        mrr = sum(row[key]["mrr_at_5"] for row in rows) / len(rows)
        print(f"{label:<21} {hit:.3f}     {recall:.3f}      {mrr:.3f}")
    print(f"Details: {COMPARISON_RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
