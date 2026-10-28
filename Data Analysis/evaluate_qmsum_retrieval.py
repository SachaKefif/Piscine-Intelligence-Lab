"""Evaluate TF-IDF and BM25 on QMSum validation questions, within each meeting.

Run: python "Data Analysis/evaluate_qmsum_retrieval.py"

Each passage that overlaps a QMSum relevant_text_span is treated as relevant.
These spans can cover a broad discussion, so the labels are approximate.
"""

import json
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from bm25_score_qmsum import bm25_scores


ROOT = Path(__file__).resolve().parents[1]
MEETINGS_PATH = ROOT / "Data" / "Processed" / "qmsum.jsonl"
PASSAGES_PATH = ROOT / "Data" / "Processed" / "qmsum_10_tours.jsonl"
RESULTS_PATH = ROOT / "Data" / "Processed" / "qmsum_retrieval_validation_results.jsonl"
TOP_K = 5


def load_validation_data():
    meetings = {}
    with MEETINGS_PATH.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                meeting = json.loads(line)
                if meeting["split"] == "validation":
                    meeting_id = meeting["meeting_id"]
                    if meeting_id in meetings:
                        raise ValueError(f"Duplicate meeting ID: {meeting_id}")
                    meetings[meeting_id] = meeting

    passages_by_meeting = defaultdict(list)
    with PASSAGES_PATH.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                passage = json.loads(line)
                if passage["split"] == "validation" and passage["text"].strip():
                    passages_by_meeting[passage["meeting_id"]].append(passage)

    for meeting_id in meetings:
        if not passages_by_meeting[meeting_id]:
            raise ValueError(f"No validation passages for meeting {meeting_id}")
    return meetings, passages_by_meeting


def gold_passage_ids(passages, spans):
    intervals = [(int(start), int(end)) for start, end in spans]
    return {
        passage["passage_id"]
        for passage in passages
        if any(
            passage["start_turn"] <= end and passage["end_turn"] >= start
            for start, end in intervals
        )
    }


def evaluate_ranking(passages, scores, gold_ids):
    ranked_indices = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    top_indices = ranked_indices[:TOP_K]
    results = [
        {
            "passage_id": passages[i]["passage_id"],
            "score": round(float(scores[i]), 6),
            "relevant": passages[i]["passage_id"] in gold_ids,
        }
        for i in top_indices
    ]
    relevant_found = sum(result["relevant"] for result in results)
    first_relevant_rank = next(
        (rank for rank, result in enumerate(results, 1) if result["relevant"]), None
    )
    return {
        "top_5": results,
        "hit_at_5": bool(relevant_found),
        "recall_at_5": relevant_found / len(gold_ids),
        "mrr_at_5": 1 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def main():
    meetings, passages_by_meeting = load_validation_data()
    rows = []

    for meeting_id, meeting in meetings.items():
        passages = passages_by_meeting[meeting_id]
        texts = [passage["text"] for passage in passages]
        vectorizer = TfidfVectorizer()
        passage_vectors = vectorizer.fit_transform(texts)

        for question_number, question in enumerate(meeting["queries"]["specific"]):
            query = question["query"]
            gold_ids = gold_passage_ids(passages, question["relevant_text_span"])
            if not gold_ids:
                raise ValueError(f"No relevant passage for {meeting_id} question {question_number}")

            question_vector = vectorizer.transform([query])
            tfidf_scores = (passage_vectors @ question_vector.T).toarray().ravel()
            bm25 = bm25_scores(texts, query)

            rows.append(
                {
                    "question_id": f"{meeting_id}:{question_number}",
                    "meeting_id": meeting_id,
                    "question": query,
                    "relevant_text_span": question["relevant_text_span"],
                    "relevant_passage_ids": sorted(gold_ids),
                    "tf_idf": evaluate_ranking(passages, tfidf_scores, gold_ids),
                    "bm25": evaluate_ranking(passages, bm25, gold_ids),
                }
            )

    if not rows:
        raise ValueError("No validation questions found")

    temporary = RESULTS_PATH.with_suffix(RESULTS_PATH.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(RESULTS_PATH)

    print(f"Evaluated {len(rows)} questions from {len(meetings)} validation meetings.")
    print("Candidate passages: only the question's meeting; relevance: any turn overlap.")
    print("Method    Hit@5   Recall@5   MRR@5")
    for method, name in (("tf_idf", "TF-IDF"), ("bm25", "BM25")):
        hit = sum(row[method]["hit_at_5"] for row in rows) / len(rows)
        recall = sum(row[method]["recall_at_5"] for row in rows) / len(rows)
        mrr = sum(row[method]["mrr_at_5"] for row in rows) / len(rows)
        print(f"{name:<8}  {hit:.3f}     {recall:.3f}      {mrr:.3f}")
    print(f"Results, including failed questions: {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
