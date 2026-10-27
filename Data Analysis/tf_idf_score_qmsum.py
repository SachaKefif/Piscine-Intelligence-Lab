"""Rank QMSum passages for a question with TF-IDF cosine similarity.

Example:
    python "Data Analysis/tf_idf_score_qmsum.py" "What was said about the XML format?" --meeting-id Bdb001

Without a question argument, the script asks for one interactively.
"""

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "Processed" / "qmsum_10_tours.jsonl"


def load_passages(meeting_id, split):
    passages = []
    with DATA_PATH.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            passage = json.loads(line)
            if meeting_id and passage["meeting_id"] != meeting_id:
                continue
            if split and passage["split"] != split:
                continue
            if passage["text"].strip():
                passages.append(passage)
    return passages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="Question to search for (in English)")
    parser.add_argument("--meeting-id", help="Search only within this meeting, e.g. Bdb001")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    question = args.question or input("Question (in English): ").strip()
    if not question:
        parser.error("A non-empty question is required")
    if not DATA_PATH.is_file():
        parser.error(f"Passage file not found: {DATA_PATH}")

    passages = load_passages(args.meeting_id, args.split)
    if not passages:
        parser.error("No passages match the selected meeting and split")

    # TfidfVectorizer normalizes each vector; their dot product is cosine similarity.
    vectorizer = TfidfVectorizer()
    passage_vectors = vectorizer.fit_transform(p["text"] for p in passages)
    question_vector = vectorizer.transform([question])
    if question_vector.nnz == 0:
        print("No question terms occur in the selected passages.")
        return

    scores = (passage_vectors @ question_vector.T).toarray().ravel()
    ranked = sorted(range(len(passages)), key=lambda index: scores[index], reverse=True)
    print(f"TF-IDF: {len(passages)} passages searched for: {question}\n")
    shown = 0
    for index in ranked:
        if scores[index] <= 0 or shown >= args.top_k:
            break
        passage = passages[index]
        shown += 1
        print(f"{shown}. {passage['passage_id']} | split={passage['split']} | score={scores[index]:.4f}")
        print(f"   Source: {', '.join(passage.get('source_files', []))}")
        print(f"   {passage['text'][:600].replace(chr(10), ' ')}\n")
    if not shown:
        print("No passage shares a searchable term with the question.")


if __name__ == "__main__":
    main()
