"""Rank QMSum passages for a question with BM25 (k1=1.5, b=0.75).

Example:
    python "Data Analysis/bm25_score_qmsum.py" "What was said about the XML format?" --meeting-id Bdb001

Without a question argument, the script asks for one interactively.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


DATA_PATH = Path(__file__).resolve().parents[1] / "Data" / "Processed" / "qmsum_10_tours.jsonl"
TOKEN_PATTERN = re.compile(r"\b\w\w+\b", flags=re.UNICODE)
K1 = 1.5
B = 0.75


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


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


def bm25_scores(texts, question):
    term_frequencies = [Counter(tokenize(text)) for text in texts]
    lengths = [sum(frequencies.values()) for frequencies in term_frequencies]
    average_length = sum(lengths) / len(lengths)
    document_frequency = Counter(
        term for frequencies in term_frequencies for term in frequencies
    )
    query_terms = set(tokenize(question))
    scores = [0.0] * len(texts)
    for term in query_terms:
        frequency = document_frequency[term]
        if not frequency:
            continue
        idf = math.log(1 + (len(texts) - frequency + 0.5) / (frequency + 0.5))
        for index, counts in enumerate(term_frequencies):
            tf = counts[term]
            if tf:
                denominator = tf + K1 * (1 - B + B * lengths[index] / average_length)
                scores[index] += idf * tf * (K1 + 1) / denominator
    return scores


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

    scores = bm25_scores([p["text"] for p in passages], question)
    ranked = sorted(range(len(passages)), key=lambda index: scores[index], reverse=True)
    print(f"BM25: {len(passages)} passages searched for: {question}\n")
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
