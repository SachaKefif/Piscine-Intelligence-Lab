"""Train and apply a meeting-disjoint English QMSum answer-pair classifier.

This model only proposes the `answers` relation. Other relations still require
the existing evidence rules and human review before entering the memory.
"""

import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Data/Datasets/QMSum/data"
MODEL = ROOT / "Data/Curated/answer_relation_model.joblib"
REPORT = ROOT / "docs/answer_relation_evaluation.json"


def pairs(directory, split):
    rows = []
    meetings = []
    for domain in ("Academic", "Product", "Committee"):
        for path in sorted((directory / domain / split).glob("*.json")):
            questions = json.loads(path.read_text(encoding="utf-8"))["specific_query_list"]
            questions = [(row["query"].strip(), row["answer"].strip()) for row in questions
                         if row.get("query") and row.get("answer")]
            if len(questions) < 2:
                continue
            meetings.append(path.stem)
            for index, (question, answer) in enumerate(questions):
                rows.append((question, answer, 1, path.stem))
                # Same-meeting negatives make topic overlap a hard case. They
                # are proxy labels, as the other answer may partly apply.
                for offset in (1, 2):
                    if offset < len(questions):
                        rows.append((question, questions[(index + offset) % len(questions)][1],
                                     0, path.stem))
    return rows, meetings


def features(rows, word_vectorizer, char_vectorizer):
    queries = [row[0] for row in rows]
    answers = [row[1] for row in rows]
    output = []
    for vectorizer in (word_vectorizer, char_vectorizer):
        q = vectorizer.transform(queries)
        a = vectorizer.transform(answers)
        output.append(np.asarray(q.multiply(a).sum(axis=1)).ravel())
    def tokens(value):
        return set(re.findall(r"\w+", value.casefold()))
    overlaps = []
    number_match = []
    for query, answer in zip(queries, answers):
        left, right = tokens(query), tokens(answer)
        overlaps.append(len(left & right) / max(1, len(left)))
        q_numbers = set(re.findall(r"\d+", query))
        a_numbers = set(re.findall(r"\d+", answer))
        number_match.append(float(not q_numbers or bool(q_numbers & a_numbers)))
    output.extend((np.asarray(overlaps), np.asarray(number_match)))
    return np.column_stack(output)


def score(rows, bundle):
    return bundle["estimator"].predict_proba(features(rows, bundle["word"], bundle["char"]))[:, 1]


def measures(rows, probabilities, threshold):
    gold = np.asarray([row[2] for row in rows])
    predicted = probabilities >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(gold, predicted, average="binary", zero_division=0)
    return {"pairs": len(rows), "positive_pairs": int(gold.sum()), "precision_proxy": round(float(precision), 4),
            "recall": round(float(recall), 4), "f1_proxy": round(float(f1), 4),
            "false_positive_rate_proxy": round(float(np.mean(predicted[gold == 0])), 4)}


def train(dataset=DATASET, model_path=MODEL, report_path=REPORT):
    train_rows, train_meetings = pairs(dataset, "train")
    val_rows, val_meetings = pairs(dataset, "val")
    if not val_rows:
        val_rows, val_meetings = pairs(dataset, "validation")
    test_rows, test_meetings = pairs(dataset, "test")
    if not train_rows or not val_rows or not test_rows:
        raise ValueError("Splits QMSum train/val/test manquants")
    if set(train_meetings) & (set(val_meetings) | set(test_meetings)) or set(val_meetings) & set(test_meetings):
        raise ValueError("Fuite de réunions entre splits")
    texts = [text for row in train_rows for text in row[:2]]
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000).fit(texts)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=60000).fit(texts)
    estimator = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    estimator.fit(features(train_rows, word, char), [row[2] for row in train_rows])
    bundle = {"word": word, "char": char, "estimator": estimator, "threshold": None}
    val_probs = score(val_rows, bundle)
    candidates = np.arange(0.3, 0.951, 0.025)
    # Choose solely on validation. Precision breaks F1 ties.
    threshold = max(candidates, key=lambda t: (measures(val_rows, val_probs, t)["f1_proxy"],
                                               measures(val_rows, val_probs, t)["precision_proxy"]))
    bundle["threshold"] = float(threshold)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report = {"task": "QMSum English question / answer association",
              "threshold_selected_on": "validation", "threshold": float(threshold),
              "train_meetings": len(train_meetings), "validation_meetings": len(val_meetings),
              "test_meetings": len(test_meetings),
              "validation": measures(val_rows, val_probs, threshold),
              "test": measures(test_rows, score(test_rows, bundle), threshold),
              "limitations": ["Other same-meeting answers are proxy negatives, not independently adjudicated non-answers.",
                              "The classifier is English QMSum-specific and proposes only answers links.",
                              "Proposals are never promoted to reviewed relations automatically."]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


@lru_cache(maxsize=1)
def load(path=MODEL):
    return joblib.load(path) if Path(path).is_file() else None


def predicts_answer(question, answer, *, model_path=MODEL):
    bundle = load(model_path)
    if bundle is None:
        return None
    probability = float(score([(question, answer, 0, "")], bundle)[0])
    return probability >= bundle["threshold"], probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.model, args.report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
