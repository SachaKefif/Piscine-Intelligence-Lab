"""Train and select a small EF-06 text classifier using train/validation only."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.svm import LinearSVC

from ingest_semantic_units import ROOT
from parse_intent import INTENTS


DATA = ROOT / "Data" / "Curated"
DEFAULT_TRAIN = DATA / "intent_train.jsonl"
DEFAULT_VALIDATION = DATA / "intent_validation.jsonl"
DEFAULT_MODEL = DATA / "intent_classifier.joblib"
DEFAULT_REPORT = DATA / "intent_training_metrics.json"


def read_rows(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows or {row["intent"] for row in rows} != set(INTENTS):
        raise ValueError(f"Classes absentes dans {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Identifiants dupliqués dans {path}")
    return rows, hashlib.sha256(raw).hexdigest()


def candidate_models():
    word = lambda: TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", sublinear_tf=True, min_df=2)
    char = lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), strip_accents="unicode", sublinear_tf=True, min_df=2)
    return {
        "word_tfidf_linear_svc": make_pipeline(word(), LinearSVC(class_weight="balanced")),
        "char_tfidf_linear_svc": make_pipeline(char(), LinearSVC(class_weight="balanced")),
        "word_char_tfidf_linear_svc": make_pipeline(
            FeatureUnion((("word", word()), ("char", char()))), LinearSVC(class_weight="balanced")
        ),
    }


def train(train_path=DEFAULT_TRAIN, validation_path=DEFAULT_VALIDATION):
    train_rows, train_hash = read_rows(train_path)
    validation_rows, validation_hash = read_rows(validation_path)
    train_projects = {row["subject"].split()[-1] for row in train_rows if row["subject"]}
    validation_projects = {row["subject"].split()[-1] for row in validation_rows if row["subject"]}
    if train_projects & validation_projects or {row["id"] for row in train_rows} & {row["id"] for row in validation_rows}:
        raise ValueError("Fuite entre entraînement et validation")

    X_train = [row["request"] for row in train_rows]
    y_train = [row["intent"] for row in train_rows]
    X_validation = [row["request"] for row in validation_rows]
    y_validation = [row["intent"] for row in validation_rows]
    candidates = candidate_models()
    scores = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        predicted = estimator.predict(X_validation)
        scores[name] = {
            "macro_f1": float(f1_score(y_validation, predicted, labels=INTENTS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_validation, predicted)),
        }
    selected = max(scores, key=lambda name: (scores[name]["macro_f1"], scores[name]["accuracy"]))
    metadata = {
        "selected": selected, "selection_metric": "validation_macro_f1",
        "validation_scores": scores,
        "train": {"count": len(train_rows), "sha256": train_hash, "projects": sorted(train_projects)},
        "validation": {"count": len(validation_rows), "sha256": validation_hash,
                       "projects": sorted(validation_projects)},
        "sklearn_version": sklearn.__version__,
        "note": "Training uses synthetic annotations; no evaluation-set rows are used for fitting or selection.",
    }
    return candidates[selected], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    try:
        estimator, metadata = train(args.train.resolve(), args.validation.resolve())
        args.model.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"estimator": estimator, "metadata": metadata}, args.model)
        args.report.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"model": str(args.model), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
