"""Diagnostic of the answers relation on original QMSum test questions and answers.

Nonmatching pairs are a proxy negative class: another answer from the same
meeting might still partially answer a question. This is not a test of the
other five relation types or automatic extraction from transcripts.
"""

import argparse
import hashlib
import json
from pathlib import Path

from predict_relations import predict_pair


ROOT = Path(__file__).resolve().parents[1]
QMSUM = ROOT / "Data/Datasets/QMSum/data"


def fixture(raw_dir):
    pairs = []
    source_hashes = {}
    for domain in ("Academic", "Product", "Committee"):
        for path in sorted((raw_dir / domain / "test").glob("*.json")):
            if path.stem in {"education_13", "education_17"}:
                continue
            data = path.read_bytes()
            source_hashes[f"{domain}/{path.name}"] = hashlib.sha256(data).hexdigest()
            questions = json.loads(data)["specific_query_list"]
            if len(questions) < 2:
                continue
            for index, question in enumerate(questions):
                own = question["answer"].strip()
                other = questions[(index + 1) % len(questions)]["answer"].strip()
                if not own or not other or own == other:
                    continue
                query = {"unit_id": f"{path.stem}:q{index}", "project_id": path.stem,
                         "type": "question", "status": "unknown", "source_date": None,
                         "source_id": path.stem, "start_turn": index, "end_turn": index,
                         "statement": question["query"], "evidence": question["query"]}
                for label, answer in ((True, own), (False, other)):
                    candidate = {"unit_id": f"{path.stem}:a{index}:{int(label)}",
                                 "project_id": path.stem, "type": "answer", "status": "asserted",
                                 "source_date": None, "source_id": path.stem,
                                 "start_turn": index, "end_turn": index,
                                 "statement": answer, "evidence": answer}
                    pairs.append((domain, path.stem, query, candidate, label))
    return pairs, source_hashes


def evaluate(raw_dir=QMSUM):
    pairs, hashes = fixture(raw_dir)
    if not pairs:
        raise ValueError("Aucune paire QMSum test disponible")
    details = []
    for domain, meeting, query, candidate, gold in pairs:
        prediction = predict_pair(candidate, query)
        details.append({"domain": domain, "meeting": meeting,
                        "question_id": query["unit_id"], "positive_pair": gold,
                        "predicted_relation": prediction["relation"],
                        "predicted_answers": prediction["relation"] == "answers"})
    positives = [row for row in details if row["positive_pair"]]
    negatives = [row for row in details if not row["positive_pair"]]
    tp = sum(row["predicted_answers"] for row in positives)
    fp = sum(row["predicted_answers"] for row in negatives)
    return {"dataset": "QMSum original specific questions and human answers; test split",
            "pairs": len(details), "positive_pairs": len(positives), "proxy_negative_pairs": len(negatives),
            "answer_recall": round(tp / len(positives), 4),
            "proxy_false_positive_rate": round(fp / len(negatives), 4),
            "answer_precision_proxy": round(tp / (tp + fp), 4) if tp + fp else None,
            "source_sha256": hashes,
            "limitations": ["Negative pairs are other answers from the same meeting, not independently adjudicated unrelated labels.",
                            "QMSum answers are human summaries; this test does not measure extraction from raw turns.",
                            "Only the answers relation is evaluated; contradiction and supersession have no QMSum gold here."],
            "details": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=QMSUM)
    parser.add_argument("--output", type=Path, default=ROOT / "Data/Processed/public_answer_relations.json")
    args = parser.parse_args()
    report = evaluate(args.raw_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"details", "source_sha256"}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
