"""Choose answer-unit reranking on QMSum validation, before new transfer tests."""

import json

from answer_relation_model import DATASET, load, score
from bm25_score_qmsum import bm25_scores


def benchmark(split="val"):
    bundle = load()
    if bundle is None:
        raise ValueError("Entraîner d'abord answer_relation_model.py")
    methods = ("bm25", "classifier", "fusion_025", "fusion_050", "fusion_075")
    counts = {name: 0 for name in methods}
    total = 0
    for domain in ("Academic", "Product", "Committee"):
        for path in sorted((DATASET / domain / split).glob("*.json")):
            questions = json.loads(path.read_text(encoding="utf-8"))["specific_query_list"]
            questions = [(row["query"].strip(), row["answer"].strip()) for row in questions
                         if row.get("query") and row.get("answer")]
            if len(questions) < 2:
                continue
            answers = [answer for _, answer in questions]
            for index, (query, _) in enumerate(questions):
                lexical = bm25_scores(answers, query)
                semantic = score([(query, answer, 0, "") for answer in answers], bundle)
                bmrank = sorted(range(len(answers)), key=lambda i: (-lexical[i], i))
                mlrank = sorted(range(len(answers)), key=lambda i: (-semantic[i], i))
                bmpos = {candidate: rank for rank, candidate in enumerate(bmrank)}
                mlpos = {candidate: rank for rank, candidate in enumerate(mlrank)}
                predictions = {"bm25": bmrank[0], "classifier": mlrank[0]}
                for weight, name in ((0.25, "fusion_025"), (0.5, "fusion_050"), (0.75, "fusion_075")):
                    predictions[name] = max(range(len(answers)), key=lambda i: (
                        (1 - weight) / (1 + bmpos[i]) + weight / (1 + mlpos[i]), -i))
                for name, predicted in predictions.items():
                    counts[name] += predicted == index
                total += 1
    return {"split": split, "questions": total,
            "top1_accuracy": {name: round(value / total, 4) for name, value in counts.items()}}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
