"""Small CLI agent: route information requests to EF-07 and drafts to EF-08/09."""

import argparse
import json
import re
from pathlib import Path

from generate_response import ResponseEngine, normalized, select_scope
from ingest_semantic_units import DEFAULT_CONVERSATIONS, DEFAULT_UNITS
from parse_intent import classify, parse_intent
from relations_temporality import DEFAULT_RELATIONS
from search_semantic_units import DEFAULT_MODEL, METHODS, UnitSearch, rank, result_rows
from semantic_memory import DEFAULT_MEMORY_TOPICS
from simulate_actions import ActionSimulator


ACTION_INTENTS = {"draft_email", "create_task", "schedule_event"}


class SemanticPipeline:
    """Load validated EF-01/04/05 memory and one EF-02 search index per session."""

    def __init__(self, *, engine=None, method="bm25", model_path=DEFAULT_MODEL,
                 device=None, batch_size=16, top_k=5):
        if method not in METHODS or top_k < 1:
            raise ValueError("Méthode de recherche ou top-k invalide")
        self.engine = engine or ResponseEngine()
        self.method = method
        self.top_k = top_k
        self.search = UnitSearch(list(self.engine.units.values()), Path(model_path), device,
                                 batch_size, dense=method != "bm25")
        self.actions = ActionSimulator(self.engine)

    def run(self, request, *, parsed=None):
        parsed = parsed or parse_intent(request)
        scores = self.search.scores(request)[self.method]
        retrieved_ids = [self.search.units[index]["unit_id"] for index in rank(scores)]
        rules_intent = classify(request)[0]
        selected, _ = select_scope(request, parsed["subject"], self.engine.topics)
        if (rules_intent == "compare" and normalized(request).startswith("compare ")
                and selected and parsed["intent"] in {"answer", "clarify"}):
            parsed = {**parsed, "intent": "compare", "clarification_reason": None}
        question = bool(re.match(r"^(qui|quel\w*|quand|ou|comment|pourquoi|est ce)\b",
                                 normalized(request)))
        if (parsed["intent"] in ACTION_INTENTS and not (question and rules_intent == "answer")) or (
            parsed["intent"] == "clarify" and rules_intent in ACTION_INTENTS
            and parsed["clarification_reason"] != "hypothetical_not_request"
        ):
            result = self.actions.prepare(request, parsed=parsed, retrieved_ids=retrieved_ids)
        else:
            result = self.engine.generate(request, parsed=parsed, retrieved_ids=retrieved_ids)
        result["retrieval"] = {"method": self.method,
                               "results": result_rows(self.search.units, scores, self.top_k)}
        return result


def handle(request, *, engine=None, parsed=None, method="bm25"):
    return SemanticPipeline(engine=engine, method=method).run(request, parsed=parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Demande en français")
    parser.add_argument("--method", choices=METHODS, default="hybrid",
                        help="Recherche EF-02 : hybride par défaut avec E5 adapté")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--conversations", type=Path, default=DEFAULT_CONVERSATIONS)
    parser.add_argument("--units", type=Path, default=DEFAULT_UNITS)
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--topics", type=Path, default=DEFAULT_MEMORY_TOPICS)
    args = parser.parse_args()
    try:
        engine = ResponseEngine(args.conversations, args.units, args.relations, args.topics)
        pipeline = SemanticPipeline(engine=engine, method=args.method, model_path=args.model,
                                    device=args.device, top_k=args.top_k)
        response = pipeline.run(args.request)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
