"""Small CLI agent: route information requests to EF-07 and drafts to EF-08/09."""

import argparse
import json
import re

from generate_response import ResponseEngine, normalized
from parse_intent import classify, parse_intent
from simulate_actions import ActionSimulator


ACTION_INTENTS = {"draft_email", "create_task", "schedule_event"}


def handle(request, *, engine=None, parsed=None):
    engine = engine or ResponseEngine()
    parsed = parsed or parse_intent(request)
    rules_intent = classify(request)[0]
    question = bool(re.match(r"^(qui|quel\w*|quand|ou|comment|pourquoi|est ce)\b",
                             normalized(request)))
    if (parsed["intent"] in ACTION_INTENTS and not (question and rules_intent == "answer")) or (
        parsed["intent"] == "clarify" and rules_intent in ACTION_INTENTS
        and parsed["clarification_reason"] != "hypothetical_not_request"
    ):
        return ActionSimulator(engine).prepare(request, parsed=parsed)
    return engine.generate(request, parsed=parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", help="Demande en français")
    args = parser.parse_args()
    try:
        response = handle(args.request)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
