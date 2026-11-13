"""Conservative candidate owner extraction from explicit English action text.

The output is a proposal, never an established responsibility or contact.
"""

import re


AMBIGUOUS = {"group", "team", "one team", "one team member", "some team",
             "team members", "group members", "they", "he", "she", "we"}


def predict_owner(text):
    match = re.match(r"^(?:the\s+)?([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+)?|project manager|industrial designer|user interface designer|marketing expert|marketing executive|interface designer)\s+(?:will|is going to|needs to)\b",
                     text, re.I)
    if not match:
        return None
    owner = match.group(1).strip()
    return None if owner.casefold() in AMBIGUOUS else owner
