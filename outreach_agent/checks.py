"""Deterministic draft checks. They run before the model-based fact check and cost nothing."""

import re

from .config import EMAIL_WORD_LIMIT, EMAIL_WORD_TOLERANCE, LINKEDIN_NOTE_LIMIT

PLACEHOLDER = re.compile(r"\[[^\]]*\]|\{[^}]*\}")
# Openers that read as templated; the draft prompt forbids them, this catches slips.
STOCK_PHRASES = (
    "i hope this email finds you well",
    "i hope this message finds you well",
    "i came across your profile",
    "i've been following your work",
    "i have been following your work",
)


def rule_problems(draft: dict, recipient_name: str = "", company: str = "") -> list[str]:
    """Problems that can be found without a model."""
    note, body = draft.get("linkedin_note", ""), draft.get("email_body", "")
    problems = []
    if len(note) > LINKEDIN_NOTE_LIMIT:
        problems.append(f"LinkedIn note is {len(note)} characters; max {LINKEDIN_NOTE_LIMIT}.")
    words = len(body.split())
    if words > EMAIL_WORD_LIMIT + EMAIL_WORD_TOLERANCE:
        problems.append(f"Email is too long ({words} words; target {EMAIL_WORD_LIMIT}).")
    if PLACEHOLDER.search(body + note + draft.get("email_subject", "")):
        problems.append("Contains a placeholder like [Company] or {name}.")
    text = (note + " " + body).lower()
    for phrase in STOCK_PHRASES:
        if phrase in text:
            problems.append(f'Stock phrase: "{phrase}".')
    first = recipient_name.split()[0].lower() if recipient_name else ""
    if first and first not in text:
        problems.append(f"Doesn't address {recipient_name} by name.")
    if company and company.lower() not in text:
        problems.append(f"Never mentions {company}.")
    return problems
