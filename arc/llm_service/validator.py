"""Output validation, in order, with a canned fallback at the end.

    1. SCHEMA         must parse; every enum value a member of the closed set
    2. GROUNDEDNESS   every amount, date and name matches the source exactly
    3. SAFETY         no threats, no legal claims, no promises we cannot keep
    4. FALLBACK       any failure falls back to the canned template
    5. LOG            prompt hash, model version, output hash, verdict

REJECT, NEVER COERCE. A model that returns an enum value one character off is
not nearly right, it is a model whose output nobody has checked. Coercing it
would mean the first genuinely wrong answer arrives looking like all the ones
that were quietly corrected.

GROUNDEDNESS IS STRING EQUALITY AGAINST THE SOURCE RECORD, and deliberately
nothing cleverer. A validator that parsed the amount and compared numerically
would accept a message saying twelve hundred and ninety nine rupees for a
record that says Rs 1,299.00, which is fine right up until it accepts one that
says twelve thousand. The source record is the only thing entitled to say what
the amount is, so the message must quote it.

THE FALLBACK IS NOT AN ERROR PATH, IT IS THE NORMAL PATH WITH THE LLM OFF. The
canned template is what every message is when `LLM_ENABLED` is false, so it is
exercised on every run rather than only when something breaks.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum

from arc.llm_service.contracts import GroundingFacts, Intent, Utterance


class Rejection(StrEnum):
    SCHEMA = "schema"
    UNGROUNDED = "ungrounded"
    UNSAFE = "unsafe"


# Phrases a collections message may never contain. Each is a claim the system
# is not entitled to make: a legal consequence it has not established, a threat,
# or a promise about somebody else's records.
#
# source: fair-practices prohibitions on misrepresenting legal consequences,
# plus the specific phrasings that recur in complaint transcripts.
_UNSAFE = (
    r"\blegal action\b",
    r"\bcourt\b",
    r"\barrest\b",
    r"\bpolice\b",
    r"\bcriminal\b",
    r"\bblacklist",
    r"\bcredit score will\b",
    r"\bseize\b",
    r"\bwe will inform your (employer|family|neighbou?rs)\b",
    r"\byou must pay (today|immediately)\b",
)
_UNSAFE_RE = tuple(re.compile(p, re.IGNORECASE) for p in _UNSAFE)

# Any run of digits that could be an amount, a date or an account number. Every
# one of these has to appear in the source record verbatim.
_FIGURE_RE = re.compile(r"[₹$]?\s?\d[\d,]*(?:\.\d+)?")


@dataclass(frozen=True)
class Verdict:
    accepted: bool
    rejection: Rejection | None = None
    detail: str = ""
    offending: tuple[str, ...] = ()

    @property
    def refused_by(self) -> str:
        return "accepted" if self.accepted else f"validator/{self.rejection}"


@dataclass
class ValidationLog:
    """What goes to the Decision Ledger. Hashes, not text."""

    prompt_hash: str
    output_hash: str
    model_version: str
    verdict: str
    task: str
    entries: list[dict[str, str]] = field(default_factory=list)

    def as_payload(self) -> dict[str, str]:
        return {
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "model_version": self.model_version,
            "verdict": self.verdict,
            "task": self.task,
        }


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def validate_schema(utterance: object) -> Verdict:
    """It must be the type it claims to be, with a closed-set intent."""
    if not isinstance(utterance, Utterance):
        return Verdict(
            False, Rejection.SCHEMA, f"expected an Utterance, got {type(utterance).__name__}"
        )
    if not isinstance(utterance.intent, Intent):
        return Verdict(False, Rejection.SCHEMA, "intent is not a member of the closed set")
    if not utterance.text.strip():
        return Verdict(False, Rejection.SCHEMA, "empty utterance")
    if not utterance.template_id:
        return Verdict(False, Rejection.SCHEMA, "no template to ground against")
    return Verdict(True)


def validate_grounding(utterance: Utterance, facts: GroundingFacts) -> Verdict:
    """Every figure in the text must appear in the source record verbatim.

    THE ATTACK THIS STOPS is a model that renders a plausible but wrong amount.
    It is plausible precisely because it is close, so a human proof-reader is
    the wrong control and a numeric tolerance is a worse one.

    THE CHECK IS SUBTRACTIVE, and that is what makes it strict. Every string
    the source record supplies is removed from the message first; anything
    number-shaped still standing afterwards came from somewhere else. So a
    message QUOTING the record passes trivially, and one that paraphrases the
    amount - which is exactly what a fluent model does - fails, because the
    paraphrase is not the string the record holds.

    Comparing figure-by-figure against the facts instead would have to decide
    whether "12" in the message is the day in "12 May 2026" or the start of a
    wrong amount, and every answer to that question is a hole.
    """
    remaining = utterance.text
    for value in sorted(facts.as_set(), key=len, reverse=True):
        if value:
            remaining = remaining.replace(value, " ")

    offending = [found.strip() for found in _FIGURE_RE.findall(remaining) if found.strip()]
    if offending:
        return Verdict(
            False,
            Rejection.UNGROUNDED,
            f"figures that do not quote the source record: {offending}",
            tuple(offending),
        )
    return Verdict(True)


def validate_safety(utterance: Utterance) -> Verdict:
    """No threat, and no legal claim the system has not established."""
    hits = [pattern.pattern for pattern in _UNSAFE_RE if pattern.search(utterance.text)]
    if hits:
        return Verdict(
            False,
            Rejection.UNSAFE,
            f"message asserts something the system is not entitled to: {hits}",
            tuple(hits),
        )
    return Verdict(True)


def validate(utterance: object, facts: GroundingFacts) -> Verdict:
    """Schema, then grounding, then safety. First failure wins and is named."""
    schema = validate_schema(utterance)
    if not schema.accepted:
        return schema
    assert isinstance(utterance, Utterance)

    grounding = validate_grounding(utterance, facts)
    if not grounding.accepted:
        return grounding

    return validate_safety(utterance)


def canned(template_id: str, facts: GroundingFacts) -> Utterance:
    """The deterministic fallback, and the normal path with the LLM off.

    Built from the source record by substitution, so it is grounded by
    construction and passes its own validator - which the tests check, because
    a fallback that could not survive validation would be a second failure
    hiding behind the first.
    """
    return Utterance(
        text=(
            f"{facts.merchant}: your payment of {facts.amount} for {facts.plan_name} "
            f"was due on {facts.due_date} and has not gone through. "
            f"Reply STOP to opt out."
        ),
        template_id=template_id,
        intent=Intent.UNCLEAR,
    )


def validated_or_canned(
    utterance: object, facts: GroundingFacts, *, template_id: str
) -> tuple[Utterance, Verdict]:
    """The whole contract in one call: validate, or fall back and say why."""
    verdict = validate(utterance, facts)
    if verdict.accepted:
        assert isinstance(utterance, Utterance)
        return utterance, verdict
    return canned(template_id, facts), verdict
