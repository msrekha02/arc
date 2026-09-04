"""The four sanctioned LLM tasks, and the much longer list of what it may not do.

The model is confined by STRUCTURE, not by prompt text. A prompt asking a model
not to compute an amount is a request; a type system in which no LLM output can
carry an amount is a constraint. Everything here is closed enums and schemas so
that a violation is a parse failure rather than a judgement call.

THE FOUR TASKS:

    1. free-text cause classification    enum + confidence, capped at 0.70
    2. message generation                text validated against a template
    3. in-call conversation              utterance + intent from a closed set
    4. PTP / intent extraction           structured record + confidence

WHAT IT MAY NEVER DO. Compute or alter a monetary amount; choose an action;
schedule a time; evaluate a compliance rule; write to the ledger; read latent
simulation state; see another subject's data; determine a cause layer without a
deterministic corroborator; decide an escalation tier; or produce free-form
output that reaches a customer unvalidated.

    NOTE WHICH OF THOSE ARE ENFORCED HERE AND WHICH ELSEWHERE. The ones about
    amounts, actions and times are enforced by these types having nowhere to
    put them. The ones about the ledger, the latent state and the Gate are
    enforced by the import bans - `arc/gate`, `arc/allocator` and `arc/money`
    may not import this package at all, so an LLM answer cannot reach a rule
    evaluation however it is dressed up.

LLM_ENABLED IS A REAL SWITCH, NOT A MOCK. With it off the client returns
nothing and every caller takes its deterministic fallback. The system must be
completely functional in that state, degrading in message quality and never in
correctness or compliance, and `make demo-adversarial` runs the whole pipeline
that way to show it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

from arc.core.types import CauseLabel, CauseLayer

# An LLM-derived cause may never exceed this, and `Cause.__post_init__` is
# where that is enforced rather than here - one place for the cap, and it is
# the domain's, not this package's.
LLM_CONFIDENCE_CAP = 0.70

# Below this, an extraction changes no state and goes to a review queue.
# source: the confidence at which a self-reported extraction stops being more
# accurate than asking a human, on the labelled sets these are calibrated on.
EXTRACTION_THRESHOLD = 0.75


def llm_enabled() -> bool:
    """Read once per call, so a test can flip it without reimporting.

    Default OFF. A system whose default requires a model is a system that
    cannot be demonstrated without one, and the degradation path is the thing
    most worth demonstrating.
    """
    return os.environ.get("LLM_ENABLED", "false").strip().lower() in {"1", "true", "yes"}


class LlmTask(StrEnum):
    """The four. A task not in this enum has no sanctioned path to a model."""

    CAUSE_CLASSIFICATION = "cause_classification"
    MESSAGE_GENERATION = "message_generation"
    CONVERSATION_TURN = "conversation_turn"
    INTENT_EXTRACTION = "intent_extraction"


class Intent(StrEnum):
    """Closed set of things a caller may be understood to have said.

    An utterance the model cannot map into this set is not understood, and
    "not understood" is a real answer that routes to a human rather than a
    guess that routes to a debit.
    """

    PROMISE_TO_PAY = "promise_to_pay"
    DISPUTE = "dispute"
    HARDSHIP = "hardship"
    WRONG_PARTY = "wrong_party"
    REQUEST_HUMAN = "request_human"
    ALREADY_PAID = "already_paid"
    REFUSE = "refuse"
    UNCLEAR = "unclear"


# Intents that must stop the automation the moment they are recognised, no
# matter how confident the model is or is not. Confidence gates whether a state
# CHANGES; it does not gate whether the machine stops.
STOP_INTENTS: frozenset[Intent] = frozenset(
    {Intent.HARDSHIP, Intent.WRONG_PARTY, Intent.REQUEST_HUMAN, Intent.DISPUTE}
)


@dataclass(frozen=True)
class CauseAnswer:
    """Task 1. An enum and a confidence, and nowhere to put anything else."""

    label: CauseLabel
    layer: CauseLayer
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@dataclass(frozen=True)
class IntentAnswer:
    """Task 4. What was understood, how sure, and the span it came from.

    NO AMOUNT FIELD, NO DATE FIELD. A promise-to-pay's amount and date are
    read from the source record by deterministic extraction and validated
    against it; the model says only that a promise was made. There is nowhere
    here for a hallucinated figure to sit.
    """

    intent: Intent
    confidence: float
    evidence_span: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")

    @property
    def actionable(self) -> bool:
        """Whether this may change state. Below threshold: review queue."""
        return self.confidence >= EXTRACTION_THRESHOLD


@dataclass(frozen=True)
class Utterance:
    """Tasks 2 and 3. Text, and the template it must be grounded in."""

    text: str
    template_id: str
    intent: Intent = Intent.UNCLEAR


@dataclass(frozen=True)
class GroundingFacts:
    """The source record every factual claim must match, character for character.

    Amounts arrive as ALREADY-FORMATTED STRINGS. The model never sees a number
    it could do arithmetic on, and the validator never has to decide whether
    two spellings of the same amount agree - they either match the source
    string or they do not.
    """

    amount: str
    due_date: str
    plan_name: str
    merchant: str

    def as_set(self) -> frozenset[str]:
        return frozenset({self.amount, self.due_date, self.plan_name, self.merchant})
