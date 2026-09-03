"""The domain contract. Frozen at M1; every later milestone imports these.

`ActionType`, `ClaimState`, `LEGAL_TRANSITIONS` and the `Claim` field set are
fixed. Adding a member to `ActionType` would silently widen the action space
that the Gate, the Allocator and the cost vectors are all defined over, so it
is closed: thirteen members, never extended at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any
from uuid import UUID

from arc.core.ids import is_subject_token
from arc.core.money import Paise
from arc.core.time_authority import ensure_utc

SHA256_BYTES = 32

# Evidence is a closed vocabulary of structured fields. The vocabulary itself
# is fixed by the normaliser at M5; what is enforced here is the SHAPE, which
# is what keeps free text out: a bank narration cannot be smuggled through a
# field that only accepts short scalars.
MAX_EVIDENCE_STRING = 128


class ClaimType(StrEnum):
    MANDATE_FAILURE = "mandate_failure"
    CARD_DECLINE = "card_decline"
    CHECKOUT_ABANDON = "checkout_abandon"
    INVOICE_OVERDUE = "invoice_overdue"


class Rail(StrEnum):
    UPI_AUTOPAY = "upi_autopay"
    ENACH = "enach"
    CARD = "card"
    INVOICE = "invoice"


class CauseLayer(StrEnum):
    """Whose fault, which decides whether a human is ever contacted.

    The layer matters more than the label: ISSUER causes require zero customer
    contact, MERCHANT causes are fixed silently at the rail, and only CUSTOMER
    causes justify outreach at all.
    """

    ISSUER = "issuer"
    MERCHANT = "merchant"
    CUSTOMER = "customer"
    UNKNOWN = "unknown"


class CohortVerdict(StrEnum):
    """INSUFFICIENT_POWER is a distinct answer and is never coerced to NORMAL.

    Silently reading "not enough sample" as "nothing wrong" is the exact bug
    that dunns 400 customers through an issuer outage on a thin issuer.
    """

    DEGRADED = "degraded"
    NORMAL = "normal"
    INSUFFICIENT_POWER = "insufficient_power"


class DiagnosisPath(StrEnum):
    """Which of the four ordered Sentinel checks answered."""

    COHORT = "cohort"
    MANDATE = "mandate"
    CODE_MAP = "code_map"
    LLM = "llm"


class CauseLabel(StrEnum):
    """Closed set of attributable causes across the four claim types."""

    # ISSUER layer
    ISSUER_OUTAGE = "issuer_outage"
    ISSUER_DEGRADED = "issuer_degraded"

    # MERCHANT layer - fixable at the rail, with no customer contact
    MANDATE_ORPHANED = "mandate_orphaned"
    MANDATE_CAP_EXCEEDED = "mandate_cap_exceeded"
    MANDATE_EXPIRED = "mandate_expired"
    PREDEBIT_NOTICE_MISSING = "predebit_notice_missing"
    WRONG_DEBIT_DATE = "wrong_debit_date"

    # CUSTOMER layer
    INSUFFICIENT_FUNDS = "insufficient_funds"
    CARD_EXPIRED = "card_expired"
    HARD_DECLINE = "hard_decline"
    DO_NOT_RETRY = "do_not_retry"
    MANDATE_REVOKED = "mandate_revoked"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    INVOICE_AWAITING_APPROVAL = "invoice_awaiting_approval"
    INVOICE_DISPUTED = "invoice_disputed"

    # Unmatched. Fails closed onto the conservative path (GI-5).
    UNKNOWN = "unknown"


class ClaimState(StrEnum):
    DETECTED = "detected"
    DIAGNOSED = "diagnosed"
    SUPPRESSED = "suppressed"
    SELF_HEALING = "self_healing"
    PLANNED = "planned"
    IN_TREATMENT = "in_treatment"
    PROMISED = "promised"
    ESCALATED = "escalated"
    DISPUTED = "disputed"
    RECOVERED = "recovered"
    REVERSED = "reversed"
    WRITTEN_OFF = "written_off"
    FORBORNE = "forborne"


class ActionType(StrEnum):
    """CLOSED. Thirteen members. Never extended at runtime.

    A bounded recovery workflow is the requirement; an open action space cannot
    be bounded, gated, costed, or optimised over.
    """

    DO_NOTHING = "do_nothing"
    RETRY = "retry"
    CARD_UPDATER = "card_updater"
    MANDATE_RE_REGISTER = "mandate_re_register"
    RAIL_FALLBACK = "rail_fallback"
    WHATSAPP_UTILITY = "whatsapp_utility"
    SMS = "sms"
    EMAIL = "email"
    PAYMENT_LINK = "payment_link"
    VOICE_CALL = "voice_call"
    INSTALMENT_OFFER = "instalment_offer"
    HUMAN_HANDOFF = "human_handoff"
    STATUTORY_NOTICE = "statutory_notice"


# An LLM-derived cause can never alone justify a money-moving action.
LLM_CONFIDENCE_CAP = 0.70


@dataclass(frozen=True)
class Cause:
    """Immutable. New evidence creates a superseding record; both are retained."""

    label: CauseLabel
    layer: CauseLayer
    confidence: float
    derived_from: DiagnosisPath
    cohort_power: CohortVerdict

    def __post_init__(self) -> None:
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError(f"confidence must be a number, got {type(self.confidence).__name__}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        if self.derived_from is DiagnosisPath.LLM and self.confidence > LLM_CONFIDENCE_CAP:
            raise ValueError(
                f"LLM-derived cause confidence {self.confidence} exceeds the "
                f"{LLM_CONFIDENCE_CAP} cap"
            )


def _reject_float(name: str, value: object) -> int:
    """Monetary fields accept int paise and nothing else (GI-2)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be integer paise, got {type(value).__name__} ({value!r}); "
            "float and Decimal are banned for monetary values"
        )
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _validate_evidence_value(key: str, value: object) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, float):
        raise TypeError(
            f"evidence_structured[{key!r}] is a float; use integer paise or basis points"
        )
    if isinstance(value, int):
        return
    if isinstance(value, str):
        if len(value) > MAX_EVIDENCE_STRING:
            raise ValueError(
                f"evidence_structured[{key!r}] is {len(value)} characters, over the "
                f"{MAX_EVIDENCE_STRING} limit; free text belongs behind evidence_ref"
            )
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_evidence_value(key, item)
        return
    raise TypeError(
        f"evidence_structured[{key!r}] is {type(value).__name__}; only scalars and "
        "flat sequences of scalars are ledgerable"
    )


def _freeze_evidence(evidence: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError(f"evidence_structured must be a mapping, got {type(evidence).__name__}")
    for key, value in evidence.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError(f"evidence_structured key {key!r} is not a closed-vocabulary name")
        _validate_evidence_value(key, value)
    return MappingProxyType(dict(evidence))


@dataclass(frozen=True)
class Claim:
    """One unit of money that should have arrived and did not.

    All four leak surfaces normalise to this type, which is what lets a single
    contact budget be shared across them.

    `amount_paise` is what failed. `ltv_remaining_paise` is what is at risk;
    the objective is weighted by the second, because aggressive recovery of a
    small failed charge that costs a high-value customer is a loss.
    """

    claim_id: UUID
    subject_token: str
    amount_paise: Paise
    ltv_remaining_paise: Paise
    claim_type: ClaimType
    rail: Rail
    detected_at: datetime
    evidence_structured: Mapping[str, Any] = field(default_factory=dict)
    evidence_ref: str | None = None
    evidence_hash: bytes = b"\x00" * SHA256_BYTES
    cause: Cause | None = None
    state: ClaimState = ClaimState.DETECTED

    def __post_init__(self) -> None:
        if not isinstance(self.claim_id, UUID):
            raise TypeError(f"claim_id must be a UUID, got {type(self.claim_id).__name__}")

        # A raw phone number or email here would enter the immutable ledger.
        if not is_subject_token(self.subject_token):
            raise ValueError(
                f"subject_token {self.subject_token!r} is not a derived token; "
                "raw identifiers must not travel past the normaliser"
            )

        _reject_float("amount_paise", self.amount_paise)
        _reject_float("ltv_remaining_paise", self.ltv_remaining_paise)
        ensure_utc(self.detected_at)

        if not isinstance(self.evidence_hash, bytes) or len(self.evidence_hash) != SHA256_BYTES:
            raise ValueError(f"evidence_hash must be {SHA256_BYTES} bytes of SHA-256 digest")
        if self.evidence_ref is not None and not str(self.evidence_ref).strip():
            raise ValueError("evidence_ref must be a non-empty pointer or None")

        object.__setattr__(self, "evidence_structured", _freeze_evidence(self.evidence_structured))


# ---------------------------------------------------------------------------
# The claim finite state machine
# ---------------------------------------------------------------------------
class IllegalTransition(Exception):
    """An attempt to move a claim along an edge that does not exist."""


LEGAL_TRANSITIONS: Mapping[ClaimState, frozenset[ClaimState]] = MappingProxyType(
    {
        ClaimState.DETECTED: frozenset({ClaimState.DIAGNOSED}),
        ClaimState.DIAGNOSED: frozenset(
            {
                ClaimState.SUPPRESSED,
                ClaimState.SELF_HEALING,
                ClaimState.PLANNED,
                ClaimState.WRITTEN_OFF,
            }
        ),
        ClaimState.SUPPRESSED: frozenset({ClaimState.PLANNED, ClaimState.DIAGNOSED}),
        ClaimState.SELF_HEALING: frozenset({ClaimState.RECOVERED, ClaimState.PLANNED}),
        ClaimState.PLANNED: frozenset({ClaimState.IN_TREATMENT, ClaimState.WRITTEN_OFF}),
        ClaimState.IN_TREATMENT: frozenset(
            {
                ClaimState.PROMISED,
                ClaimState.ESCALATED,
                ClaimState.DISPUTED,
                ClaimState.RECOVERED,
                ClaimState.WRITTEN_OFF,
                ClaimState.FORBORNE,
            }
        ),
        ClaimState.PROMISED: frozenset(
            {
                ClaimState.RECOVERED,
                ClaimState.IN_TREATMENT,
                ClaimState.WRITTEN_OFF,
                ClaimState.FORBORNE,
            }
        ),
        ClaimState.ESCALATED: frozenset(
            {ClaimState.RECOVERED, ClaimState.WRITTEN_OFF, ClaimState.FORBORNE}
        ),
        ClaimState.DISPUTED: frozenset({ClaimState.IN_TREATMENT, ClaimState.WRITTEN_OFF}),
        ClaimState.RECOVERED: frozenset({ClaimState.REVERSED}),
        ClaimState.REVERSED: frozenset({ClaimState.IN_TREATMENT, ClaimState.WRITTEN_OFF}),
        # Absorbing. Zero outgoing edges, including to each other.
        # FORBORNE is the hardship path: no expected-value argument reopens it.
        ClaimState.FORBORNE: frozenset(),
        ClaimState.WRITTEN_OFF: frozenset(),
    }
)

ABSORBING_STATES: frozenset[ClaimState] = frozenset(
    state for state, onward in LEGAL_TRANSITIONS.items() if not onward
)


def can_transition(frm: ClaimState, to: ClaimState) -> bool:
    return to in LEGAL_TRANSITIONS[frm]


def transition(claim: Claim, to_state: ClaimState) -> Claim:
    """Move a claim along a legal edge, returning a new Claim.

    Raises rather than returning a status, because a state machine that can be
    ignored is not a state machine. There is no `force` argument.
    """
    if not isinstance(to_state, ClaimState):
        raise IllegalTransition(f"{to_state!r} is not a ClaimState")

    onward = LEGAL_TRANSITIONS[claim.state]
    if to_state not in onward:
        if claim.state in ABSORBING_STATES:
            raise IllegalTransition(
                f"{claim.state} is absorbing; no transition out is legal (attempted -> {to_state})"
            )
        legal = ", ".join(sorted(onward)) or "nothing"
        raise IllegalTransition(f"{claim.state} -> {to_state} is not legal; legal: {legal}")

    return replace(claim, state=to_state)
