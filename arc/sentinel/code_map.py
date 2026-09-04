"""Deterministic lookup from a decline code to a cause. The CHEAP check.

It runs third, not first, and that ordering is the single most consequential
decision in this layer. Run it first and an issuer outage becomes four hundred
delinquent customers, each of whom gets dunned for their bank's incident.

Two properties matter more than coverage.

UNMATCHED FAILS CLOSED (GI-5). An unknown code returns UNKNOWN at zero
confidence and goes to a review queue. It is never guessed at, and it never
falls through to a permissive default.

A CODE THIS TABLE CANNOT SAFELY NAME IS ALSO UNKNOWN. The domain's cause labels
are closed, and there is no member for a risk-driven or technical decline.
Forcing one onto HARD_DECLINE would permanently block retries on a transaction
that was merely refused once, so those codes are listed here and mapped to
UNKNOWN on purpose - an explicit, reviewable "no safe label" rather than an
omission that reads as an oversight.

Confidences are the probability that the code means what the table says, not
the probability of recovery. They are high because a network code IS the
issuer's own statement of reason, and low where the code is ambiguous about
layer - MD01 "no mandate" being the case that matters, since it could be the
customer revoking or our own registration having broken. Mandate health runs
BEFORE this table precisely so that ambiguity is resolved by evidence rather
than by a coin flip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from arc.core.types import CauseLabel, CauseLayer, ClaimType, Rail


@dataclass(frozen=True)
class CodeMeaning:
    label: CauseLabel
    layer: CauseLayer
    confidence: float
    note: str = ""

    @property
    def is_confident(self) -> bool:
        return self.confidence > 0.0


UNMAPPED = CodeMeaning(
    label=CauseLabel.UNKNOWN,
    layer=CauseLayer.UNKNOWN,
    confidence=0.0,
    note="no entry in the code map; conservative path and review queue",
)

_NO_SAFE_LABEL = CodeMeaning(
    label=CauseLabel.UNKNOWN,
    layer=CauseLayer.UNKNOWN,
    confidence=0.0,
    note="known code with no safe label in the closed cause vocabulary",
)

_FUNDS = CodeMeaning(CauseLabel.INSUFFICIENT_FUNDS, CauseLayer.CUSTOMER, 0.92)
_EXPIRED = CodeMeaning(CauseLabel.CARD_EXPIRED, CauseLayer.CUSTOMER, 0.95)
_HARD = CodeMeaning(CauseLabel.HARD_DECLINE, CauseLayer.CUSTOMER, 0.97)
_STOP = CodeMeaning(CauseLabel.DO_NOT_RETRY, CauseLayer.CUSTOMER, 0.95)
_ISSUER = CodeMeaning(CauseLabel.ISSUER_DEGRADED, CauseLayer.ISSUER, 0.80)
_CAP = CodeMeaning(CauseLabel.MANDATE_CAP_EXCEEDED, CauseLayer.MERCHANT, 0.90)

# "No mandate" is genuinely ambiguous: the customer may have revoked it, or our
# own registration may have broken. Deliberately below the Gate's 0.80
# money-moving threshold, so a claim that reaches this entry cannot fund an
# action on the strength of it alone.
_NO_MANDATE = CodeMeaning(
    CauseLabel.MANDATE_REVOKED,
    CauseLayer.CUSTOMER,
    # Not 0.70. That is the LLM confidence cap's value, and a reader grepping
    # for where the cap is enforced should not land here by coincidence.
    0.68,
    note="ambiguous between customer revocation and broken registration",
)

# source: the code vocabularies the adapters at M5 already parse - ISO 8583
# for cards, ISO 20022 and the NACH return family for eNACH, and the NPCI UPI
# error list. Entries here mirror that table exactly; a code the gateway can
# emit and this map does not name is a defect the review queue surfaces.
CODE_MAP: Mapping[tuple[Rail, str], CodeMeaning] = MappingProxyType(
    {
        # -- card, ISO 8583 -------------------------------------------------
        (Rail.CARD, "51"): _FUNDS,
        (Rail.CARD, "54"): _EXPIRED,
        (Rail.CARD, "41"): _HARD,
        (Rail.CARD, "43"): _HARD,
        (Rail.CARD, "14"): _HARD,
        (Rail.CARD, "05"): _STOP,
        (Rail.CARD, "61"): _CAP,
        (Rail.CARD, "91"): _ISSUER,
        (Rail.CARD, "57"): _NO_SAFE_LABEL,
        (Rail.CARD, "62"): _NO_SAFE_LABEL,
        (Rail.CARD, "96"): _NO_SAFE_LABEL,
        # -- eNACH, ISO 20022 and the NACH return family --------------------
        (Rail.ENACH, "AM04"): _FUNDS,
        (Rail.ENACH, "AC01"): _HARD,
        (Rail.ENACH, "AC04"): _HARD,
        (Rail.ENACH, "AC06"): _HARD,
        (Rail.ENACH, "R16"): _HARD,
        (Rail.ENACH, "MD06"): _STOP,
        (Rail.ENACH, "R08"): _STOP,
        (Rail.ENACH, "MD01"): _NO_MANDATE,
        (Rail.ENACH, "AG01"): _CAP,
        (Rail.ENACH, "MS03"): _ISSUER,
        (Rail.ENACH, "FF01"): _NO_SAFE_LABEL,
        (Rail.ENACH, "RR04"): _NO_SAFE_LABEL,
        # -- UPI Autopay, NPCI error codes ----------------------------------
        (Rail.UPI_AUTOPAY, "Z9"): _FUNDS,
        (Rail.UPI_AUTOPAY, "ZM"): _NO_MANDATE,
        (Rail.UPI_AUTOPAY, "U69"): _CAP,
        (Rail.UPI_AUTOPAY, "91"): _ISSUER,
        (Rail.UPI_AUTOPAY, "U30"): _NO_SAFE_LABEL,
        (Rail.UPI_AUTOPAY, "U16"): _NO_SAFE_LABEL,
    }
)

# Advice codes that forbid re-presentation outright. They override whatever the
# decline code said, because a network instruction to stop is not a diagnosis
# to be weighed against others.
# source: Mastercard merchant advice codes 03 and 21.
DO_NOT_RETRY_ADVICE: frozenset[str] = frozenset({"MAC03", "MAC21"})

# Claims that never carry a decline code. An abandoned checkout and an overdue
# invoice are still failures with a cause; the cause is just not on a wire.
_CLAIM_TYPE_FALLBACK: Mapping[ClaimType, CodeMeaning] = MappingProxyType(
    {
        ClaimType.CHECKOUT_ABANDON: CodeMeaning(
            CauseLabel.CHECKOUT_ABANDONED, CauseLayer.CUSTOMER, 0.88
        ),
        ClaimType.INVOICE_OVERDUE: CodeMeaning(
            CauseLabel.INVOICE_AWAITING_APPROVAL,
            CauseLayer.CUSTOMER,
            0.60,
            note="an overdue invoice is awaiting approval until something says otherwise",
        ),
    }
)


def code_lookup(rail: Rail, code: str | None, advice: str | None = None) -> CodeMeaning:
    """What this code means, or UNKNOWN. Never a guess, never a default.

    A do-not-retry advice code wins over the decline code: the network has
    instructed the merchant to stop presenting, and that instruction does not
    become weaker because the decline reason underneath it looked soft.
    """
    if advice in DO_NOT_RETRY_ADVICE:
        return _STOP
    if code is None:
        return UNMAPPED
    return CODE_MAP.get((rail, code), UNMAPPED)


def claim_type_fallback(claim_type: ClaimType) -> CodeMeaning:
    """The cause for a claim that never had a code to look up."""
    return _CLAIM_TYPE_FALLBACK.get(claim_type, UNMAPPED)
