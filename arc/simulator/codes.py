"""The decline-code vocabulary the fake gateways speak.

SCOPE, stated permanently rather than as a caveat: what this module asserts is
the *structure and distribution* of decline codes - which family fires for
which underlying reason, how often, and on which rail. The code strings are a
curated subset of published vocabularies, named per entry; they are not a
complete transcription of any network's current table.

Three published vocabularies are mixed here because three rails are:

  card         ISO 8583 response codes (the numeric two-digit family)
  enach        ISO 20022 external return-reason codes (AC01, AM04, MD01 ...)
               plus the NACH return-reason family that shares NACHA's R-codes
  upi_autopay  the NPCI UPI error-code family (Z9, U30 ...)

`semantic` is the ground truth: what actually went wrong. The Sentinel's code
map at M6 has to rediscover that mapping from the code alone, and 5% of the
time the emitted code does not match the semantic at all (REMAP_RATE), which
is the adversarial noise the code map has to survive.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from arc.core.types import Rail


class Semantic(StrEnum):
    """Ground-truth reason for a failure. Never emitted on the wire."""

    INSUFFICIENT_FUNDS = "insufficient_funds"
    ISSUER_UNAVAILABLE = "issuer_unavailable"
    MANDATE_MISSING = "mandate_missing"
    MANDATE_CAP = "mandate_cap"
    CARD_EXPIRED = "card_expired"
    HARD_DECLINE = "hard_decline"
    DO_NOT_RETRY = "do_not_retry"
    TECHNICAL = "technical"
    RISK_DECLINE = "risk_decline"


@dataclass(frozen=True, slots=True)
class DeclineCode:
    code: str
    rail: Rail
    semantic: Semantic
    description: str


# source: ISO 8583 response codes. 51 insufficient funds, 05 do not honour,
# 41 lost card, 43 stolen card, 54 expired card, 57 transaction not permitted,
# 61 exceeds amount limit, 62 restricted card, 91 issuer or switch inoperative,
# 14 invalid card number.
_CARD = (
    DeclineCode("51", Rail.CARD, Semantic.INSUFFICIENT_FUNDS, "Insufficient funds"),
    DeclineCode("05", Rail.CARD, Semantic.DO_NOT_RETRY, "Do not honour"),
    DeclineCode("41", Rail.CARD, Semantic.HARD_DECLINE, "Lost card"),
    DeclineCode("43", Rail.CARD, Semantic.HARD_DECLINE, "Stolen card"),
    DeclineCode("54", Rail.CARD, Semantic.CARD_EXPIRED, "Expired card"),
    DeclineCode("57", Rail.CARD, Semantic.RISK_DECLINE, "Transaction not permitted"),
    DeclineCode("61", Rail.CARD, Semantic.MANDATE_CAP, "Exceeds amount limit"),
    DeclineCode("62", Rail.CARD, Semantic.RISK_DECLINE, "Restricted card"),
    DeclineCode("91", Rail.CARD, Semantic.ISSUER_UNAVAILABLE, "Issuer or switch inoperative"),
    DeclineCode("14", Rail.CARD, Semantic.HARD_DECLINE, "Invalid card number"),
    DeclineCode("96", Rail.CARD, Semantic.TECHNICAL, "System malfunction"),
)

# source: ISO 20022 external return-reason codes (AC01 incorrect account, AC04
# closed account, AC06 blocked account, AG01 transaction forbidden, AM04
# insufficient funds, MD01 no mandate, MD06 refund request by end customer,
# MS03 reason not specified by agent) plus the NACH return family's shared
# NACHA R-codes (R08 payment stopped, R16 account frozen). MD06 and R08 are
# also the do-not-retry codes the M3 registry already blocks on.
_ENACH = (
    DeclineCode("AM04", Rail.ENACH, Semantic.INSUFFICIENT_FUNDS, "Insufficient funds"),
    DeclineCode("AC04", Rail.ENACH, Semantic.HARD_DECLINE, "Closed account number"),
    DeclineCode("AC01", Rail.ENACH, Semantic.HARD_DECLINE, "Incorrect account number"),
    DeclineCode("AC06", Rail.ENACH, Semantic.HARD_DECLINE, "Blocked account"),
    DeclineCode("MD01", Rail.ENACH, Semantic.MANDATE_MISSING, "No mandate"),
    DeclineCode("MD06", Rail.ENACH, Semantic.DO_NOT_RETRY, "Refund request by end customer"),
    DeclineCode("R08", Rail.ENACH, Semantic.DO_NOT_RETRY, "Payment stopped by drawer"),
    DeclineCode("R16", Rail.ENACH, Semantic.HARD_DECLINE, "Account frozen"),
    DeclineCode("AG01", Rail.ENACH, Semantic.MANDATE_CAP, "Transaction forbidden - limit"),
    DeclineCode("MS03", Rail.ENACH, Semantic.ISSUER_UNAVAILABLE, "Reason not specified by agent"),
    DeclineCode("FF01", Rail.ENACH, Semantic.TECHNICAL, "Invalid file format"),
    DeclineCode("RR04", Rail.ENACH, Semantic.RISK_DECLINE, "Regulatory reason"),
)

# source: NPCI UPI error-code family. Z9 insufficient funds, U30 debit
# failure, U16 risk decline, ZM invalid credential / mandate, 91 issuer
# unavailable (shared with the card family via the switch).
_UPI = (
    DeclineCode("Z9", Rail.UPI_AUTOPAY, Semantic.INSUFFICIENT_FUNDS, "Insufficient funds"),
    DeclineCode("U30", Rail.UPI_AUTOPAY, Semantic.TECHNICAL, "Debit failure"),
    DeclineCode("U16", Rail.UPI_AUTOPAY, Semantic.RISK_DECLINE, "Risk threshold exceeded"),
    DeclineCode("ZM", Rail.UPI_AUTOPAY, Semantic.MANDATE_MISSING, "Invalid mandate or credential"),
    DeclineCode("91", Rail.UPI_AUTOPAY, Semantic.ISSUER_UNAVAILABLE, "Issuer unavailable"),
    DeclineCode("U69", Rail.UPI_AUTOPAY, Semantic.MANDATE_CAP, "Debit exceeds mandate limit"),
)

CODES: tuple[DeclineCode, ...] = _CARD + _ENACH + _UPI

# source: Mastercard merchant advice codes. 01 new account information,
# 02 try again later, 03 do not try again, 21 recurring payment cancellation.
# MAC03 and MAC21 are the two that forbid re-presentation; the M3 registry
# blocks a retry on seeing either.
MERCHANT_ADVICE_CODES: tuple[str, ...] = ("MAC01", "MAC02", "MAC03", "MAC21")
DO_NOT_RETRY_ADVICE: frozenset[str] = frozenset({"MAC03", "MAC21"})

# The Sentinel has to survive a gateway that lies. 5% of emitted codes are
# drawn from a semantic other than the true one - the wrong-code injection the
# build doc calls for.
REMAP_RATE = 0.05

_BY_RAIL: dict[Rail, tuple[DeclineCode, ...]] = {
    Rail.CARD: _CARD,
    Rail.ENACH: _ENACH,
    Rail.UPI_AUTOPAY: _UPI,
    Rail.INVOICE: (),
}


def codes_for(rail: Rail) -> tuple[DeclineCode, ...]:
    return _BY_RAIL[rail]


# Not every rail's published vocabulary has a distinct code for every reason.
# Where it does not, this table declares which code the reason collapses onto,
# rather than letting the collapse happen silently. The collapse is itself
# realistic - it is precisely why the Sentinel's code map cannot be a clean
# lookup and why an unmapped code must fail closed rather than guess.
_COLLAPSES_ONTO: Mapping[tuple[Rail, Semantic], Semantic] = MappingProxyType(
    {
        # A revoked or closed UPI mandate comes back as an invalid mandate.
        (Rail.UPI_AUTOPAY, Semantic.DO_NOT_RETRY): Semantic.MANDATE_MISSING,
        (Rail.UPI_AUTOPAY, Semantic.HARD_DECLINE): Semantic.MANDATE_MISSING,
        (Rail.UPI_AUTOPAY, Semantic.CARD_EXPIRED): Semantic.MANDATE_MISSING,
        # A card has no mandate of its own; a lost credential reads as expired.
        (Rail.CARD, Semantic.MANDATE_MISSING): Semantic.CARD_EXPIRED,
        # A bank mandate has no card behind it to expire.
        (Rail.ENACH, Semantic.CARD_EXPIRED): Semantic.MANDATE_MISSING,
    }
)


def code_for(rail: Rail, semantic: Semantic) -> DeclineCode | None:
    """The truthful code for a reason on a rail, or None if the rail has none.

    An invoice does not decline, so `Rail.INVOICE` returns None throughout;
    an overdue invoice carries an ageing bucket, not a response code.

    Where the rail has no distinct code, the declared collapse applies once -
    never recursively, so the table cannot hide a cycle.
    """
    for entry in _BY_RAIL[rail]:
        if entry.semantic is semantic:
            return entry

    collapsed = _COLLAPSES_ONTO.get((rail, semantic))
    if collapsed is not None:
        for entry in _BY_RAIL[rail]:
            if entry.semantic is collapsed:
                return entry
    return None


def emit_code(rail: Rail, semantic: Semantic, generator: np.random.Generator) -> str | None:
    """The code that actually goes on the wire, remapped 5% of the time.

    The remap is drawn from the same rail, so it stays plausible: a card
    decline never comes back as a NACH return reason. What it loses is its
    relationship to the truth, which is the point.
    """
    truthful = code_for(rail, semantic)
    if truthful is None:
        return None

    pool = _BY_RAIL[rail]
    if len(pool) > 1 and generator.random() < REMAP_RATE:
        wrong = tuple(entry for entry in pool if entry.semantic is not semantic)
        if wrong:
            return wrong[int(generator.integers(len(wrong)))].code
    return truthful.code


def describe(rail: Rail, code: str | None) -> str:
    """Human-readable reason for a code, as the gateway would send it."""
    if code is None:
        return ""
    for entry in _BY_RAIL[rail]:
        if entry.code == code:
            return entry.description
    return "Unspecified"
