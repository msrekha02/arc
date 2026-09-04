"""What an adapter is allowed to say, and the closed vocabulary it says it in.

`RawEvent` is the output of translation and the input to decision. It is
deliberately not a `Claim`: it still speaks in the gateway's terms - a debit
that failed, an invoice that aged - and it still carries personal data, because
the redaction boundary is at L1 and not here.

The split inside `RawEvent` is the whole point. `PersonalData` is everything
that must reach the subject store and nothing else. Every other field is
structured, closed-vocabulary, and safe to hash-chain.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.core.types import Rail


class WireKind(StrEnum):
    """What the gateway says happened, in a closed set.

    Not `ClaimType`. Mapping one onto the other is a decision, and decisions
    belong to the normaliser; an adapter that chose a claim type would be
    deciding what kind of problem this is.
    """

    DEBIT_CAPTURED = "debit_captured"
    DEBIT_FAILED = "debit_failed"
    CHECKOUT_ABANDONED = "checkout_abandoned"
    INVOICE_OVERDUE = "invoice_overdue"


# Events that mean money did not arrive. A capture is ingested too, because the
# cohort detector at M6 needs the denominator and because a capture supersedes
# the failure it follows.
FAILURE_KINDS: frozenset[WireKind] = frozenset(
    {WireKind.DEBIT_FAILED, WireKind.CHECKOUT_ABANDONED, WireKind.INVOICE_OVERDUE}
)


class Initiator(StrEnum):
    """Who presented the debit.

    The gateway retries on its own schedule and those attempts count against
    the network cap whether or not we issued them. Losing that distinction here
    makes the retry counter wrong at M9.
    """

    MERCHANT = "merchant"
    GATEWAY = "gateway"


@dataclass(frozen=True)
class PersonalData:
    """Everything that must not cross the redaction boundary.

    Kept as one object rather than loose fields so the normaliser hands the
    whole thing to the subject store in a single move. There is no method here
    that returns a subset for the ledger, because that method is how a name
    ends up hash-chained.
    """

    name: str | None = None
    email: str | None = None
    phone: str | None = None
    # Free text from the rail: bank narrations, gateway descriptions. This is
    # the field that makes the boundary necessary rather than tidy.
    narration: str | None = None
    # Rail identifiers that resolve to a person or to their bank instrument:
    # VPA, IFSC, account tail, UMRN, GSTIN. A mandate reference belongs here
    # rather than on the event, so there is exactly one place holding a raw
    # identifier and exactly one derivation that crosses the boundary.
    identifiers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identifiers", MappingProxyType(dict(self.identifiers)))

    def is_empty(self) -> bool:
        return not any((self.name, self.email, self.phone, self.narration, self.identifiers))


@dataclass(frozen=True)
class RawEvent:
    """One verified, parsed delivery. Translated, not yet decided upon."""

    source: str
    event_id: str
    event_timestamp: datetime
    kind: WireKind
    rail: Rail

    account_ref: str
    customer_ref: str | None
    amount_paise: Paise

    raw: bytes
    raw_hash: bytes

    succeeded: bool = False
    attempt: int = 1
    initiated_by: Initiator = Initiator.MERCHANT

    decline_code: str | None = None
    advice_code: str | None = None
    # The issuer is a bank, not a person, so it crosses the boundary as-is and
    # becomes the key the cohort detector groups on at M6.
    issuer_ref: str | None = None
    ageing_bucket: str | None = None
    days_overdue: int | None = None

    # A NACH return file reports a settlement DATE and no instant. Saying so
    # is more honest than inventing a time, and it lets the pipeline take the
    # instant from the signed delivery instead of guessing midnight.
    date_only: bool = False

    personal: PersonalData = field(default_factory=PersonalData)

    def __post_init__(self) -> None:
        ensure_utc(self.event_timestamp)
        if not self.source or not self.event_id:
            raise ValueError("an event must carry a source and an event id")
        if not isinstance(self.raw_hash, bytes) or len(self.raw_hash) != 32:
            raise ValueError("raw_hash must be a SHA-256 digest")

    @property
    def is_failure(self) -> bool:
        return self.kind in FAILURE_KINDS and not self.succeeded


class MalformedPayload(ValueError):
    """The bytes verified but did not parse. Never a silent drop."""


def require(payload: Mapping[str, Any], *path: str) -> Any:
    """Read a nested field, or say precisely which one was missing.

    Adapters fail loudly on a shape they do not recognise. A gateway that
    changes its payload must trip the source breaker, not quietly produce a
    claim with a zero amount.
    """
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise MalformedPayload(f"missing field {'.'.join(path)}")
        cursor = cursor[key]
    return cursor


def optional(payload: Mapping[str, Any], *path: str) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, Mapping) or key not in cursor:
            return None
        cursor = cursor[key]
    return cursor
