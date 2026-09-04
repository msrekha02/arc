"""Is our own setup broken? The second question, asked before blaming anyone.

A mandate that silently orphaned after a card reissue produces a debit failure
indistinguishable, from the code alone, from a customer who revoked it. The
difference is everything: one is repaired at the rail with zero contact, the
other is a customer-layer cause that justifies outreach. Asking this before
opening the code map is what keeps the ~3% orphaned cohort out of the dunning
queue.

MERCHANT-LAYER CAUSES ARE OUR FAULT. The customer never withheld anything, so
the honest response is to fix the mandate and re-present, not to message them
about a failure we caused.

GROUPING IS ON THE PSEUDONYM, NEVER THE RAW UMRN. `mandate_ref` is the one-way
`mnd_` derivation L1 puts on the claim; the raw identifier stayed behind on the
erasable side of the redaction boundary and must not reappear here. The history
refuses anything that is not a pseudonym, so a well-meaning caller who has the
real UMRN to hand cannot use it by accident.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.core.types import CauseLabel, CauseLayer

MANDATE_REF_RE = re.compile(r"^mnd_[0-9a-f]{32}$")

# Consecutive failures since a reissue before the mandate is called orphaned.
# One failure after a reissue is a coincidence; a clean run of them, with no
# success in between, is a broken registration.
ORPHAN_FAILURE_RUN = 2

# source: RBI e-mandate framework - a pre-debit notification must reach the
# customer at least 24 hours before presentation.
PREDEBIT_NOTICE_HOURS = 24

# Confidences. High where the evidence is arithmetic (an amount above a cap is
# not a matter of opinion), lower where it is inferential (a run of failures
# after a reissue is strong but circumstantial).
CONFIDENCE_CAP_EXCEEDED = 0.96
CONFIDENCE_EXPIRED = 0.94
CONFIDENCE_ORPHANED = 0.88
CONFIDENCE_PREDEBIT = 0.90
CONFIDENCE_WRONG_DATE = 0.82


class RawMandateIdentifier(ValueError):
    """A raw UMRN was offered where a pseudonym was required.

    Refused rather than hashed on the caller's behalf: silently accepting it
    would mean a raw bank identifier had travelled this far, and the fix is at
    the boundary that let it through, not here.
    """


@dataclass(frozen=True)
class MandateFacts:
    """What OUR OWN records say about the mandate behind this claim.

    Passed in, never fetched - the Sentinel performs no I/O, so a diagnosis is
    reproducible under replay six months later.

    `status` is what the merchant's records claim. For the orphaned cohort it
    says `active`, and it is wrong. That is what makes the cohort silent, and
    why this check reasons from dates and outcomes rather than from a field
    that would simply hand over the answer.
    """

    mandate_ref: str | None = None
    status: str = "unknown"
    registered_at: datetime | None = None
    instrument_reissued_at: datetime | None = None
    expires_at: datetime | None = None
    cap_paise: Paise | None = None

    # Pre-debit notification. `predebit_required` says whether we actually
    # track notices for this mandate; only then does an absent one mean a
    # notice was not sent.
    #
    # WHY the flag rather than treating absence as failure: "we have no record"
    # and "no notice went out" are different facts, and conflating them would
    # make almost every eNACH claim a merchant-layer cause. That is not
    # conservative, it is wrong - the claim would route to SELF_HEALING, never
    # be pursued, and the money would be written off by a bookkeeping gap.
    # Abstaining is not failing open: the code map still runs, and it produces
    # a customer-layer cause with a real confidence behind it.
    predebit_required: bool = False
    predebit_notice_at: datetime | None = None

    scheduled_debit_day: int | None = None

    def __post_init__(self) -> None:
        if self.mandate_ref is not None and not MANDATE_REF_RE.match(self.mandate_ref):
            raise RawMandateIdentifier(
                "mandate_ref must be the mnd_ pseudonym derived at the redaction "
                "boundary, never a raw UMRN"
            )


@dataclass(frozen=True)
class MandateAttempt:
    """One presentation against a mandate, as the ingest stream saw it."""

    mandate_ref: str
    at: datetime
    succeeded: bool


@dataclass
class MandateHistory:
    """Presentations grouped by mandate pseudonym."""

    _attempts: dict[str, list[MandateAttempt]] = field(default_factory=dict)

    def record(self, mandate_ref: str, at: datetime, *, succeeded: bool) -> None:
        if not MANDATE_REF_RE.match(mandate_ref or ""):
            raise RawMandateIdentifier(
                f"{mandate_ref!r} is not a mandate pseudonym; grouping on a raw "
                "UMRN would put a bank identifier into the diagnosis path"
            )
        ensure_utc(at)
        self._attempts.setdefault(mandate_ref, []).append(
            MandateAttempt(mandate_ref, at, succeeded)
        )

    def since(self, mandate_ref: str | None, after: datetime | None) -> list[MandateAttempt]:
        """Attempts strictly after `after`, oldest first."""
        if mandate_ref is None:
            return []
        attempts = sorted(self._attempts.get(mandate_ref, []), key=lambda a: a.at)
        if after is None:
            return attempts
        return [attempt for attempt in attempts if attempt.at > after]

    def attempts_for(self, mandate_ref: str | None) -> list[MandateAttempt]:
        return self.since(mandate_ref, None)


def build_history(events: Iterable) -> MandateHistory:
    """Group an ingest stream by mandate pseudonym.

    Takes the pseudonym from the caller rather than deriving it, because
    deriving it here would need the raw UMRN and this module must never see
    one. Events without a mandate reference are skipped, not guessed at.
    """
    history = MandateHistory()
    for mandate_ref, at, succeeded in events:
        if mandate_ref:
            history.record(mandate_ref, at, succeeded=succeeded)
    return history


@dataclass(frozen=True)
class MandateResult:
    """A merchant-layer finding, or nothing. Never a customer-layer one."""

    label: CauseLabel | None = None
    confidence: float = 0.0
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.label is not None

    @property
    def layer(self) -> CauseLayer:
        return CauseLayer.MERCHANT if self.found else CauseLayer.UNKNOWN


def mandate_health(
    *,
    amount_paise: Paise,
    facts: MandateFacts,
    history: MandateHistory,
    at: datetime,
) -> MandateResult:
    """Look for a fault in OUR setup, in descending order of certainty.

    Arithmetic first - an amount above a cap is not a matter of opinion - then
    calendar facts, then the inferential one. A weaker finding must never
    pre-empt a stronger one, because the label decides which repair runs.
    """
    ensure_utc(at)

    if facts.cap_paise is not None and int(amount_paise) > int(facts.cap_paise):
        return MandateResult(
            CauseLabel.MANDATE_CAP_EXCEEDED,
            CONFIDENCE_CAP_EXCEEDED,
            "the debit is larger than the cap the mandate was registered with",
        )

    if facts.expires_at is not None and facts.expires_at <= at:
        return MandateResult(
            CauseLabel.MANDATE_EXPIRED,
            CONFIDENCE_EXPIRED,
            "the mandate expired before this presentation",
        )

    orphaned = _orphaned(facts, history)
    if orphaned.found:
        return orphaned

    if _predebit_notice_missing(facts, at):
        return MandateResult(
            CauseLabel.PREDEBIT_NOTICE_MISSING,
            CONFIDENCE_PREDEBIT,
            f"no pre-debit notification at least {PREDEBIT_NOTICE_HOURS}h before presentation",
        )

    if _wrong_debit_date(facts, at):
        return MandateResult(
            CauseLabel.WRONG_DEBIT_DATE,
            CONFIDENCE_WRONG_DATE,
            "presented on a day the mandate was not registered for",
        )

    return MandateResult()


def _orphaned(facts: MandateFacts, history: MandateHistory) -> MandateResult:
    """A mandate registered against a credential that no longer resolves.

    The evidence is three facts in combination, none of which is conclusive
    alone: the instrument was reissued after the mandate was registered, every
    presentation since has failed with no success among them, and our own
    records still say the mandate is active.

    That last condition is the one that makes it silent. If the records said
    `orphaned` there would be nothing to detect and no cohort to find.
    """
    reissued = facts.instrument_reissued_at
    registered = facts.registered_at
    if reissued is None or registered is None or reissued <= registered:
        return MandateResult()

    attempts = history.since(facts.mandate_ref, reissued)
    if len(attempts) < ORPHAN_FAILURE_RUN:
        return MandateResult()
    if any(attempt.succeeded for attempt in attempts):
        return MandateResult()

    return MandateResult(
        CauseLabel.MANDATE_ORPHANED,
        CONFIDENCE_ORPHANED,
        f"instrument reissued after registration and {len(attempts)} presentations "
        f"since have all failed, while our records still say {facts.status!r}",
    )


def _predebit_notice_missing(facts: MandateFacts, at: datetime) -> bool:
    """Only answerable where notices are tracked. Otherwise this check abstains."""
    if not facts.predebit_required:
        return False
    if facts.predebit_notice_at is None:
        return True
    return at - facts.predebit_notice_at < timedelta(hours=PREDEBIT_NOTICE_HOURS)


def _wrong_debit_date(facts: MandateFacts, at: datetime) -> bool:
    if facts.scheduled_debit_day is None:
        return False
    return at.day != facts.scheduled_debit_day
