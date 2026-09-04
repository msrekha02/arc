"""The event vocabulary, and the cancellation set that is a stopping rule.

MOVED OUT OF `inngest_fns/` DELIBERATELY. Durable functions are the biggest
consumer of these events but they are not the owner: the erasure orchestration
in the Conductor emits `subject.erasure`, the money ledger emits
`claim.recovered`, and a console reads the log to explain why a run stopped. A
vocabulary owned by one consumer forces every other consumer to import that
consumer, which is how `conductor -> inngest_fns -> conductor` came about.

    THE CYCLE WAS NOT HYPOTHETICAL. It held only because `runtime.py` happened
    to import nothing from the Conductor. The first durable function that
    needed a conductor helper would have closed the loop, and the failure would
    have arrived as an ImportError in whichever module was unlucky enough to be
    imported first.

WHY `cancelOn` IS WHERE THE BEHAVIOURAL STOPS PHYSICALLY LIVE. STOP-HARDSHIP,
STOP-DISPUTE, STOP-ERASURE and STOP-OPTOUT are not checks a function performs.
They are subscriptions. A hardship signal arriving while a retry function is
three days into a sleep kills that run where it lies - nothing polls, nothing
wakes up to discover it should have stopped, and there is no window in which a
run is cancellable in principle and still running in fact.

    THE ALTERNATIVE IS THE BUG. A function that checks for hardship when it
    wakes has already decided to wake, and between the signal and the wake it
    is a scheduled contact to somebody who has told you they are in distress.
    The check would pass and the person would still have been in the queue.

MATCH KEYS DIFFER BY EVENT AND THAT IS THE POINT. `claim.recovered` matches a
claim: one claim is recovered, its siblings are not. `subject.hardship` matches
a SUBJECT, because hardship is a property of a person and must stop every claim
they hold. Matching hardship on a claim would leave the other two running.
"""

from __future__ import annotations

from enum import StrEnum


class MatchOn(StrEnum):
    """Which identifier a subscription compares against."""

    CLAIM = "claim_id"
    SUBJECT = "subject_token"
    TENANT = "tenant_id"


class EventName(StrEnum):
    """Closed set. A typo in an event name is a subscription that never fires."""

    CLAIM_RECOVERED = "claim.recovered"
    CLAIM_DISPUTED = "claim.disputed"
    CLAIM_RETRY_SCHEDULED = "claim.retry_scheduled"
    CLAIM_NEEDS_ALLOCATION = "claim.needs_allocation"

    SUBJECT_HARDSHIP = "subject.hardship"
    SUBJECT_ERASURE = "subject.erasure"
    CONSENT_WITHDRAWN = "consent.withdrawn"

    PAYMENT_RECEIVED = "payment.received"
    PTP_RECORDED = "ptp.recorded"

    SYSTEM_FREEZE = "system.freeze"


# What each event is keyed by. Getting one of these wrong is a silent failure:
# the subscription simply never matches, and the run sleeps on.
MATCH_FIELD: dict[EventName, MatchOn] = {
    EventName.CLAIM_RECOVERED: MatchOn.CLAIM,
    EventName.CLAIM_DISPUTED: MatchOn.CLAIM,
    EventName.CLAIM_RETRY_SCHEDULED: MatchOn.CLAIM,
    EventName.CLAIM_NEEDS_ALLOCATION: MatchOn.CLAIM,
    EventName.SUBJECT_HARDSHIP: MatchOn.SUBJECT,
    EventName.SUBJECT_ERASURE: MatchOn.SUBJECT,
    EventName.CONSENT_WITHDRAWN: MatchOn.SUBJECT,
    EventName.PAYMENT_RECEIVED: MatchOn.SUBJECT,
    EventName.PTP_RECORDED: MatchOn.CLAIM,
    EventName.SYSTEM_FREEZE: MatchOn.TENANT,
}

# The six. Every durable function carries exactly this set.
#
# WHY IT IS ONE CONSTANT AND NOT A PER-FUNCTION LIST: a function that omitted
# one would keep sleeping through the signal it omitted, and the omission would
# be invisible until the signal arrived. `assert_cancels_on_every_stop` is what
# makes adding a function without the full set fail rather than ship.
CANCEL_ON: frozenset[EventName] = frozenset(
    {
        EventName.CLAIM_RECOVERED,
        EventName.CLAIM_DISPUTED,
        EventName.SUBJECT_HARDSHIP,
        EventName.SUBJECT_ERASURE,
        EventName.CONSENT_WITHDRAWN,
        EventName.SYSTEM_FREEZE,
    }
)


class IncompleteCancellation(AssertionError):
    """A durable function subscribes to fewer stops than it must."""


def assert_cancels_on_every_stop(function_id: str, cancel_on: frozenset[EventName]) -> None:
    """A function that sleeps must be cancellable by every behavioural stop."""
    missing = CANCEL_ON - cancel_on
    if missing:
        raise IncompleteCancellation(
            f"{function_id} does not cancel on {sorted(m.value for m in missing)}. "
            "A run that cannot be cancelled by one of these will sleep through it, "
            "and the stopping rule it implements does not exist"
        )


def match_value(
    event: EventName, *, claim_id: str | None, subject_token: str | None, tenant_id: str
) -> str:
    """The value a subscription to `event` compares against."""
    field = MATCH_FIELD[event]
    if field is MatchOn.CLAIM:
        if claim_id is None:
            raise ValueError(f"{event} is keyed by claim and no claim was given")
        return claim_id
    if field is MatchOn.SUBJECT:
        if subject_token is None:
            raise ValueError(f"{event} is keyed by subject and no subject was given")
        return subject_token
    return tenant_id
