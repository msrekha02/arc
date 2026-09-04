"""Promise-to-pay tracker: wait for the payment, record what happened, requeue.

WHAT THIS FUNCTION DOES NOT DO, AND WHY THAT IS THE WHOLE DESIGN.

    It does not choose the next action after a broken promise.

Escalation authority belongs to the Allocator. A broken promise is not an
instruction to escalate; it is a FEATURE, and the Allocator re-scores the claim
with it in the vector alongside everything else it knows - the subject's other
claims, the contact budget, the shadow prices, whether contact for this person
carries negative uplift. A tracker that escalated would be making a portfolio
decision from inside a single claim's run, with no view of the budget it was
spending and no propensity attached to what it chose.

    THE OFF-POLICY CONSEQUENCE IS THE SHARP ONE. An action chosen here has no
    logged propensity, because no distribution was sampled. It would appear in
    the logs as an action that happened with probability nothing, and every
    importance ratio touching it would be a division by zero. One escalation
    decided in the wrong place makes the batch it belongs to unmeasurable.

So this function records KEPT or BROKEN and hands the claim back. That is all.

CENSORING IS NOT BREAKAGE. A promise whose date has not arrived is UNRESOLVED,
which is a third answer and not a soft BROKEN. Coding it broken is what biases
a promise-to-pay model pessimistic, and M7's Model C is fitted on exactly these
records.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from arc.conductor.worker import requeue_for_allocation
from arc.core.money import Paise, paise
from arc.core.time_authority import ensure_utc
from arc.events.names import (
    CANCEL_ON,
    EventName,
    assert_cancels_on_every_stop,
)
from arc.events.runs import DurableRun, finish_run, start_run
from arc.inngest_fns.runtime import Clock, Step
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType

FUNCTION_ID = "ptp-tracker"

assert_cancels_on_every_stop(FUNCTION_ID, CANCEL_ON)

# How long after the promised date a payment still counts as keeping it.
# source: settlement lag on Indian retail rails, where a payment initiated on
# the promised day can land the following working day through no fault of the
# person who made the promise.
DEFAULT_GRACE = timedelta(hours=36)


class PromiseOutcome(StrEnum):
    """Three answers, not two. UNRESOLVED is censored, never coerced."""

    KEPT = "kept"
    BROKEN = "broken"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class Promise:
    claim_id: UUID
    subject_token: str
    promise_date: datetime
    amount_paise: Paise
    grace: timedelta = DEFAULT_GRACE
    tenant_id: str = "default"

    def __post_init__(self) -> None:
        ensure_utc(self.promise_date)

    @property
    def deadline(self) -> datetime:
        return self.promise_date + self.grace


@dataclass(frozen=True)
class PromiseResult:
    run: DurableRun
    outcome: PromiseOutcome
    paid_paise: Paise
    requeued: bool
    # Deliberately absent: any field naming a next action. There is nowhere in
    # this type to put one, which is the structural half of the rule the
    # docstring states.


async def promise_to_pay_tracker(
    conn: Any,
    promise: Promise,
    *,
    clock: Clock,
    ledger: DecisionLedger | None = None,
    step: Step | None = None,
) -> PromiseResult:
    """Wait for payment until the deadline, then record and hand back."""
    ledger = ledger or DecisionLedger()
    run = (
        step.run
        if step is not None
        else await start_run(
            conn,
            function_id=FUNCTION_ID,
            at=clock.now(),
            claim_id=promise.claim_id,
            subject_token=promise.subject_token,
            tenant_id=promise.tenant_id,
        )
    )
    step = step or Step(conn, run, clock)

    paid = await step.wait_for_event(
        "await-payment",
        EventName.PAYMENT_RECEIVED,
        match=promise.subject_token,
        timeout_at=promise.deadline,
    )

    if paid is not None:
        amount = paise(int(paid.payload.get("amount_paise", int(promise.amount_paise))))
        await step.run_step(
            "close-kept",
            lambda: _record(
                conn,
                promise,
                outcome=PromiseOutcome.KEPT,
                amount=amount,
                at=paid.occurred_at,
                ledger=ledger,
            ),
        )
        await finish_run(conn, run, at=step.now(), outcome=PromiseOutcome.KEPT.value)
        return PromiseResult(
            run=run, outcome=PromiseOutcome.KEPT, paid_paise=amount, requeued=False
        )

    # The deadline passed with no payment. BROKEN - and only because the
    # deadline actually passed. A promise still in the future would never have
    # reached this line; `wait_for_event` would still be waiting.
    await step.run_step(
        "close-broken",
        lambda: _record(
            conn,
            promise,
            outcome=PromiseOutcome.BROKEN,
            amount=paise(0),
            at=step.now(),
            ledger=ledger,
        ),
    )

    # Back to the Allocator. NOT to an escalation decided here.
    await step.run_step(
        "requeue-broken",
        lambda: _requeue(conn, promise, at=step.now(), ledger=ledger),
    )
    await finish_run(conn, run, at=step.now(), outcome=PromiseOutcome.BROKEN.value)
    return PromiseResult(run=run, outcome=PromiseOutcome.BROKEN, paid_paise=paise(0), requeued=True)


def classify(promise: Promise, *, paid_at: datetime | None, at: datetime) -> PromiseOutcome:
    """The three-way coding, as a pure function so it can be tested alone.

    UNRESOLVED IS RETURNED WHENEVER THE DEADLINE IS STILL AHEAD, regardless of
    how unlikely payment looks. A promise dated the twentieth is neither kept
    nor broken on the eighteenth, and a model trained on data that says
    otherwise learns that promises are broken more often than they are.
    """
    ensure_utc(at)
    if paid_at is not None:
        ensure_utc(paid_at)
        return PromiseOutcome.KEPT if paid_at <= promise.deadline else PromiseOutcome.BROKEN
    if at <= promise.deadline:
        return PromiseOutcome.UNRESOLVED
    return PromiseOutcome.BROKEN


async def _record(
    conn: Any,
    promise: Promise,
    *,
    outcome: PromiseOutcome,
    amount: Paise,
    at: datetime,
    ledger: DecisionLedger,
) -> dict[str, Any]:
    await ledger.append(
        conn,
        LedgerEntry(
            event_type=LedgerEventType.PROMISE_OUTCOME,
            occurred_at=at,
            claim_id=promise.claim_id,
            subject_token=promise.subject_token,
            payload={
                "outcome": outcome.value,
                "promise_date": promise.promise_date,
                "deadline": promise.deadline,
                "amount_paise": int(amount),
                "promised_paise": int(promise.amount_paise),
            },
        ),
    )
    return {"outcome": outcome.value, "amount_paise": int(amount)}


async def _requeue(
    conn: Any, promise: Promise, *, at: datetime, ledger: DecisionLedger
) -> dict[str, Any]:
    """Hand the claim back with the broken promise recorded as a REASON.

    A reason, not an instruction. The Allocator reads it as one feature among
    many and may well decide the next action is `do_nothing`.
    """
    await requeue_for_allocation(
        conn,
        promise.claim_id,
        reason="PTP_BROKEN",
        at=at,
        ledger=ledger,
        subject_token=promise.subject_token,
    )
    return {"reason": "PTP_BROKEN", "prior_promise_outcome": PromiseOutcome.BROKEN.value}
