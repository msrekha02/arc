"""Salary-aligned retry: sleep to payday, re-gate, enqueue.

The prevention thesis in one function. A debit presented the day before a
salary credit fails; the same debit presented the day after succeeds. Nothing
about the customer changed, so the whole difference is timing, and timing is
free.

WHAT THIS FUNCTION IS ALLOWED TO DECIDE: when to wake, and whether to hop a
deferral. That is all. It does not choose an action - the action was chosen by
the Allocator and is carried in the event. It does not escalate. It does not
touch a channel. On BLOCK it stops; on three deferrals it hands the claim back.

THE DEFER LOOP IS BOUNDED AT THREE AND THAT NUMBER IS LOAD-BEARING. Each hop
re-uses a decision the Allocator made for a different moment. One or two hops
is a cooldown expiring, which is what deferral is for. A fourth would mean the
world has moved far enough that the decision itself is stale, so the claim
returns to the Allocator and is re-scored with a fresh propensity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from arc.conductor.worker import requeue_for_allocation
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType
from arc.events.names import CANCEL_ON, assert_cancels_on_every_stop
from arc.events.runs import DurableRun, finish_run, start_run
from arc.inngest_fns.gated_enqueue import (
    MAX_DEFER_HOPS,
    CertifyingGate,
    ContextSource,
    EnqueueResult,
    Outcome,
    gated_enqueue,
)
from arc.inngest_fns.runtime import Clock, Step
from arc.ledger.decision_ledger import DecisionLedger

FUNCTION_ID = "salary-aligned-retry"

# Checked at import, not at call. A function whose cancellation set is
# incomplete must not be constructible, because the gap only shows up when the
# signal it omitted arrives.
assert_cancels_on_every_stop(FUNCTION_ID, CANCEL_ON)


@dataclass(frozen=True)
class RetryRequest:
    claim_id: UUID
    subject_token: str
    cycle_id: UUID
    action: ActionType
    planned_at: datetime
    tenant_id: str = "default"
    reservation_key: str | None = None

    def __post_init__(self) -> None:
        ensure_utc(self.planned_at)


@dataclass(frozen=True)
class RetryOutcome:
    run: DurableRun
    result: EnqueueResult
    hops: int
    requeued: bool


async def salary_aligned_retry(
    conn: Any,
    request: RetryRequest,
    *,
    gate: CertifyingGate,
    contexts: ContextSource,
    clock: Clock,
    ledger: DecisionLedger | None = None,
    step: Step | None = None,
) -> RetryOutcome:
    """Sleep until payday, then re-gate. Cancellable throughout.

    The sleep is where a hardship signal kills this run. Nothing in here polls
    for one: `sleep_until` checks the subscriptions on both sides of the sleep,
    so a signal that lands in the middle stops the run at the instant it wakes.
    """
    ledger = ledger or DecisionLedger()
    run = (
        step.run
        if step is not None
        else await start_run(
            conn,
            function_id=FUNCTION_ID,
            at=clock.now(),
            claim_id=request.claim_id,
            subject_token=request.subject_token,
            tenant_id=request.tenant_id,
        )
    )
    step = step or Step(conn, run, clock)

    await step.sleep_until("wait-for-payday", request.planned_at)

    result = await gated_enqueue(
        step,
        conn,
        claim_id=request.claim_id,
        subject_token=request.subject_token,
        cycle_id=request.cycle_id,
        action=request.action,
        gate=gate,
        contexts=contexts,
        at=step.now(),
        ledger=ledger,
        reservation_key=request.reservation_key,
    )

    hops = 0
    while result.is_retryable and hops < MAX_DEFER_HOPS:
        # `is_retryable` is DEFER-and-has-a-timestamp. A BLOCK never enters
        # this loop, which is the point: there is nothing to sleep until.
        assert result.until is not None
        await step.sleep_until(f"defer-{hops}", result.until)
        hops += 1
        result = await gated_enqueue(
            step,
            conn,
            claim_id=request.claim_id,
            subject_token=request.subject_token,
            cycle_id=request.cycle_id,
            action=request.action,
            gate=gate,
            contexts=contexts,
            at=step.now(),
            ledger=ledger,
            reservation_key=request.reservation_key,
        )

    requeued = False
    if result.outcome is Outcome.DEFER:
        # Deferrals exhausted. Back to the Allocator for a fresh decision with
        # a fresh propensity, rather than a fourth hop on a stale one.
        await step.run_step(
            "requeue-defer-exhausted",
            lambda: _requeue(
                conn,
                request,
                reason="DEFER_BUDGET_EXHAUSTED",
                at=step.now(),
                ledger=ledger,
            ),
        )
        requeued = True
        result = EnqueueResult(outcome=Outcome.REQUEUED, until=result.until)
    elif result.outcome in (Outcome.BLOCKED, Outcome.BUDGET_GONE):
        # A BLOCK terminates the treatment path. The claim still returns to the
        # Allocator, because a blocked claim is not a resolved one - it is one
        # this path cannot serve, and the next cycle re-scores it.
        await step.run_step(
            "requeue-blocked",
            lambda: _requeue(
                conn,
                request,
                reason=f"GATE_{result.outcome.value}",
                at=step.now(),
                ledger=ledger,
            ),
        )
        requeued = True

    await finish_run(conn, run, at=step.now(), outcome=result.outcome.value)
    return RetryOutcome(run=run, result=result, hops=hops, requeued=requeued)


async def _requeue(
    conn: Any,
    request: RetryRequest,
    *,
    reason: str,
    at: datetime,
    ledger: DecisionLedger,
) -> dict[str, Any]:
    await requeue_for_allocation(
        conn,
        request.claim_id,
        reason=reason,
        at=at,
        ledger=ledger,
        subject_token=request.subject_token,
    )
    return {"reason": reason}
