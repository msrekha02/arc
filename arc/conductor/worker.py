"""Dispatch: Gate touchpoint 3, the channel call, and the outcome.

GATE TOUCHPOINT 3 IS THE FIRST THING THAT HAPPENS, before any provider is
reached. If the certificate window has closed, the row is cancelled, an
`ABANDONED_UNEXECUTED` entry is appended, the budget is released, and the claim
goes back to the Allocator for a fresh decision.

WHY THAT BRANCH EXISTS AND WHY IT MUST NEVER EXECUTE ANYWAY. It is the single
line that keeps M11's propensity log honest. The Allocator sampled an action
and wrote down the probability it sampled with, for a specific planned moment.
Executing that decision an hour later means the logged propensity describes a
world that no longer holds - the contact window may have closed, the subject
may have been contacted by another cycle, the certificate's temporal rules were
evaluated against a time that has passed. The number in the log would then be
attached to an action that could not legitimately have been taken, and every
downstream estimate inherits the lie. Re-deciding costs one cycle. Executing
stale authorisation costs the measurement.

"It is only four minutes past" is exactly the reasoning the window exists to
refuse, so there is no grace parameter here and no `force` argument.

THE WORKER IS DELIBERATELY DUMB ABOUT POLICY. It does not choose actions, does
not escalate, does not decide what to do after a failure beyond retry-or-die.
Escalation authority belongs to the Allocator (L4), and a worker that
second-guessed it would be a second policy nobody is testing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from arc.conductor import reservations
from arc.conductor.fsm import ClaimNotFound, current_state
from arc.conductor.outbox import (
    MAX_ATTEMPTS,
    OutboxRow,
    OutboxStatus,
    backoff,
    claim_batch,
    mark,
    reschedule,
)
from arc.core.time_authority import ensure_utc
from arc.core.types import ABSORBING_STATES, ClaimState
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType


class RetryableError(Exception):
    """The provider might succeed later. The row goes back with a backoff."""


class PermanentError(Exception):
    """The provider will never succeed. The row is dead and goes to the DLQ."""


class DispatchOutcome(StrEnum):
    SENT = "sent"
    CERT_EXPIRED = "cert_expired"
    TERMINAL_STATE = "terminal_state"
    RESCHEDULED = "rescheduled"
    DEAD = "dead"


@dataclass(frozen=True)
class DispatchResult:
    row_id: int
    outcome: DispatchOutcome
    requeued: bool = False
    detail: str = ""


@runtime_checkable
class ChannelLike(Protocol):
    """The effector, as the Conductor sees it.

    Narrow on purpose: a channel receives a payload and an idempotency key and
    returns what happened. It gets no claim, no cause and no state, because a
    channel that could see those could branch on them, and policy in the
    effector layer is policy that cannot be audited.
    """

    async def send(self, payload: Mapping[str, Any], idempotency_key: str) -> Any: ...


async def requeue_for_allocation(
    conn: Any,
    claim_id: UUID,
    *,
    reason: str,
    at: datetime,
    ledger: DecisionLedger | None = None,
    subject_token: str | None = None,
) -> None:
    """Send a claim back to L4 for a genuinely fresh decision.

    IN_TREATMENT does not return to PLANNED - that edge does not exist in M1's
    table and inventing one here would be a second state machine. The claim is
    marked as needing allocation through the ledger, and the next cycle picks
    it up with a new cycle id, a new certificate, and a new propensity.
    """
    ledger = ledger or DecisionLedger()
    await ledger.append(
        conn,
        LedgerEntry(
            event_type=LedgerEventType.ABANDONED_UNEXECUTED,
            occurred_at=at,
            claim_id=claim_id,
            subject_token=subject_token,
            payload={"reason": reason, "requeued_for_allocation": True},
        ),
    )


async def dispatch(
    conn: Any,
    row: OutboxRow,
    channels: Mapping[str, ChannelLike],
    *,
    at: datetime,
    ledger: DecisionLedger | None = None,
) -> DispatchResult:
    """Send one row, or refuse to.

    The order of the guards is the order of their authority: a closed
    certificate window outranks everything, a terminal claim outranks the
    provider, and only then does anything leave the building.
    """
    ensure_utc(at)
    ledger = ledger or DecisionLedger()

    # ---- GATE TOUCHPOINT 3 -------------------------------------------------
    if not row.certificate_covers(at):
        await mark(conn, row.id, OutboxStatus.CANCELLED, error="CERT_EXPIRED")
        await reservations.release(conn, row.idempotency_key, at)
        await requeue_for_allocation(
            conn,
            row.claim_id,
            reason="CERT_EXPIRED",
            at=at,
            ledger=ledger,
            subject_token=row.subject_token,
        )
        return DispatchResult(
            row_id=row.id,
            outcome=DispatchOutcome.CERT_EXPIRED,
            requeued=True,
            detail=(
                f"certificate valid [{row.cert_valid_from.isoformat()}, "
                f"{row.cert_valid_until.isoformat()}], dispatch at {at.isoformat()}"
            ),
        )

    # ---- the claim may have moved while this row waited --------------------
    try:
        state = await current_state(conn, row.claim_id)
    except ClaimNotFound:
        state = None
    if state is not None and state in ABSORBING_STATES:
        await mark(conn, row.id, OutboxStatus.CANCELLED, error=f"TERMINAL_{state.value.upper()}")
        await reservations.release(conn, row.idempotency_key, at)
        return DispatchResult(
            row_id=row.id,
            outcome=DispatchOutcome.TERMINAL_STATE,
            detail=f"claim is {state}, which is absorbing",
        )

    channel = channels.get(row.channel)
    if channel is None:
        await mark(conn, row.id, OutboxStatus.DEAD, error=f"NO_CHANNEL:{row.channel}")
        await reservations.release(conn, row.idempotency_key, at)
        return DispatchResult(row_id=row.id, outcome=DispatchOutcome.DEAD, detail="no such channel")

    try:
        result = await channel.send(row.payload, row.idempotency_key)
    except RetryableError as exc:
        if row.attempts >= MAX_ATTEMPTS:
            await mark(conn, row.id, OutboxStatus.DEAD, error=f"RETRIES_EXHAUSTED:{exc}")
            await reservations.release(conn, row.idempotency_key, at)
            return DispatchResult(row_id=row.id, outcome=DispatchOutcome.DEAD, detail=str(exc))
        await reschedule(conn, row.id, not_before=at + backoff(row.attempts), error=str(exc))
        return DispatchResult(row_id=row.id, outcome=DispatchOutcome.RESCHEDULED, detail=str(exc))
    except PermanentError as exc:
        await mark(conn, row.id, OutboxStatus.DEAD, error=str(exc))
        await reservations.release(conn, row.idempotency_key, at)
        return DispatchResult(row_id=row.id, outcome=DispatchOutcome.DEAD, detail=str(exc))

    await mark(conn, row.id, OutboxStatus.SENT)
    # The budget was genuinely spent, so it is consumed rather than returned.
    await reservations.consume(conn, row.idempotency_key, at)
    await ledger.append(
        conn,
        LedgerEntry(
            event_type=LedgerEventType.CHANNEL_DISPATCHED,
            occurred_at=at,
            claim_id=row.claim_id,
            subject_token=row.subject_token,
            payload={
                "channel": row.channel,
                "action_type": row.action_type.value,
                "idempotency_key": row.idempotency_key,
                "certificate_id": str(row.certificate_id),
                "attempts": row.attempts,
                "outcome": str(getattr(result, "outcome", "delivered")),
            },
        ),
    )
    return DispatchResult(row_id=row.id, outcome=DispatchOutcome.SENT)


async def run_worker(
    pool_acquire: Any,
    worker_id: str,
    channels: Mapping[str, ChannelLike],
    *,
    clock: Any,
    batch: int = 10,
    lease: timedelta = timedelta(minutes=2),
    max_idle_polls: int = 3,
    poll_interval: float = 0.01,
) -> list[DispatchResult]:
    """Claim, dispatch, repeat until the queue is empty.

    Each claimed batch is dispatched on its own connection. `pool_acquire` is
    an async context manager factory so a worker holds exactly one connection
    at a time, which is what makes twenty of them twenty real Postgres sessions
    rather than twenty coroutines sharing one.
    """
    results: list[DispatchResult] = []
    idle = 0

    while idle < max_idle_polls:
        async with pool_acquire() as conn:
            rows = await claim_batch(conn, worker_id, batch, at=clock.now(), lease=lease)
            if not rows:
                idle += 1
            else:
                idle = 0
                for row in rows:
                    results.append(await dispatch(conn, row, channels, at=clock.now()))
        if idle:
            await asyncio.sleep(poll_interval)

    return results


async def release_on_terminal(conn: Any, claim_id: UUID, *, at: datetime, state: ClaimState) -> int:
    """Free every live reservation a claim still holds once it is finished.

    Called when a claim reaches a terminal state by any route - recovered,
    written off, forborne. A reservation outliving its claim is the quiet
    starvation bug: nothing errors, the cycle simply has less budget than it
    thinks, and the cause is invisible in every dashboard.
    """
    ensure_utc(at)
    if state not in ABSORBING_STATES and state is not ClaimState.RECOVERED:
        return 0
    live = await reservations.live_for(conn, claim_id)
    freed = 0
    for reservation in live:
        freed += await reservations.release(conn, reservation.idempotency_key, at)
    return freed
