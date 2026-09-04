"""Durable run lifecycle: the row, the subscriptions, and how a run ends.

SHARED, NOT INNGEST-OWNED. `durable_runs` is read and written from outside the
durable-function package: the erasure sweep cancels every live run for a
subject, the console explains why a run stopped, and an operator freezing the
system needs to know what is asleep. Keeping the table here and the STEP
SEMANTICS in `inngest_fns/runtime.py` is the split that matters - this module
is state, that one is execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from arc.core.time_authority import ensure_utc
from arc.events.bus import Subscription
from arc.events.names import CANCEL_ON, EventName, match_value


@dataclass
class DurableRun:
    """One function invocation, and the steps it has already completed."""

    run_id: UUID
    function_id: str
    claim_id: UUID | None
    subject_token: str | None
    tenant_id: str
    started_at: datetime
    cancel_on: frozenset[EventName] = field(default_factory=lambda: CANCEL_ON)

    def subscriptions(self) -> list[Subscription]:
        subscriptions: list[Subscription] = []
        for event in sorted(self.cancel_on):
            try:
                key = match_value(
                    event,
                    claim_id=str(self.claim_id) if self.claim_id else None,
                    subject_token=self.subject_token,
                    tenant_id=self.tenant_id,
                )
            except ValueError:
                # A run with no claim cannot be cancelled by a claim-keyed
                # event. Skipping is correct and silent; raising would stop a
                # subject-level function from existing at all.
                continue
            subscriptions.append(Subscription(event=event, match=key))
        return subscriptions


async def start_run(
    conn: Any,
    *,
    function_id: str,
    at: datetime,
    claim_id: UUID | None = None,
    subject_token: str | None = None,
    tenant_id: str = "default",
    cancel_on: frozenset[EventName] | None = None,
    run_id: UUID | None = None,
) -> DurableRun:
    ensure_utc(at)
    identifier = run_id or uuid4()
    await conn.execute(
        """
        INSERT INTO durable_runs
            (run_id, function_id, claim_id, subject_token, tenant_id, status, started_at)
        VALUES ($1, $2, $3, $4, $5, 'running', $6)
        ON CONFLICT (run_id) DO NOTHING
        """,
        identifier,
        function_id,
        claim_id,
        subject_token,
        tenant_id,
        at,
    )
    return DurableRun(
        run_id=identifier,
        function_id=function_id,
        claim_id=claim_id,
        subject_token=subject_token,
        tenant_id=tenant_id,
        started_at=at,
        cancel_on=cancel_on if cancel_on is not None else CANCEL_ON,
    )


async def finish_run(conn: Any, run: DurableRun, *, at: datetime, outcome: str) -> None:
    ensure_utc(at)
    await conn.execute(
        """
        UPDATE durable_runs
           SET status = 'completed', finished_at = $2, outcome = $3
         WHERE run_id = $1 AND status NOT IN ('cancelled', 'failed')
        """,
        run.run_id,
        at,
        outcome,
    )


async def run_status(conn: Any, run_id: UUID) -> dict[str, Any] | None:
    row = await conn.fetchrow(
        "SELECT status, outcome, cancelled_by, finished_at FROM durable_runs WHERE run_id = $1",
        run_id,
    )
    return dict(row) if row is not None else None


async def cancel_runs_for_subject(
    conn: Any, subject_token: str, *, at: datetime, event: EventName
) -> int:
    """Mark every live run for a subject cancelled.

    Used by the erasure orchestration, which must not leave a sleeping run
    holding a claim id belonging to a subject whose data has just been
    destroyed. The event is emitted separately; this is the bookkeeping that
    makes the cancellation visible without waiting for each run to next wake.
    """
    ensure_utc(at)
    return int(
        await conn.fetchval(
            """
            WITH cancelled AS (
                UPDATE durable_runs
                   SET status = 'cancelled', cancelled_by = $3, finished_at = $2,
                       outcome = 'CANCELLED'
                 WHERE subject_token = $1
                   AND status IN ('running', 'sleeping', 'waiting')
             RETURNING run_id
            )
            SELECT count(*) FROM cancelled
            """,
            subject_token,
            at,
            event.value,
        )
        or 0
    )
