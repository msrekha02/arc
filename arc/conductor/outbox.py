"""The transactional outbox: the idempotency key, enqueue, claim, and reap.

WHY AN OUTBOX AND NOT A BROKER. The decision, the budget reservation, the
ledger append and the intent to dispatch have to commit together or not at all.
A broker cannot enrol in a Postgres transaction, so publishing after the commit
reopens exactly the window this table closes: the process can die between the
two and the effect is lost, or it can publish and then fail to commit and the
effect happens twice. The transaction IS the guarantee.

WHY `FOR UPDATE SKIP LOCKED`. N workers poll one table. Each skips rows another
worker already holds rather than queueing behind them, so throughput scales
with workers instead of serialising on the head of the queue. No advisory
locks, no worker-id partitioning, no Redis, no second datastore to keep
consistent with the first.

THE LEASE IS THE SAFETY NET, NOT THE MECHANISM. `SKIP LOCKED` prevents two
workers claiming a row concurrently; the lease is what recovers a row whose
worker died holding it. Its duration must comfortably exceed the channel
timeout, because a lease that expires while the first worker is still waiting
on the provider hands the same row to a second worker, and at that point the
only thing standing between the customer and two messages is the idempotency
key being honoured on the far side.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType

# The lease must outlast the slowest channel call by a wide margin. A voice
# provider can hold a connection for tens of seconds; two minutes leaves room
# for a retry inside the provider without a second worker being handed the row.
DEFAULT_LEASE = timedelta(minutes=2)

# Exponential, capped. The cap matters more than the base: an unbounded
# backoff on a permanently failing channel turns a dead row into one that
# retries once a day forever.
MAX_BACKOFF = timedelta(minutes=15)
MAX_ATTEMPTS = 6


class OutboxStatus(StrEnum):
    PENDING = "pending"
    IN_FLIGHT = "in_flight"
    SENT = "sent"
    FAILED = "failed"
    DEAD = "dead"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[OutboxStatus] = frozenset(
    {OutboxStatus.SENT, OutboxStatus.DEAD, OutboxStatus.CANCELLED}
)


def idempotency_key(
    claim_id: UUID,
    action_type: ActionType,
    cycle_id: UUID,
    certificate_id: UUID,
) -> str:
    """sha256(claim_id:action_type:cycle_id:certificate_id).

    THE ATTEMPT COUNTER IS ABSENT AND MUST STAY ABSENT. A dispatch retry of the
    same decision reuses this key, so the provider recognises it and does not
    charge anybody twice. A genuine re-decision after a wake produces a new
    `cycle_id` and a new certificate, so it gets a new key and is allowed
    through. Folding `attempts` in would make every retry look like a fresh
    instruction, which is the failure mode the key exists to prevent.

    It is stable across retries BY CONSTRUCTION rather than by care: the Gate
    derives `certificate_id` as a UUIDv5 over the evaluation, so re-certifying
    identical inputs yields an identical certificate and therefore an identical
    key. Nothing here has to remember to reuse anything.
    """
    material = ":".join([str(claim_id), action_type.value, str(cycle_id), str(certificate_id)])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class OutboxRow:
    """One dispatch intent, as the table holds it."""

    id: int
    claim_id: UUID
    subject_token: str
    cycle_id: UUID
    action_type: ActionType
    channel: str
    payload: dict[str, Any]
    idempotency_key: str
    certificate_id: UUID
    cert_valid_from: datetime
    cert_valid_until: datetime
    not_before: datetime
    priority: int
    status: OutboxStatus
    attempts: int
    lease_owner: str | None
    lease_expires_at: datetime | None
    last_error: str | None

    @classmethod
    def from_record(cls, record: Any) -> OutboxRow:
        payload = record["payload"]
        return cls(
            id=record["id"],
            claim_id=record["claim_id"],
            subject_token=record["subject_token"],
            cycle_id=record["cycle_id"],
            action_type=ActionType(record["action_type"]),
            channel=record["channel"],
            payload=json.loads(payload) if isinstance(payload, str) else dict(payload),
            idempotency_key=record["idempotency_key"],
            certificate_id=record["certificate_id"],
            cert_valid_from=record["cert_valid_from"],
            cert_valid_until=record["cert_valid_until"],
            not_before=record["not_before"],
            priority=record["priority"],
            status=OutboxStatus(record["status"]),
            attempts=record["attempts"],
            lease_owner=record["lease_owner"],
            lease_expires_at=record["lease_expires_at"],
            last_error=record["last_error"],
        )

    def certificate_covers(self, moment: datetime) -> bool:
        """GATE TOUCHPOINT 3, as a predicate.

        Half-open at neither end: a certificate is valid AT its boundaries, and
        the window was already narrowed to the exact minute by the Gate.
        """
        return self.cert_valid_from <= moment <= self.cert_valid_until


async def enqueue(
    conn: Any,
    *,
    claim_id: UUID,
    subject_token: str,
    cycle_id: UUID,
    action_type: ActionType,
    channel: str,
    payload: dict[str, Any],
    certificate_id: UUID,
    cert_valid_from: datetime,
    cert_valid_until: datetime,
    not_before: datetime,
    priority: int = 0,
) -> int | None:
    """Insert one dispatch intent. Returns the row id, or None if it existed.

    `ON CONFLICT DO NOTHING` makes enqueue itself idempotent, which is what
    lets an Inngest step replay - or a retried commit - run twice without
    double-enqueuing.
    """
    for moment in (cert_valid_from, cert_valid_until, not_before):
        ensure_utc(moment)

    key = idempotency_key(claim_id, action_type, cycle_id, certificate_id)
    return await conn.fetchval(
        """
        INSERT INTO outbox
            (claim_id, subject_token, cycle_id, action_type, channel, payload,
             idempotency_key, certificate_id, cert_valid_from, cert_valid_until,
             not_before, priority)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12)
        ON CONFLICT (idempotency_key) DO NOTHING
        RETURNING id
        """,
        claim_id,
        subject_token,
        cycle_id,
        action_type.value,
        channel,
        json.dumps(payload),
        key,
        certificate_id,
        cert_valid_from,
        cert_valid_until,
        not_before,
        priority,
    )


async def claim_batch(
    conn: Any,
    worker_id: str,
    limit: int,
    *,
    at: datetime,
    lease: timedelta = DEFAULT_LEASE,
) -> list[OutboxRow]:
    """Take up to `limit` ready rows, leased to this worker.

    The CTE locks the rows it selects and the UPDATE marks them in one
    statement, so there is no window between choosing a row and owning it.
    `SKIP LOCKED` is what makes twenty workers useful rather than twenty
    workers taking turns.
    """
    ensure_utc(at)
    records = await conn.fetch(
        """
        WITH claimed AS (
            SELECT id
              FROM outbox
             WHERE status = 'pending'
               AND not_before <= $3
             ORDER BY priority DESC, not_before
             FOR UPDATE SKIP LOCKED
             LIMIT $2
        )
        UPDATE outbox o
           SET status           = 'in_flight',
               lease_owner      = $1,
               lease_expires_at = $3 + $4::interval,
               attempts         = o.attempts + 1
          FROM claimed c
         WHERE o.id = c.id
        RETURNING o.*
        """,
        worker_id,
        limit,
        at,
        lease,
    )
    return [OutboxRow.from_record(record) for record in records]


async def reap_expired_leases(conn: Any, at: datetime) -> int:
    """Return rows whose worker died holding them.

    Without this a crashed worker strands its batch forever: the rows are
    `in_flight`, nothing is flying them, and no query looks at them again.
    """
    ensure_utc(at)
    reclaimed = await conn.fetch(
        """
        UPDATE outbox
           SET status = 'pending', lease_owner = NULL, lease_expires_at = NULL
         WHERE status = 'in_flight'
           AND lease_expires_at < $1
        RETURNING id
        """,
        at,
    )
    return len(reclaimed)


async def mark(
    conn: Any,
    row_id: int,
    status: OutboxStatus,
    *,
    error: str | None = None,
) -> None:
    """Settle a row and drop its lease."""
    await conn.execute(
        """
        UPDATE outbox
           SET status = $2, lease_owner = NULL, lease_expires_at = NULL,
               last_error = COALESCE($3, last_error)
         WHERE id = $1
        """,
        row_id,
        status.value,
        error,
    )


def backoff(attempts: int) -> timedelta:
    """Exponential with a cap. `attempts` is already incremented by the claim."""
    seconds = min(2 ** max(attempts, 1), int(MAX_BACKOFF.total_seconds()))
    return timedelta(seconds=seconds)


async def reschedule(conn: Any, row_id: int, *, not_before: datetime, error: str) -> None:
    """Hand a row back for a later attempt, releasing the lease."""
    ensure_utc(not_before)
    await conn.execute(
        """
        UPDATE outbox
           SET status = 'pending', lease_owner = NULL, lease_expires_at = NULL,
               not_before = $2, last_error = $3
         WHERE id = $1
        """,
        row_id,
        not_before,
        error,
    )


async def fetch_row(conn: Any, row_id: int) -> OutboxRow | None:
    record = await conn.fetchrow("SELECT * FROM outbox WHERE id = $1", row_id)
    return None if record is None else OutboxRow.from_record(record)


async def by_idempotency_key(conn: Any, key: str) -> OutboxRow | None:
    record = await conn.fetchrow("SELECT * FROM outbox WHERE idempotency_key = $1", key)
    return None if record is None else OutboxRow.from_record(record)


async def counts_by_status(conn: Any) -> dict[OutboxStatus, int]:
    records = await conn.fetch("SELECT status, count(*) AS n FROM outbox GROUP BY status")
    return {OutboxStatus(r["status"]): r["n"] for r in records}


async def pending_ids(conn: Any, ids: Sequence[int]) -> list[int]:
    records = await conn.fetch(
        "SELECT id FROM outbox WHERE id = ANY($1::bigint[]) AND status = 'pending'",
        list(ids),
    )
    return [r["id"] for r in records]
