"""The event log: publish, and ask what has happened since.

Shared infrastructure, not a durable-function detail. `durable_events` is where
a hardship signal, a recovery, an erasure and a system freeze all land, and the
things that WRITE them are spread across the system - the money ledger, the
Sentinel, the Conductor's erasure sweep, an operator hitting the kill switch.
Only the reading side is concentrated in `inngest_fns`, and owning a table from
its biggest reader is what created the import cycle this module removes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from arc.core.time_authority import ensure_utc
from arc.events.names import EventName, match_value


@dataclass(frozen=True)
class Subscription:
    """One cancelOn entry, resolved to the value it compares against."""

    event: EventName
    match: str


@dataclass(frozen=True)
class ObservedEvent:
    name: EventName
    match_key: str
    occurred_at: datetime
    payload: Mapping[str, Any]


async def emit(
    conn: Any,
    event: EventName,
    *,
    at: datetime,
    claim_id: UUID | None = None,
    subject_token: str | None = None,
    tenant_id: str = "default",
    payload: Mapping[str, Any] | None = None,
) -> int:
    """Publish one event. The only way anything cancels a run."""
    ensure_utc(at)
    key = match_value(
        event,
        claim_id=str(claim_id) if claim_id else None,
        subject_token=subject_token,
        tenant_id=tenant_id,
    )
    return await conn.fetchval(
        """
        INSERT INTO durable_events (name, match_key, occurred_at, payload)
        VALUES ($1, $2, $3, $4::jsonb)
        RETURNING event_seq
        """,
        event.value,
        key,
        at,
        json.dumps(dict(payload or {})),
    )


async def events_since(
    conn: Any,
    subscriptions: Sequence[Subscription],
    *,
    since: datetime,
) -> ObservedEvent | None:
    """The first matching event at or after `since`, if any.

    Half-open from `since` INCLUSIVE, because a cancellation that lands at the
    exact instant a run started must still cancel it. Excluding the boundary
    here would create a one-tick window in which a hardship signal is dropped.
    """
    if not subscriptions:
        return None
    ensure_utc(since)
    names = [s.event.value for s in subscriptions]
    keys = [s.match for s in subscriptions]
    row = await conn.fetchrow(
        """
        SELECT name, match_key, occurred_at, payload
          FROM durable_events
         WHERE occurred_at >= $3
           AND (name, match_key) IN (
                SELECT * FROM unnest($1::text[], $2::text[])
           )
      ORDER BY occurred_at, event_seq
         LIMIT 1
        """,
        names,
        keys,
        since,
    )
    if row is None:
        return None
    return ObservedEvent(
        name=EventName(row["name"]),
        match_key=row["match_key"],
        occurred_at=row["occurred_at"],
        payload=json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"],
    )
