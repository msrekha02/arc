"""Dedupe on `(source, event_id)` over a rolling 30-day window.

Gateways redeliver on acknowledgement timeout. Without this, one webhook
becomes two claims, the same failure is diagnosed twice, and the contact budget
is spent twice on one person.

The window is 30 days because unbounded dedupe state grows forever and a
redelivery a month later is a different operational event, not a duplicate.
An event whose only record has aged out is treated as new and its record is
refreshed - deliberately, and stated here so nobody later reads it as a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from arc.core.time_authority import ensure_utc

DEDUPE_WINDOW = timedelta(days=30)


@dataclass(frozen=True)
class DedupeVerdict:
    """Whether to process this delivery, and why not if not."""

    is_new: bool
    source: str
    event_id: str


class Dedupe:
    """One row per `(source, event_id)`, claimed atomically."""

    def __init__(self, window: timedelta = DEDUPE_WINDOW) -> None:
        if window <= timedelta(0):
            raise ValueError("the dedupe window must be positive")
        self._window = window

    async def claim(self, conn: Any, source: str, event_id: str, at: datetime) -> DedupeVerdict:
        """Try to claim an event id. A returned row means nobody had it.

        The insert IS the check. A check-then-insert races two workers into
        both believing they are first, which is the same class of bug the
        outbox reservations avoid at M9.
        """
        ensure_utc(at)
        row = await conn.fetchrow(
            """
            INSERT INTO ingest_dedupe (source, event_id, first_seen_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (source, event_id) DO UPDATE
               SET first_seen_at = EXCLUDED.first_seen_at
             WHERE ingest_dedupe.first_seen_at < $4
            RETURNING event_id
            """,
            source,
            event_id,
            at,
            at - self._window,
        )
        return DedupeVerdict(is_new=row is not None, source=source, event_id=event_id)

    async def purge(self, conn: Any, at: datetime) -> int:
        """Drop records that have aged out of the window."""
        ensure_utc(at)
        result = await conn.execute(
            "DELETE FROM ingest_dedupe WHERE first_seen_at < $1", at - self._window
        )
        return int(str(result).rsplit(" ", 1)[-1])
