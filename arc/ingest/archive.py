"""The raw immutable archive. Written BEFORE the payload is parsed.

WHY before: a parser bug is discovered later, and the only way to recover the
claims it mangled is to replay the original bytes through a fixed parser. If
the archive were written after a successful parse, the deliveries that most
need replaying - the ones that failed - would be the ones not kept.

It records DELIVERIES, not events. A redelivered webhook is archived twice and
deduplicated once, and keeping those two facts in separate tables is what lets
"how often does this gateway redeliver" be answerable at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from arc.core.time_authority import ensure_utc


@dataclass(frozen=True)
class ArchivedDelivery:
    archive_id: int
    payload_hash: bytes


class RawArchive:
    """Append-only storage of exactly what arrived."""

    async def store(
        self,
        conn: Any,
        *,
        source: str,
        raw: bytes,
        signature: str,
        payload_hash: bytes,
        received_at: datetime,
        signature_valid: bool,
    ) -> ArchivedDelivery:
        """Keep the bytes, the signature, and whether it verified.

        An unverified delivery is archived too. Refusing to keep it would
        discard the evidence of an attack, and the whole point of the archive
        is that it holds what arrived rather than what we accepted.
        """
        ensure_utc(received_at)
        archive_id = await conn.fetchval(
            """
            INSERT INTO raw_events
                (source, payload_hash, signature, body, received_at, signature_valid)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING archive_id
            """,
            source,
            payload_hash,
            signature,
            raw,
            received_at,
            signature_valid,
        )
        return ArchivedDelivery(archive_id=int(archive_id), payload_hash=payload_hash)

    async def annotate(
        self,
        conn: Any,
        archive_id: int,
        *,
        event_id: str | None = None,
        event_timestamp: datetime | None = None,
        parse_error: str | None = None,
    ) -> None:
        """Link a delivery to what it turned out to be, after the parse.

        Nullable on purpose: a delivery that never parsed still has a row, and
        `parse_error` is how a breaker later counts how badly one source is
        behaving.
        """
        if event_timestamp is not None:
            ensure_utc(event_timestamp)
        await conn.execute(
            """
            UPDATE raw_events
               SET event_id = $2, event_timestamp = $3, parse_error = $4
             WHERE archive_id = $1
            """,
            archive_id,
            event_id,
            event_timestamp,
            parse_error,
        )

    async def fetch(self, conn: Any, archive_id: int) -> bytes | None:
        return await conn.fetchval("SELECT body FROM raw_events WHERE archive_id = $1", archive_id)

    async def replay_source(self, conn: Any, source: str) -> list[bytes]:
        """Every delivery from one source, in arrival order, for a re-parse."""
        rows = await conn.fetch(
            """
            SELECT body FROM raw_events
             WHERE source = $1 AND signature_valid
             ORDER BY archive_id
            """,
            source,
        )
        return [row["body"] for row in rows]
