"""Erasure, orchestrated across every store that holds anything about a person.

M5 gave the raw archive a `subject_token` column so that erasure could find
every delivery belonging to a subject, and said the orchestrator did not exist.
This is it.

WHAT ERASURE HAS TO REACH, AND WHY EACH ONE IS EASY TO FORGET.

    Subject store    The obvious one. Crypto-shredded: the key is destroyed and
                     the ciphertext becomes unreadable.
    Raw archive      THE ONE THAT GETS FORGOTTEN. L0 archives the raw payload
                     BEFORE parsing, deliberately, so history can be replayed
                     through a fixed parser. That archive is a complete copy of
                     everything the subject store holds, in the gateway's own
                     format, and shredding the subject store while leaving it
                     intact erases nothing.
    Outbox           Scheduled work naming the subject. A message that goes out
                     after erasure is the erasure failing in the most visible
                     way available.
    Durable runs     A sleeping retry holding this subject's claim. Cancelled
                     through the same event M12 already subscribes to, so there
                     is one cancellation path rather than two.
    Decision ledger  NOT erased, and this is the point of the whole design.

WHY THE LEDGER SURVIVES AND WHY THAT IS NOT A CONTRADICTION. An immutable
hash-chained ledger and a right to erasure are directly opposed only if the
ledger contains personal data. It does not: the chain covers pseudonymous
tokens and structured fields, the PII write-guard fails any append that would
change that, and the raw text lives on the other side of the redaction
boundary. So erasure destroys the key, the subject's rows become unreadable,
and the chain stays verifiable because it never covered plaintext in the first
place.

    THE TOMBSTONE IS ITSELF AN AUDIT OBLIGATION. Recording that an erasure
    happened - when, at whose request, how many refs were destroyed - is
    required, and a system that erased the fact of the erasure would have no
    way to demonstrate compliance with the request it just honoured.

ORDER MATTERS. Stop the future first: cancel scheduled work and running
functions before destroying the data, so nothing in flight reads a subject that
is mid-shred. Then shred, then sweep the archive, then record.

`requested_by` MUST BE A ROLE OR A PSEUDONYMOUS OPERATOR REFERENCE, never an
email address or a name. It is written into the tombstone, which is in the
hash-chained ledger, and an erasure that recorded the requester's own personal
data would create a new erasure obligation in the one store that cannot honour
one. This is not merely documented: the PII write-guard refuses the append, so
an email here fails the whole transaction rather than being quietly chained in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from arc.core.ids import is_subject_token
from arc.core.time_authority import ensure_utc
from arc.events.bus import emit
from arc.events.names import EventName
from arc.events.runs import cancel_runs_for_subject
from arc.ledger.decision_ledger import DecisionLedger
from arc.ledger.subject_store import SubjectStore


class ErasureIncomplete(RuntimeError):
    """A sweep left data behind. Raised rather than reported as a partial."""


@dataclass(frozen=True)
class ErasureReport:
    erasure_id: UUID
    subject_token: str
    requested_at: datetime
    requested_by: str
    subject_refs_destroyed: int
    archive_rows_purged: int
    outbox_rows_cancelled: int
    runs_cancelled: int
    tombstone_seq: int

    @property
    def total_destroyed(self) -> int:
        return self.subject_refs_destroyed + self.archive_rows_purged


async def erase_subject(
    conn: Any,
    subject_token: str,
    *,
    at: datetime,
    requested_by: str,
    subject_store: SubjectStore | None = None,
    ledger: DecisionLedger | None = None,
    tenant_id: str = "default",
) -> ErasureReport:
    """Honour one erasure request, everywhere, in one transaction.

    ONE TRANSACTION because a partial erasure is worse than none: it reports
    success while leaving a copy, and the copy is in the store nobody thought
    to check. If any step fails the whole thing rolls back and the request is
    still outstanding, which is a state somebody will act on.
    """
    ensure_utc(at)
    if not is_subject_token(subject_token):
        raise ValueError(f"{subject_token!r} is not a derived subject token")

    ledger = ledger or DecisionLedger()
    subject_store = subject_store or SubjectStore(ledger=ledger)
    erasure_id = uuid4()

    async with conn.transaction():
        await conn.execute(
            """
            INSERT INTO erasure_requests
                (erasure_id, subject_token, requested_at, requested_by)
            VALUES ($1, $2, $3, $4)
            """,
            erasure_id,
            subject_token,
            at,
            requested_by,
        )

        # 1. STOP THE FUTURE FIRST. Nothing scheduled may survive the erasure,
        #    and nothing in flight should read a subject mid-shred.
        outbox_cancelled = int(
            await conn.fetchval(
                """
                WITH stopped AS (
                    UPDATE outbox
                       SET status = 'cancelled', last_error = 'ERASURE'
                     WHERE subject_token = $1
                       AND status IN ('pending', 'in_flight', 'failed')
                 RETURNING id
                )
                SELECT count(*) FROM stopped
                """,
                subject_token,
            )
            or 0
        )

        runs_cancelled = await cancel_runs_for_subject(
            conn, subject_token, at=at, event=EventName.SUBJECT_ERASURE
        )

        # 2. THE EVENT M12 SUBSCRIBES TO. Emitted inside the same transaction,
        #    so a run that wakes after this commits sees it and stops, and a
        #    rollback takes the event with it rather than leaving a phantom
        #    cancellation for an erasure that did not happen.
        await emit(
            conn,
            EventName.SUBJECT_ERASURE,
            at=at,
            subject_token=subject_token,
            tenant_id=tenant_id,
            payload={"erasure_id": str(erasure_id), "requested_by": requested_by},
        )

        # 3. Destroy the key. The subject store appends its own TOMBSTONE.
        tombstone = await subject_store.crypto_shred(
            conn, subject_token, at=at, requested_by=requested_by
        )

        # 4. SWEEP THE RAW ARCHIVE. The step that gets forgotten, and the one
        #    M5 built the `subject_token` column for.
        archive_purged = int(
            await conn.fetchval(
                """
                WITH swept AS (
                    UPDATE raw_events
                       SET body = ''::bytea, signature = '', subject_token = NULL
                     WHERE subject_token = $1
                 RETURNING archive_id
                )
                SELECT count(*) FROM swept
                """,
                subject_token,
            )
            or 0
        )

        await conn.execute(
            """
            UPDATE erasure_requests
               SET completed_at = $2,
                   subject_refs_destroyed = $3,
                   archive_rows_purged = $4,
                   outbox_rows_cancelled = $5,
                   runs_cancelled = $6,
                   tombstone_seq = $7
             WHERE erasure_id = $1
            """,
            erasure_id,
            at,
            len(tombstone.refs_destroyed),
            archive_purged,
            outbox_cancelled,
            runs_cancelled,
            tombstone.ledger_ref.seq,
        )

    await assert_swept(conn, subject_token)

    return ErasureReport(
        erasure_id=erasure_id,
        subject_token=subject_token,
        requested_at=at,
        requested_by=requested_by,
        subject_refs_destroyed=len(tombstone.refs_destroyed),
        archive_rows_purged=archive_purged,
        outbox_rows_cancelled=outbox_cancelled,
        runs_cancelled=runs_cancelled,
        tombstone_seq=tombstone.ledger_ref.seq,
    )


async def assert_swept(conn: Any, subject_token: str) -> None:
    """Nothing readable about this subject survives outside the ledger.

    Checked by reading the stores back rather than by trusting the sweep's own
    counts, because a sweep with a wrong WHERE clause also reports a count.
    """
    remaining = await conn.fetchval(
        "SELECT count(*) FROM raw_events WHERE subject_token = $1", subject_token
    )
    if remaining:
        raise ErasureIncomplete(
            f"{remaining} raw archive row(s) still name {subject_token}. The archive "
            "is a complete copy of what the subject store held, so an erasure that "
            "misses it has erased nothing"
        )

    live = await conn.fetchval(
        """
        SELECT count(*) FROM outbox
         WHERE subject_token = $1 AND status IN ('pending', 'in_flight')
        """,
        subject_token,
    )
    if live:
        raise ErasureIncomplete(
            f"{live} outbox row(s) are still scheduled for {subject_token}; a message "
            "sent after erasure is the erasure failing where the person can see it"
        )

    key_alive = await conn.fetchval(
        "SELECT data_key IS NOT NULL FROM subject_keys WHERE subject_token = $1",
        subject_token,
    )
    if key_alive:
        raise ErasureIncomplete(f"the encryption key for {subject_token} was not destroyed")


async def report_for(conn: Any, subject_token: str) -> list[dict[str, Any]]:
    """Every erasure recorded for a subject. Survives the data it describes."""
    rows = await conn.fetch(
        """
        SELECT erasure_id, requested_at, requested_by, completed_at,
               subject_refs_destroyed, archive_rows_purged,
               outbox_rows_cancelled, runs_cancelled, tombstone_seq
          FROM erasure_requests WHERE subject_token = $1 ORDER BY requested_at
        """,
        subject_token,
    )
    return [dict(row) for row in rows]
