"""Four modes, per-layer semantics, and a resume that cannot stampede.

    NORMAL   L0-L4 run   L5 certify   L6 reserve + dispatch   L7 execute
    SHADOW   L0-L4 run   L5 certify   L6 record intent only   L7 idle
    DRAIN    admission stopped        L6 finish in-flight     L7 in-flight
    FREEZE   L0-L4 run   L5 certify   L6 RELEASE + mark HELD  L7 idle

WHY SHADOW SCHEDULES NOTHING AT ALL, rather than scheduling and holding. A
shadow mode that queued work would accumulate a backlog for the whole duration
of whatever it was switched on for, and the moment it switched off that backlog
would go out at once - to people whose circumstances had moved on by however
long the shadow lasted. So SHADOW takes no reservation and writes no outbox
row. It records the intent, which is what makes it useful for observing what
the system WOULD have done, and nothing else.

WHY FREEZE RELEASES RESERVATIONS. A frozen system holding budget starves the
portfolio for the duration of the freeze while doing nothing with it. It is
also safe to release precisely because long-horizon work never held a hard
reservation in the first place - that is the two-tier rule at M9 paying off
here.

WHY HELD WORK IS INVALIDATED ON RESUME AND NEVER EXECUTED. Two reasons, and
either alone is sufficient. The world changed during the freeze: the whole
point of freezing is that something was wrong, and executing decisions made
before it would act on the state that made freezing necessary. And the
certificates have expired anyway - a freeze long enough to matter outlasts a
certificate window, so executing held work would be executing on stale
authorisation, which the Conductor refuses everywhere else.

    THE TEMPTING BUG IS A "RESUME" PATH THAT DISPATCHES. It looks efficient:
    the work was already decided, already certified, already costed. It is the
    thundering herd and the stale-authorisation bug at once, and `resume`
    below has no branch that can execute - held items are requeued through the
    same path everything else uses, so there is no special-case resume code to
    get wrong.

THE ADMISSION RAMP IS NOT POLITENESS. Coming back at full volume after a freeze
sends a burst that is itself a volume surge, which trips CB-VOLUME, which
freezes the system again. 5, 25, 60, 100 percent of the trailing median over
four cycles is slow enough that the breaker sees a ramp rather than a spike.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from arc.conductor import reservations
from arc.conductor.worker import requeue_for_allocation
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType
from arc.ledger.decision_ledger import DecisionLedger


class Mode(StrEnum):
    NORMAL = "normal"
    SHADOW = "shadow"
    DRAIN = "drain"
    FREEZE = "freeze"


# The ramp, as fractions of trailing median volume. Four steps, and the last
# one is the end of the ramp rather than a cap.
RAMP: tuple[float, ...] = (0.05, 0.25, 0.60, 1.00)


class HeldWorkExecuted(AssertionError):
    """Something tried to dispatch work a freeze had invalidated.

    An assertion rather than a guard that returns False: held work reaching a
    dispatcher means the resume path has a branch it must not have, and
    continuing would send a message authorised by an expired certificate.
    """


@dataclass(frozen=True)
class ControlState:
    mode: Mode
    changed_at: datetime
    changed_by: str
    reason: str
    ramp_step: int | None
    ramp_started_at: datetime | None

    @property
    def admits_new_work(self) -> bool:
        """Whether L0-L4 output may become scheduled work at all.

        DRAIN stops admission and lets in-flight finish, which is what makes it
        different from FREEZE: nothing new arrives, but nothing already
        authorised is thrown away either.
        """
        return self.mode is Mode.NORMAL

    @property
    def reserves_budget(self) -> bool:
        return self.mode is Mode.NORMAL

    @property
    def dispatches(self) -> bool:
        """DRAIN still dispatches: that is the whole meaning of draining."""
        return self.mode in (Mode.NORMAL, Mode.DRAIN)

    @property
    def ramp_fraction(self) -> float:
        """Share of trailing median volume admissible right now."""
        if self.ramp_step is None:
            return 1.0
        return RAMP[min(self.ramp_step, len(RAMP) - 1)]

    def admission_cap(self, trailing_median: int) -> int:
        """How many items may be admitted this cycle.

        Rounded DOWN, and floored at zero rather than one: a ramp that always
        lets one through is not a ramp when the trailing median is small.
        """
        if not self.admits_new_work:
            return 0
        return int(trailing_median * self.ramp_fraction)


async def current_mode(conn: Any) -> ControlState:
    row = await conn.fetchrow(
        """
        SELECT mode, changed_at, changed_by, reason, ramp_step, ramp_started_at
          FROM system_control WHERE only_row
        """
    )
    if row is None:  # pragma: no cover - the migration seeds this row
        raise LookupError("system_control has no row; migration 005 did not run")
    return ControlState(
        mode=Mode(row["mode"]),
        changed_at=row["changed_at"],
        changed_by=row["changed_by"],
        reason=row["reason"],
        ramp_step=row["ramp_step"],
        ramp_started_at=row["ramp_started_at"],
    )


async def set_mode(
    conn: Any,
    mode: Mode,
    *,
    at: datetime,
    changed_by: str,
    reason: str,
    ramp_step: int | None = None,
) -> ControlState:
    ensure_utc(at)
    await conn.execute(
        """
        UPDATE system_control
           SET mode = $1::system_mode,
               changed_at = $2::timestamptz,
               changed_by = $3,
               reason = $4,
               ramp_step = $5::smallint,
               -- Both branches are cast, or Postgres cannot deduce a type for
               -- a parameter used as both a timestamp and a CASE result.
               ramp_started_at = CASE
                   WHEN $5::smallint IS NULL THEN NULL::timestamptz
                   ELSE $2::timestamptz END
         WHERE only_row
        """,
        mode.value,
        at,
        changed_by,
        reason,
        ramp_step,
    )
    return await current_mode(conn)


# ---------------------------------------------------------------------------
# FREEZE
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FreezeReport:
    outbox_cancelled: int
    reservations_released: int
    held: int


async def freeze(
    conn: Any, *, at: datetime, changed_by: str, reason: str, ledger: DecisionLedger | None = None
) -> FreezeReport:
    """Stop, release the budget, and mark scheduled work HELD.

    Held rows are moved out of the dispatch path entirely - status `cancelled`
    in the outbox, recorded in `held_work` - so no worker can pick them up.
    They are NOT deleted: the resume path has to be able to hand each one back
    to the Allocator, and a claim that vanished during a freeze would simply
    never be treated again.
    """
    ensure_utc(at)
    ledger = ledger or DecisionLedger()

    async with conn.transaction():
        await set_mode(conn, Mode.FREEZE, at=at, changed_by=changed_by, reason=reason)

        scheduled = await conn.fetch(
            """
            SELECT id, claim_id, subject_token, idempotency_key, action_type
              FROM outbox
             WHERE status IN ('pending', 'failed')
            """
        )
        for row in scheduled:
            await conn.execute(
                """
                INSERT INTO held_work
                    (claim_id, subject_token, outbox_id, idempotency_key,
                     action_type, held_at, reason)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (idempotency_key) DO NOTHING
                """,
                row["claim_id"],
                row["subject_token"],
                row["id"],
                row["idempotency_key"],
                row["action_type"],
                at,
                reason,
            )

        cancelled = int(
            await conn.fetchval(
                """
                WITH moved AS (
                    UPDATE outbox SET status = 'cancelled', last_error = $1
                     WHERE status IN ('pending', 'failed')
                 RETURNING id
                )
                SELECT count(*) FROM moved
                """,
                f"KILL_SWITCH:FREEZE:{reason}",
            )
            or 0
        )

        released = 0
        for row in scheduled:
            released += await reservations.release(conn, row["idempotency_key"], at)

    return FreezeReport(
        outbox_cancelled=cancelled, reservations_released=released, held=len(scheduled)
    )


# ---------------------------------------------------------------------------
# RESUME
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResumeReport:
    mode: Mode
    ramp_step: int
    invalidated: int
    requeued: int
    executed: int = 0  # ALWAYS zero. See `assert_nothing_executed` below.


async def resume(
    conn: Any,
    *,
    at: datetime,
    changed_by: str,
    ledger: DecisionLedger | None = None,
) -> ResumeReport:
    """Come back, ramped, with every held item invalidated.

    THERE IS NO EXECUTE BRANCH IN THIS FUNCTION. Held items are requeued for a
    fresh decision through `requeue_for_allocation`, the same path a cancelled
    dispatch and an expired certificate already use. Adding an execute branch
    would need new code that does not exist, which is the point.
    """
    ensure_utc(at)
    ledger = ledger or DecisionLedger()

    async with conn.transaction():
        held = await conn.fetch(
            """
            SELECT held_id, claim_id, subject_token, idempotency_key
              FROM held_work WHERE released_at IS NULL
          ORDER BY held_id
            """
        )
        for row in held:
            await requeue_for_allocation(
                conn,
                row["claim_id"],
                reason="RESUME_HELD_INVALIDATED",
                at=at,
                ledger=ledger,
                subject_token=row["subject_token"],
            )
            await conn.execute(
                "UPDATE held_work SET released_at = $2 WHERE held_id = $1", row["held_id"], at
            )

        state = await set_mode(
            conn,
            Mode.NORMAL,
            at=at,
            changed_by=changed_by,
            reason="resume with ramped admission",
            ramp_step=0,
        )

    return ResumeReport(
        mode=state.mode,
        ramp_step=0,
        invalidated=len(held),
        requeued=len(held),
    )


async def advance_ramp(conn: Any, *, at: datetime) -> ControlState:
    """One rung. At the top the ramp clears rather than sticking at 100%."""
    ensure_utc(at)
    state = await current_mode(conn)
    if state.ramp_step is None:
        return state
    nxt = state.ramp_step + 1
    if nxt >= len(RAMP):
        return await set_mode(
            conn,
            state.mode,
            at=at,
            changed_by="ramp",
            reason="ramp complete",
            ramp_step=None,
        )
    return await set_mode(
        conn,
        state.mode,
        at=at,
        changed_by="ramp",
        reason=f"ramp step {nxt}",
        ramp_step=nxt,
    )


async def held_keys(conn: Any) -> list[str]:
    rows = await conn.fetch(
        "SELECT idempotency_key FROM held_work WHERE released_at IS NULL ORDER BY held_id"
    )
    return [r["idempotency_key"] for r in rows]


async def assert_nothing_executed(conn: Any, keys: Sequence[str]) -> None:
    """No held key may appear as a sent outbox row. Raises if one does.

    The check the resume path is graded on. It reads the outbox rather than
    trusting the resume function's own report, because a resume path that
    executed held work would also be the thing writing that report.
    """
    if not keys:
        return
    sent = await conn.fetch(
        """
        SELECT idempotency_key, status FROM outbox
         WHERE idempotency_key = ANY($1::text[])
           AND status IN ('sent', 'in_flight')
        """,
        list(keys),
    )
    if sent:
        raise HeldWorkExecuted(
            "work held by a freeze was dispatched on resume: "
            + ", ".join(f"{r['idempotency_key']}={r['status']}" for r in sent)
            + ". Held decisions were made before the freeze and their certificates "
            "have expired; they must be returned to the Allocator, never executed"
        )


# ---------------------------------------------------------------------------
# SHADOW
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShadowIntent:
    """What the system WOULD have done. Recorded, never scheduled."""

    claim_id: UUID
    subject_token: str
    action: ActionType
    at: datetime


async def record_shadow_intent(
    conn: Any, intent: ShadowIntent, *, ledger: DecisionLedger | None = None
) -> int:
    """Append the intent to the ledger. No reservation, no outbox row.

    Deliberately NOT an enqueue with a flag. A flag on a real outbox row is one
    forgotten `WHERE` clause away from being dispatched, and the whole value of
    SHADOW is that there is nothing for a worker to find.
    """
    from arc.ledger.decision_ledger import LedgerEntry, LedgerEventType

    ledger = ledger or DecisionLedger()
    ref = await ledger.append(
        conn,
        LedgerEntry(
            event_type=LedgerEventType.DECISION,
            occurred_at=intent.at,
            claim_id=intent.claim_id,
            subject_token=intent.subject_token,
            payload={
                "mode": Mode.SHADOW.value,
                "intended_action": intent.action.value,
                "scheduled": False,
                "reserved": False,
            },
        ),
    )
    return ref.seq
