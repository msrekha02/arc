"""Budgets are RESERVED, never checked (GI-3).

THE DIFFERENCE IS NOT PEDANTRY. Checking reads the remaining budget, decides,
and then writes; between the read and the write another worker does the same,
and both proceed. Under twenty workers that race fires constantly, and the
budget it overruns is a network attempt cap whose penalty is a real fine. So
the reservation is one conditional UPDATE - the cap and the amount already
taken live in the same row, and the row either accepts the increment or it does
not. There is no window to lose.

TWO TIERS, AND THE HORIZON DECIDES WHICH.

    <= 15 minutes    HARD reserve at plan time. The action runs almost
                     immediately, so holding the budget costs nothing and
                     guarantees it is there.

    >  15 minutes    SOFT intent. Recorded, visible to the Allocator as
                     pipeline demand, but the budget is NOT locked. It converts
                     to a hard reservation at wake, inside the same transaction
                     that enqueues the dispatch.

WHY: a retry scheduled for payday three days out would otherwise hold a
network attempt for three days, and a portfolio of those starves every claim
that could have used the budget today. It is also why a FREEZE can release
reservations safely - long-horizon work never held one.

RELEASE HAPPENS ON BOTH PATHS. On a terminal outcome, because the budget was
spent or abandoned and either way it is no longer promised. On expiry, because
a reservation whose decision never executed is a leak, and a leaked reservation
is a starvation bug that stays invisible until the portfolio quietly stops
treating anybody.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from arc.allocator.budgets import PRICED_BUDGETS, BudgetKey, CostVector
from arc.core.time_authority import ensure_utc

# The horizon that separates the two tiers.
HARD_RESERVE_HORIZON = timedelta(minutes=15)

# How long an unconsumed hard reservation survives before the reaper takes it
# back. Comfortably longer than a certificate window, so an ordinary dispatch
# is never swept out from under itself, and far shorter than a cycle.
RESERVATION_TTL = timedelta(hours=2)


class ReservationStatus(StrEnum):
    SOFT = "soft"
    HARD = "hard"
    CONSUMED = "consumed"
    RELEASED = "released"
    EXPIRED = "expired"


LIVE_STATUSES: frozenset[ReservationStatus] = frozenset(
    {ReservationStatus.SOFT, ReservationStatus.HARD}
)


class BudgetExhausted(RuntimeError):
    """The cap refused the reservation. The action does not run."""


class UnknownBudget(LookupError):
    """A budget was reserved against a cycle that never declared a cap."""


@dataclass(frozen=True)
class Reservation:
    reservation_id: int
    cycle_id: UUID
    claim_id: UUID
    budget_key: BudgetKey
    amount: int
    status: ReservationStatus
    idempotency_key: str


def tier_for(horizon: timedelta) -> ReservationStatus:
    """Which tier a decision executing `horizon` from now belongs in."""
    return ReservationStatus.HARD if horizon <= HARD_RESERVE_HORIZON else ReservationStatus.SOFT


async def declare_caps(conn: Any, cycle_id: UUID, caps: Mapping[BudgetKey, int]) -> None:
    """Publish the cycle's caps. Idempotent, so a retried cycle is harmless."""
    for key, cap in caps.items():
        if isinstance(cap, bool) or not isinstance(cap, int):
            raise TypeError(f"cap for {key} must be an integer, got {type(cap).__name__}")
        await conn.execute(
            """
            INSERT INTO budget_caps (cycle_id, budget_key, cap)
            VALUES ($1, $2, $3)
            ON CONFLICT (cycle_id, budget_key) DO NOTHING
            """,
            cycle_id,
            key.value,
            cap,
        )


async def remaining(conn: Any, cycle_id: UUID, key: BudgetKey) -> int | None:
    record = await conn.fetchrow(
        "SELECT cap, reserved FROM budget_caps WHERE cycle_id = $1 AND budget_key = $2",
        cycle_id,
        key.value,
    )
    return None if record is None else record["cap"] - record["reserved"]


async def reserve(
    conn: Any,
    *,
    cycle_id: UUID,
    claim_id: UUID,
    subject_token: str,
    cost: CostVector,
    idempotency_key: str,
    at: datetime,
    horizon: timedelta,
    ttl: timedelta = RESERVATION_TTL,
) -> list[Reservation]:
    """Take the whole cost vector, or none of it.

    ALL OR NOTHING ACROSS DIMENSIONS. A voice call needs a contact slot and
    three voice minutes; reserving the slot and failing on the minutes would
    consume budget for an action that never runs. The caller is inside a
    transaction, so raising unwinds every dimension already taken.

    Idempotent on `(idempotency_key, budget_key)`: a retried commit finds its
    own reservation and reuses it rather than taking the budget twice.
    """
    ensure_utc(at)
    status = tier_for(horizon)
    expires_at = at + ttl
    taken: list[Reservation] = []

    for key in PRICED_BUDGETS:
        amount = cost.as_tuple()[PRICED_BUDGETS.index(key)]
        if amount <= 0:
            continue

        existing = await conn.fetchrow(
            """
            SELECT reservation_id, status, amount
              FROM budget_reservations
             WHERE idempotency_key = $1 AND budget_key = $2
            """,
            idempotency_key,
            key.value,
        )
        if existing is not None:
            taken.append(
                Reservation(
                    reservation_id=existing["reservation_id"],
                    cycle_id=cycle_id,
                    claim_id=claim_id,
                    budget_key=key,
                    amount=existing["amount"],
                    status=ReservationStatus(existing["status"]),
                    idempotency_key=idempotency_key,
                )
            )
            continue

        if status is ReservationStatus.HARD:
            # THE ATOMIC STEP. The predicate is inside the UPDATE, so the
            # decision and the write are the same operation and no second
            # worker can slip between them.
            accepted = await conn.fetchval(
                """
                UPDATE budget_caps
                   SET reserved = reserved + $3
                 WHERE cycle_id = $1
                   AND budget_key = $2
                   AND reserved + $3 <= cap
                RETURNING reserved
                """,
                cycle_id,
                key.value,
                amount,
            )
            if accepted is None:
                declared = await conn.fetchval(
                    "SELECT cap FROM budget_caps WHERE cycle_id = $1 AND budget_key = $2",
                    cycle_id,
                    key.value,
                )
                if declared is None:
                    raise UnknownBudget(
                        f"cycle {cycle_id} declared no cap for {key}; an undeclared budget "
                        "is not an unlimited one"
                    )
                raise BudgetExhausted(
                    f"{key} cannot take {amount} for claim {claim_id} in cycle {cycle_id}; "
                    "the cap refused it and the action does not run"
                )

        record = await conn.fetchrow(
            """
            INSERT INTO budget_reservations
                (cycle_id, claim_id, subject_token, budget_key, amount, status,
                 idempotency_key, reserved_at, expires_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING reservation_id
            """,
            cycle_id,
            claim_id,
            subject_token,
            key.value,
            amount,
            status.value,
            idempotency_key,
            at,
            expires_at,
        )
        taken.append(
            Reservation(
                reservation_id=record["reservation_id"],
                cycle_id=cycle_id,
                claim_id=claim_id,
                budget_key=key,
                amount=amount,
                status=status,
                idempotency_key=idempotency_key,
            )
        )

    return taken


async def harden(
    conn: Any, idempotency_key: str, at: datetime, ttl: timedelta = RESERVATION_TTL
) -> list[Reservation]:
    """Convert soft intent into a real hold, at wake.

    This is the second half of the two-tier rule and it can FAIL: the budget
    the decision assumed three days ago may be gone. Failing here is correct -
    the claim returns to the Allocator for a fresh decision against the world
    as it is now, rather than executing against the world as it was.

    THE EXPIRY IS RESET, and it has to be. A soft intent recorded on Monday
    carries an expiry a couple of hours after Monday; hardening it on Thursday
    without moving that expiry produces a hold that is born already stale, and
    the very next sweep frees it while the dispatch it belongs to is still
    pending. The budget would then be handed to another decision while this one
    is in flight, which is precisely the over-commitment the reservation exists
    to prevent.
    """
    ensure_utc(at)
    rows = await conn.fetch(
        """
        SELECT reservation_id, cycle_id, claim_id, budget_key, amount
          FROM budget_reservations
         WHERE idempotency_key = $1 AND status = 'soft'
        """,
        idempotency_key,
    )
    hardened: list[Reservation] = []
    for row in rows:
        accepted = await conn.fetchval(
            """
            UPDATE budget_caps
               SET reserved = reserved + $3
             WHERE cycle_id = $1 AND budget_key = $2 AND reserved + $3 <= cap
            RETURNING reserved
            """,
            row["cycle_id"],
            row["budget_key"],
            row["amount"],
        )
        if accepted is None:
            raise BudgetExhausted(
                f"{row['budget_key']} was available when this was planned and is not now; "
                "the claim goes back to the Allocator rather than executing on a stale hold"
            )
        await conn.execute(
            """
            UPDATE budget_reservations
               SET status = 'hard', reserved_at = $2, expires_at = $3
             WHERE reservation_id = $1
            """,
            row["reservation_id"],
            at,
            at + ttl,
        )
        hardened.append(
            Reservation(
                reservation_id=row["reservation_id"],
                cycle_id=row["cycle_id"],
                claim_id=row["claim_id"],
                budget_key=BudgetKey(row["budget_key"]),
                amount=row["amount"],
                status=ReservationStatus.HARD,
                idempotency_key=idempotency_key,
            )
        )
    return hardened


async def _settle(conn: Any, idempotency_key: str, at: datetime, status: ReservationStatus) -> int:
    """Move live reservations to a terminal status, returning hard holds."""
    ensure_utc(at)
    rows = await conn.fetch(
        """
        UPDATE budget_reservations
           SET status = $3, released_at = $2
         WHERE idempotency_key = $1
           AND status IN ('soft', 'hard')
        RETURNING cycle_id, budget_key, amount, status
        """,
        idempotency_key,
        at,
        status.value,
    )
    # No cap is credited back here. Only a HARD reservation ever incremented
    # `reserved`, and a consumed one was genuinely spent - crediting it would
    # hand the cycle budget it already used.
    return len(rows)


async def release(conn: Any, idempotency_key: str, at: datetime) -> int:
    """Give the budget back. Called on every terminal outcome.

    A dead dispatch, a cancelled certificate, a claim that reached a terminal
    state - all of them free the hold. Forgetting one of these paths is how a
    portfolio quietly stops treating anybody.
    """
    ensure_utc(at)
    # The CTE reads the PRIOR status before the UPDATE rewrites it. A plain
    # `UPDATE ... RETURNING status` hands back the NEW value, so every row
    # would come back 'released', the hard-reservation branch below would
    # never fire, and the cap would never be credited. That failure is silent:
    # no error, no alert, just a cycle that has less budget every time a
    # dispatch is abandoned until it can treat nobody.
    rows = await conn.fetch(
        """
        WITH target AS (
            SELECT reservation_id, cycle_id, budget_key, amount, status
              FROM budget_reservations
             WHERE idempotency_key = $1
               AND status IN ('soft', 'hard')
             FOR UPDATE
        ), settled AS (
            UPDATE budget_reservations b
               SET status = 'released', released_at = $2
              FROM target t
             WHERE b.reservation_id = t.reservation_id
            RETURNING b.reservation_id
        )
        SELECT cycle_id, budget_key, amount, status FROM target
        """,
        idempotency_key,
        at,
    )
    released = 0
    for row in rows:
        if ReservationStatus(row["status"]) is ReservationStatus.HARD:
            await conn.execute(
                """
                UPDATE budget_caps
                   SET reserved = GREATEST(reserved - $3, 0)
                 WHERE cycle_id = $1 AND budget_key = $2
                """,
                row["cycle_id"],
                row["budget_key"],
                row["amount"],
            )
        released += 1
    return released


async def consume(conn: Any, idempotency_key: str, at: datetime) -> int:
    """The action ran. The budget is spent rather than returned."""
    ensure_utc(at)
    return await _settle(conn, idempotency_key, at, ReservationStatus.CONSUMED)


async def expire_stale(conn: Any, at: datetime) -> int:
    """Sweep reservations whose decision never executed.

    The other half of release. A crash between reserving and dispatching leaves
    a hold nobody will ever settle, and without this sweep the cap erodes by a
    little on every crash until the cycle can treat nobody and no single event
    explains why.
    """
    ensure_utc(at)
    # Same shape as `release`, and for the same reason: the prior status has
    # to survive the UPDATE or the credit-back never happens.
    rows = await conn.fetch(
        """
        WITH target AS (
            SELECT reservation_id, cycle_id, budget_key, amount, status
              FROM budget_reservations
             WHERE status IN ('soft', 'hard')
               AND expires_at < $1
             FOR UPDATE
        ), settled AS (
            UPDATE budget_reservations b
               SET status = 'expired', released_at = $1
              FROM target t
             WHERE b.reservation_id = t.reservation_id
            RETURNING b.reservation_id
        )
        SELECT cycle_id, budget_key, amount, status FROM target
        """,
        at,
    )
    for row in rows:
        if ReservationStatus(row["status"]) is ReservationStatus.HARD:
            await conn.execute(
                """
                UPDATE budget_caps
                   SET reserved = GREATEST(reserved - $3, 0)
                 WHERE cycle_id = $1 AND budget_key = $2
                """,
                row["cycle_id"],
                row["budget_key"],
                row["amount"],
            )
    return len(rows)


async def live_for(conn: Any, claim_id: UUID) -> list[Reservation]:
    rows = await conn.fetch(
        """
        SELECT reservation_id, cycle_id, claim_id, budget_key, amount, status,
               idempotency_key
          FROM budget_reservations
         WHERE claim_id = $1 AND status IN ('soft', 'hard')
        """,
        claim_id,
    )
    return [
        Reservation(
            reservation_id=r["reservation_id"],
            cycle_id=r["cycle_id"],
            claim_id=r["claim_id"],
            budget_key=BudgetKey(r["budget_key"]),
            amount=r["amount"],
            status=ReservationStatus(r["status"]),
            idempotency_key=r["idempotency_key"],
        )
        for r in rows
    ]


async def pipeline_demand(conn: Any, cycle_id: UUID) -> dict[BudgetKey, int]:
    """Soft intent, which the Allocator reads as demand it has not yet paid for.

    Not subtracted from the cap - that is the point of the tier - but a cycle
    that ignores it will over-commit the budget its own pipeline is about to
    need.
    """
    rows = await conn.fetch(
        """
        SELECT budget_key, COALESCE(SUM(amount), 0) AS total
          FROM budget_reservations
         WHERE cycle_id = $1 AND status = 'soft'
         GROUP BY budget_key
        """,
        cycle_id,
    )
    return {BudgetKey(r["budget_key"]): int(r["total"]) for r in rows}


# ---------------------------------------------------------------------------
# The network attempt cap, which is not ours alone
# ---------------------------------------------------------------------------
NETWORK_WINDOW = timedelta(days=30)


async def record_network_attempt(
    conn: Any,
    *,
    instrument_ref: str,
    claim_id: UUID,
    subject_token: str,
    rail: str,
    attempted_at: datetime,
    initiated_by: str,
    attempt_ref: str,
) -> bool:
    """Record one presentation against an instrument. Returns False if known.

    THE GATEWAY'S OWN RETRIES COME THROUGH HERE TOO. It re-presents on its own
    schedule after a failure, without asking, and those attempts count against
    the network cap exactly as ours do. A counter that only knows about ARC's
    retries is wrong in the direction that gets the merchant fined: the cap can
    be exceeded without ARC ever having issued the excess attempt.

    Deduplicated on `(instrument_ref, attempt_ref)` because gateway webhooks
    are redelivered by design.
    """
    ensure_utc(attempted_at)
    inserted = await conn.fetchval(
        """
        INSERT INTO network_attempts
            (instrument_ref, claim_id, subject_token, rail, attempted_at,
             initiated_by, attempt_ref)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (instrument_ref, attempt_ref) DO NOTHING
        RETURNING attempt_id
        """,
        instrument_ref,
        claim_id,
        subject_token,
        rail,
        attempted_at,
        initiated_by,
        attempt_ref,
    )
    return inserted is not None


async def network_attempts_in_window(
    conn: Any, instrument_ref: str, at: datetime, window: timedelta = NETWORK_WINDOW
) -> int:
    """Attempts by ANYONE against this instrument in the rolling window.

    Half-open `[at - window, at)`, like every window in ARC.
    """
    ensure_utc(at)
    return await conn.fetchval(
        """
        SELECT count(*) FROM network_attempts
         WHERE instrument_ref = $1
           AND attempted_at >= $2
           AND attempted_at <  $3
        """,
        instrument_ref,
        at - window,
        at,
    )


async def network_attempts_by_initiator(
    conn: Any, instrument_ref: str, at: datetime, window: timedelta = NETWORK_WINDOW
) -> dict[str, int]:
    rows = await conn.fetch(
        """
        SELECT initiated_by, count(*) AS n FROM network_attempts
         WHERE instrument_ref = $1 AND attempted_at >= $2 AND attempted_at < $3
         GROUP BY initiated_by
        """,
        instrument_ref,
        at - window,
        at,
    )
    return {r["initiated_by"]: r["n"] for r in rows}


def within_network_cap(used: int, cap: int, requesting: int = 1) -> bool:
    return used + requesting <= cap


__all__ = [
    "HARD_RESERVE_HORIZON",
    "LIVE_STATUSES",
    "NETWORK_WINDOW",
    "RESERVATION_TTL",
    "BudgetExhausted",
    "Reservation",
    "ReservationStatus",
    "UnknownBudget",
    "consume",
    "declare_caps",
    "expire_stale",
    "harden",
    "live_for",
    "network_attempts_by_initiator",
    "network_attempts_in_window",
    "pipeline_demand",
    "record_network_attempt",
    "remaining",
    "reserve",
    "release",
    "tier_for",
    "within_network_cap",
]
