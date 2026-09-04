"""Concurrency load test: `python -m tests.load_test --workers 20 --rows 10000`.

Asserts what the build document asks for - `dispatched == rows`,
`duplicates == 0` - and counts both at the PROVIDER rather than at the table.

WHY THAT DISTINCTION IS THE WHOLE POINT. Counting outbox rows in `sent` would
pass even if two workers dispatched the same row: the second UPDATE overwrites
the first, the table shows one row, and the customer has received two messages.
The number that matters is how many times a provider was actually called, and
whether any idempotency key was presented to it twice.

A standalone module rather than a pytest case because ten thousand rows through
twenty workers takes long enough that it does not belong in the ordinary gate.
`tests/test_conductor.py` runs the same scenario three times at a smaller size,
which is what catches the ordinary ways `SKIP LOCKED` gets misused.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from arc.allocator.budgets import BudgetKey
from arc.conductor import reservations
from arc.conductor.commit import CommitRequest, commit_decision
from arc.conductor.outbox import claim_batch
from arc.conductor.worker import dispatch
from arc.core.ids import subject_token
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict
from tests.conductor_db import create_scratch_database, drop_scratch_database

TOKEN = subject_token("+919876543210", pepper=b"m9-load-test-pepper-00000000000")
T0 = datetime(2026, 3, 17, 10, 0, tzinfo=UTC)

# Far longer than any dispatch here, so a duplicate would be a genuine
# `SKIP LOCKED` defect rather than a lease that quietly expired mid-flight.
LEASE = timedelta(minutes=30)

CAPS: dict[BudgetKey, int] = dict.fromkeys(BudgetKey, 10_000_000)
CAPS.pop(BudgetKey.EXPLORE, None)


class Clock:
    def __init__(self, at: datetime) -> None:
        self.at = at

    def now(self) -> datetime:
        return self.at


@dataclass
class RecordingProvider:
    """Every call is recorded. Deduplication is measured, never relied on."""

    invocations: list[str] = field(default_factory=list)
    effects: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, payload: Mapping[str, Any], idempotency_key: str) -> Any:
        async with self._lock:
            self.invocations.append(idempotency_key)
            self.effects[idempotency_key] = self.effects.get(idempotency_key, 0) + 1
        return {"outcome": "delivered"}

    @property
    def duplicates(self) -> int:
        return len(self.invocations) - len(set(self.invocations))


def certificate_for(claim_id: UUID) -> Certificate:
    return Certificate(
        certificate_id=uuid4(),
        decision=Verdict.ALLOW,
        valid_from=T0 - timedelta(hours=1),
        valid_until=T0 + timedelta(hours=1),
        evaluated_rules=(),
        blocking_rule_ids=(),
        defer_until=None,
        rule_registry_version="load-test",
        action=ActionType.SMS,
        issued_at=T0,
        claim_id=claim_id,
    )


async def seed(pool: asyncpg.Pool, rows: int, cycle_id: UUID) -> list[UUID]:
    claim_ids = [uuid4() for _ in range(rows)]
    async with pool.acquire() as conn:
        await reservations.declare_caps(conn, cycle_id, CAPS)
        await conn.executemany(
            """
            INSERT INTO claims
                (claim_id, subject_token, amount_paise, ltv_remaining_paise,
                 claim_type, rail, detected_at, evidence_hash, state)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            [
                (
                    claim_id,
                    TOKEN,
                    129_900,
                    1_500_000,
                    ClaimType.CARD_DECLINE.value,
                    Rail.CARD.value,
                    T0 - timedelta(days=1),
                    b"\x00" * 32,
                    ClaimState.PLANNED.value,
                )
                for claim_id in claim_ids
            ],
        )
        for claim_id in claim_ids:
            await commit_decision(
                conn,
                CommitRequest(
                    claim_id=claim_id,
                    subject_token=TOKEN,
                    cycle_id=cycle_id,
                    action=ActionType.SMS,
                    certificate=certificate_for(claim_id),
                    decision_time=T0,
                    planned_execution_time=T0,
                    pi_intended=0.25,
                    shadow_prices={},
                    payload={"template": "utility_reminder_v3"},
                ),
            )
    return claim_ids


async def drain(
    pool: asyncpg.Pool, workers: int, provider: RecordingProvider, clock: Clock
) -> None:
    async def worker(worker_id: str) -> None:
        idle = 0
        while idle < 3:
            async with pool.acquire() as conn:
                batch = await claim_batch(conn, worker_id, 25, at=clock.now(), lease=LEASE)
                if not batch:
                    idle += 1
                else:
                    idle = 0
                    for row in batch:
                        await dispatch(conn, row, {"sms": provider}, at=clock.now())
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(f"worker-{i:02d}") for i in range(workers)))


async def cleanup(pool: asyncpg.Pool, claim_ids: list[UUID], cycle_id: UUID) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM outbox WHERE claim_id = ANY($1::uuid[])", claim_ids)
        await conn.execute(
            "DELETE FROM budget_reservations WHERE claim_id = ANY($1::uuid[])", claim_ids
        )
        await conn.execute("DELETE FROM budget_caps WHERE cycle_id = $1", cycle_id)
        await conn.execute("DELETE FROM claims WHERE claim_id = ANY($1::uuid[])", claim_ids)


async def run(workers: int, rows: int, dsn: str) -> int:
    cycle_id = uuid4()
    clock = Clock(T0)
    provider = RecordingProvider()

    pool = await asyncpg.create_pool(dsn, min_size=workers // 2, max_size=workers + 6)
    assert pool is not None
    try:
        print(f"seeding {rows} rows ...")
        started = time.perf_counter()
        claim_ids = await seed(pool, rows, cycle_id)
        seeded = time.perf_counter() - started

        print(f"draining with {workers} workers ...")
        started = time.perf_counter()
        await drain(pool, workers, provider, clock)
        drained = time.perf_counter() - started

        async with pool.acquire() as conn:
            statuses = {
                r["status"]: r["n"]
                for r in await conn.fetch(
                    """
                    SELECT status, count(*) AS n FROM outbox
                     WHERE claim_id = ANY($1::uuid[]) GROUP BY status
                    """,
                    claim_ids,
                )
            }

        dispatched = len(provider.invocations)
        distinct = len(set(provider.invocations))
        duplicates = provider.duplicates

        print()
        print("=" * 62)
        print(f"  workers                      {workers}")
        print(f"  rows                         {rows}")
        print(f"  seed time                    {seeded:.1f}s")
        print(
            f"  drain time                   {drained:.1f}s "
            f"({rows / max(drained, 1e-9):.0f} rows/s)"
        )
        print(f"  provider invocations         {dispatched}")
        print(f"  distinct idempotency keys    {distinct}")
        print(f"  duplicate invocations        {duplicates}")
        print(f"  outbox statuses              {statuses}")
        print("=" * 62)

        failures: list[str] = []
        if dispatched != rows:
            failures.append(f"dispatched {dispatched}, expected {rows}")
        if duplicates != 0:
            failures.append(f"{duplicates} duplicate provider invocations")
        if distinct != rows:
            failures.append(f"{distinct} distinct keys, expected {rows}")
        if statuses.get("sent") != rows:
            failures.append(f"outbox sent={statuses.get('sent')}, expected {rows}")

        if failures:
            print("FAIL: " + "; ".join(failures))
            return 1
        print("PASS: every row dispatched exactly once")
        return 0
    finally:
        await cleanup(pool, claim_ids, cycle_id)
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Conductor concurrency load test")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument(
        "--dsn",
        default=None,
        help=(
            "target database. Defaults to a scratch database created and dropped "
            "around the run, because this test commits tens of thousands of "
            "append-only ledger rows that cannot be deleted afterwards."
        ),
    )
    args = parser.parse_args()

    if args.dsn:
        return asyncio.run(run(args.workers, args.rows, args.dsn))

    dsn = create_scratch_database("loadtest")
    try:
        return asyncio.run(run(args.workers, args.rows, dsn))
    finally:
        drop_scratch_database("loadtest")


if __name__ == "__main__":
    sys.exit(main())
