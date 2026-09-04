"""M9 acceptance gate: exactly-once state, effectively-once effect.

The nine named tests are:

    test_20_concurrent_workers_zero_duplicate_dispatch
    test_crash_between_decide_and_dispatch_leaves_no_orphan
    test_lease_expiry_reclaims_row
    test_idempotency_key_stable_across_retries
    test_expired_certificate_cancels_and_requeues
    test_expired_certificate_never_executes
    test_reservation_released_on_terminal_and_on_timeout
    test_illegal_fsm_transition_rejected
    test_gateway_own_retry_counted_against_network_budget

plus `tests/load_test.py`, runnable as
`python -m tests.load_test --workers 20 --rows 10000`.

DUPLICATES ARE COUNTED AT THE PROVIDER, NOT AT THE TABLE. Those are different
properties and only one of them is the one that matters. Counting outbox rows
in `sent` would pass even if two workers both called the provider and the
second one's UPDATE simply overwrote the first - the customer would have
received two messages and the table would show one row. So the fake provider
records EVERY invocation, and the assertion is that the number of invocations
equals the number of distinct idempotency keys equals the number of rows. A
provider that deduplicates internally is a second line of defence and is
measured separately, because relying on it would mean the Conductor had already
failed.

CONCURRENCY TESTS COMMIT FOR REAL. They cannot run inside a rolled-back
transaction, because `SKIP LOCKED` between twenty workers is only meaningful
across twenty committed sessions. They clean up after themselves by claim id;
the decision ledger is append-only by design and its rows stay.
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from arc.allocator.budgets import BudgetKey, CostVector, cost_of
from arc.conductor import reservations
from arc.conductor.commit import CommitRequest, UncertifiedAction, commit_decision
from arc.conductor.fsm import (
    ClaimNotFound,
    ConcurrentTransition,
    current_state,
    transition,
)
from arc.conductor.outbox import (
    OutboxStatus,
    by_idempotency_key,
    claim_batch,
    enqueue,
    fetch_row,
    idempotency_key,
    reap_expired_leases,
)
from arc.conductor.worker import (
    DispatchOutcome,
    PermanentError,
    RetryableError,
    dispatch,
    release_on_terminal,
)
from arc.core.ids import subject_token
from arc.core.time_authority import TimezoneBasis, TzBasisKind
from arc.core.types import ActionType, ClaimState, ClaimType, IllegalTransition, Rail
from arc.gate.context import Channel, ConsentState, GateContext
from arc.gate.evaluator import Certificate, Gate
from arc.gate.lattice import Verdict
from arc.gate.registry import load_registry
from tests.conductor_db import scratch_database

REPO_ROOT = Path(__file__).resolve().parents[1]

PEPPER = b"m9-acceptance-gate-pepper-000000"
TOKEN = subject_token("+919876543210", pepper=PEPPER)
T0 = datetime(2026, 3, 17, 10, 0, tzinfo=UTC)
IST = TimezoneBasis(TzBasisKind.DECLARED, "Asia/Kolkata")

# Long enough that the concurrency tests never trip it, so a duplicate
# dispatch there is a real defect rather than an expired lease.
TEST_LEASE = timedelta(minutes=5)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    """A scratch database, created for this session and dropped after it.

    THE CONCURRENCY TESTS COMMIT. They have to - `SKIP LOCKED` between twenty
    workers only means anything across twenty committed sessions. But the
    decision ledger is append-only by construction, so the rows they write
    cannot be deleted afterwards, and writing them into the shared development
    database would leave permanent audit entries in it. An audit log with test
    traffic in it is not an audit log, and M2's gate rightly asserts the ledger
    starts empty.
    """
    try:
        yield from scratch_database("conductor")
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover
        pytest.skip(f"postgres unavailable: {exc}")


@pytest.fixture
async def conn(dsn: str) -> AsyncIterator[asyncpg.Connection]:
    """One connection inside a transaction that is always rolled back."""
    connection = await asyncpg.connect(dsn)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


@pytest.fixture
async def pool(dsn: str) -> AsyncIterator[asyncpg.Pool]:
    """A real pool. Concurrency tests commit and clean up by claim id."""
    created = await asyncpg.create_pool(dsn, min_size=4, max_size=26)
    assert created is not None
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture(scope="module")
def gate() -> Gate:
    return Gate(load_registry())


class FrozenClock:
    """Time as a parameter, not a reading. Nothing here calls a wall clock."""

    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, delta: timedelta) -> None:
        self._at += delta


# ---------------------------------------------------------------------------
# The fake provider
# ---------------------------------------------------------------------------
@dataclass
class ChannelResult:
    outcome: str = "delivered"
    deduplicated: bool = False


@dataclass
class FakeProvider:
    """Records EVERY call, and separately what actually took effect.

    Two counters because they answer different questions. `invocations` is what
    the Conductor did - if the same key appears twice, two workers dispatched
    one row and the Conductor failed, whatever the provider then did about it.
    `effects` is what reached the customer, which a provider honouring
    idempotency keys keeps at one per key regardless.

    Asserting only on `effects` would let a double-dispatch pass, because the
    provider would have absorbed it. The Conductor's guarantee is that it never
    gets that far.
    """

    name: str = "fake"
    invocations: list[str] = field(default_factory=list)
    effects: dict[str, int] = field(default_factory=dict)
    fail_with: Exception | None = None
    fail_times: int = 0
    delay: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, payload: Mapping[str, Any], idempotency_key: str) -> ChannelResult:
        async with self._lock:
            self.invocations.append(idempotency_key)
            first = idempotency_key not in self.effects

        if self.delay:
            await asyncio.sleep(self.delay)

        if self.fail_with is not None and self.fail_times > 0:
            async with self._lock:
                if self.fail_times > 0:
                    self.fail_times -= 1
                    raise self.fail_with

        async with self._lock:
            self.effects[idempotency_key] = self.effects.get(idempotency_key, 0) + 1
        return ChannelResult(deduplicated=not first)

    @property
    def duplicate_invocations(self) -> int:
        return len(self.invocations) - len(set(self.invocations))


def channel_map(provider: FakeProvider) -> dict[str, FakeProvider]:
    """Every channel served by one recorder, so any route is counted."""
    return {channel.value: provider for channel in Channel}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
async def insert_claim(
    conn: Any,
    claim_id: UUID,
    *,
    state: ClaimState = ClaimState.PLANNED,
    token: str = TOKEN,
    amount: int = 129_900,
) -> None:
    await conn.execute(
        """
        INSERT INTO claims
            (claim_id, subject_token, amount_paise, ltv_remaining_paise,
             claim_type, rail, detected_at, evidence_hash, state)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (claim_id) DO NOTHING
        """,
        claim_id,
        token,
        amount,
        1_500_000,
        ClaimType.CARD_DECLINE.value,
        Rail.CARD.value,
        T0 - timedelta(days=1),
        b"\x00" * 32,
        state.value,
    )


def make_certificate(
    claim_id: UUID,
    action: ActionType = ActionType.SMS,
    *,
    valid_from: datetime = T0 - timedelta(minutes=15),
    valid_until: datetime = T0 + timedelta(minutes=15),
    decision: Verdict = Verdict.ALLOW,
    certificate_id: UUID | None = None,
) -> Certificate:
    return Certificate(
        certificate_id=certificate_id or uuid4(),
        decision=decision,
        valid_from=valid_from,
        valid_until=valid_until,
        evaluated_rules=(),
        blocking_rule_ids=(),
        defer_until=None,
        rule_registry_version="test-registry-1",
        action=action,
        issued_at=valid_from,
        claim_id=claim_id,
    )


def make_request(
    claim_id: UUID,
    cycle_id: UUID,
    *,
    action: ActionType = ActionType.SMS,
    certificate: Certificate | None = None,
    planned: datetime | None = None,
    token: str = TOKEN,
) -> CommitRequest:
    certificate = certificate or make_certificate(claim_id, action)
    return CommitRequest(
        claim_id=claim_id,
        subject_token=token,
        cycle_id=cycle_id,
        action=action,
        certificate=certificate,
        decision_time=T0,
        planned_execution_time=planned or T0,
        pi_intended=0.31,
        shadow_prices={"contact": 1200.0},
        payload={"template": "utility_reminder_v3", "amount_paise": 129_900},
    )


DEFAULT_CAPS: dict[BudgetKey, int] = {
    BudgetKey.CONTACT: 50_000,
    BudgetKey.VOICE: 50_000,
    BudgetKey.RUPEE: 500_000_000,
    BudgetKey.RETRY: 50_000,
    BudgetKey.HUMAN: 50_000,
    BudgetKey.CONCESSION: 50_000,
}


async def cleanup(pool: asyncpg.Pool, claim_ids: list[UUID], cycle_id: UUID) -> None:
    """Remove everything this test committed, except the append-only ledger."""
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM outbox WHERE claim_id = ANY($1::uuid[])", claim_ids)
        await conn.execute(
            "DELETE FROM budget_reservations WHERE claim_id = ANY($1::uuid[])", claim_ids
        )
        await conn.execute("DELETE FROM budget_caps WHERE cycle_id = $1", cycle_id)
        await conn.execute(
            "DELETE FROM network_attempts WHERE claim_id = ANY($1::uuid[])", claim_ids
        )
        await conn.execute("DELETE FROM claims WHERE claim_id = ANY($1::uuid[])", claim_ids)


# ===========================================================================
# 1 - THE MILESTONE
# ===========================================================================
async def _run_concurrency_round(
    pool: asyncpg.Pool, *, rows: int, workers: int, round_no: int
) -> tuple[FakeProvider, dict[str, int]]:
    cycle_id = uuid4()
    claim_ids = [uuid4() for _ in range(rows)]
    clock = FrozenClock(T0)
    provider = FakeProvider(name=f"round{round_no}")

    async with pool.acquire() as conn:
        await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
        for claim_id in claim_ids:
            await insert_claim(conn, claim_id)
        for claim_id in claim_ids:
            await commit_decision(conn, make_request(claim_id, cycle_id))

    async def worker(worker_id: str) -> None:
        idle = 0
        while idle < 3:
            async with pool.acquire() as connection:
                batch = await claim_batch(
                    connection, worker_id, 20, at=clock.now(), lease=TEST_LEASE
                )
                if not batch:
                    idle += 1
                else:
                    idle = 0
                    for row in batch:
                        await dispatch(connection, row, channel_map(provider), at=clock.now())
            await asyncio.sleep(0)

    await asyncio.gather(*(worker(f"w{i}-{round_no}") for i in range(workers)))

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
    await cleanup(pool, claim_ids, cycle_id)
    return provider, statuses


@pytest.mark.parametrize("round_no", [1, 2, 3])
async def test_20_concurrent_workers_zero_duplicate_dispatch(
    pool: asyncpg.Pool, round_no: int
) -> None:
    """Twenty workers, one queue, and nobody sends the same thing twice.

    RUN REPEATEDLY. A concurrency bug is probabilistic - a single green run
    means the race did not happen to fire, not that it cannot. Three rounds
    with independent data is still not a proof, but it is enough to catch the
    ordinary ways `SKIP LOCKED` gets misused.

    THE ASSERTION IS ON PROVIDER INVOCATIONS. Counting `sent` rows would pass
    even if two workers both dispatched one row, because the second UPDATE
    would simply overwrite the first and the table would look fine while the
    customer had received two messages.
    """
    rows = 600
    provider, statuses = await _run_concurrency_round(
        pool, rows=rows, workers=20, round_no=round_no
    )

    assert len(provider.invocations) == rows, (
        f"round {round_no}: the provider was called {len(provider.invocations)} times "
        f"for {rows} rows"
    )
    assert provider.duplicate_invocations == 0, (
        f"round {round_no}: {provider.duplicate_invocations} idempotency keys were "
        "dispatched more than once; two workers claimed the same row"
    )
    assert len(set(provider.invocations)) == rows
    # The provider's own deduplication saw nothing to deduplicate, which is
    # what "the Conductor never got that far" looks like from the far side.
    assert set(provider.effects.values()) == {1}
    assert statuses.get("sent") == rows, f"round {round_no}: statuses were {statuses}"


# ===========================================================================
# 2 - the crash
# ===========================================================================
CRASH_SCRIPT = textwrap.dedent(
    """
    import asyncio, sys
    from datetime import UTC, datetime, timedelta
    from pathlib import Path
    from uuid import UUID

    import asyncpg
    from arc.allocator.budgets import BudgetKey
    from arc.conductor import reservations
    from arc.conductor.fsm import transition
    from arc.conductor.outbox import enqueue
    from arc.core.types import ActionType, ClaimState
    from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType

    DSN, CLAIM, CYCLE, TOKEN, MARKER = sys.argv[1:6]
    claim_id, cycle_id = UUID(CLAIM), UUID(CYCLE)
    T0 = datetime(2026, 3, 17, 10, 0, tzinfo=UTC)


    async def main():
        conn = await asyncpg.connect(DSN)
        tx = conn.transaction()
        await tx.start()

        # Everything commit_decision does, in the same order...
        await transition(conn, claim_id, frm=ClaimState.PLANNED,
                         to=ClaimState.IN_TREATMENT, at=T0)
        await reservations.reserve(
            conn, cycle_id=cycle_id, claim_id=claim_id, subject_token=TOKEN,
            cost=__import__("arc.allocator.budgets", fromlist=["cost_of"]).cost_of(
                ActionType.SMS),
            idempotency_key="crash-test-key", at=T0, horizon=timedelta(minutes=1),
        )
        await DecisionLedger().append(conn, LedgerEntry(
            event_type=LedgerEventType.DECISION, occurred_at=T0, claim_id=claim_id,
            subject_token=TOKEN, payload={"intended_action": "sms"}))
        await enqueue(
            conn, claim_id=claim_id, subject_token=TOKEN, cycle_id=cycle_id,
            action_type=ActionType.SMS, channel="sms", payload={"x": 1},
            certificate_id=UUID(int=7), cert_valid_from=T0 - timedelta(minutes=5),
            cert_valid_until=T0 + timedelta(minutes=5), not_before=T0)

        # ...and then dies, with the transaction still open.
        Path(MARKER).write_text("ready", encoding="utf-8")
        await asyncio.sleep(600)


    asyncio.run(main())
    """
)


async def test_crash_between_decide_and_dispatch_leaves_no_orphan(
    pool: asyncpg.Pool, dsn: str, tmp_path: Path
) -> None:
    """Kill the process mid-transaction. Nothing may survive.

    A REAL KILL, NOT A MOCKED FAILURE. Mocking an exception tests the `except`
    branch; it does not test that Postgres unwinds a half-applied transaction
    when the client vanishes without saying goodbye, which is what actually
    happens when a pod is evicted.

    The orphan this prevents is specific and silent: the claim reads
    IN_TREATMENT forever, no outbox row exists so nothing will dispatch it,
    nothing is leased so no reaper looks at it, and no counter anywhere is
    wrong. One customer is simply never contacted again.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    marker = tmp_path / "ready.txt"

    async with pool.acquire() as conn:
        await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
        await insert_claim(conn, claim_id)

    script = tmp_path / "crasher.py"
    script.write_text(CRASH_SCRIPT, encoding="utf-8")

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        dsn,
        str(claim_id),
        str(cycle_id),
        TOKEN,
        str(marker),
        cwd=str(REPO_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not marker.exists():
            if process.returncode is not None:
                _, err = await process.communicate()
                pytest.fail(f"the crasher exited early:\n{err.decode(errors='replace')}")
            await asyncio.sleep(0.05)
        assert marker.exists(), "the crasher never reached the pre-commit point"

        # The writes are visible to that transaction and to nobody else.
        async with pool.acquire() as conn:
            assert await current_state(conn, claim_id) is ClaimState.PLANNED, (
                "an uncommitted transition leaked to another session"
            )

        # SIGKILL, not a graceful shutdown. No chance to commit, no atexit
        # hook, no polite connection close - the socket simply goes away.
        process.kill()
        await asyncio.wait_for(process.wait(), timeout=30)
    finally:
        if process.returncode is None:  # pragma: no cover
            process.kill()

    # Postgres notices the dead client and unwinds. Poll rather than assume.
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        async with pool.acquire() as conn:
            still_open = await conn.fetchval(
                """
                SELECT count(*) FROM pg_stat_activity
                 WHERE datname = current_database() AND state = 'idle in transaction'
                """
            )
        if not still_open:
            break
        await asyncio.sleep(0.1)

    async with pool.acquire() as conn:
        assert await current_state(conn, claim_id) is ClaimState.PLANNED, (
            "the FSM transition survived a crash that never committed"
        )
        assert (
            await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0
        ), "an orphaned outbox row survived"
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM budget_reservations WHERE claim_id = $1", claim_id
            )
            == 0
        ), "a reservation leaked; the budget is now short and nothing will free it"
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM decision_ledger WHERE claim_id = $1", claim_id
            )
            == 0
        ), "a decision was recorded for something that did not happen"
        assert (
            await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)
            == (DEFAULT_CAPS[BudgetKey.CONTACT])
        )

    await cleanup(pool, [claim_id], cycle_id)


# ===========================================================================
# 3
# ===========================================================================
async def test_lease_expiry_reclaims_row(conn: Any) -> None:
    """A worker that dies holding a row must not strand it forever."""
    claim_id, cycle_id = uuid4(), uuid4()
    clock = FrozenClock(T0)
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    await commit_decision(conn, make_request(claim_id, cycle_id))

    lease = timedelta(minutes=2)
    taken = await claim_batch(conn, "doomed-worker", 10, at=clock.now(), lease=lease)
    assert len(taken) == 1
    row = taken[0]
    assert row.lease_owner == "doomed-worker"
    assert row.attempts == 1

    # Still leased: another worker sees nothing.
    assert await claim_batch(conn, "healthy", 10, at=clock.now(), lease=lease) == []
    assert await reap_expired_leases(conn, clock.now()) == 0

    # After the lease expires the reaper returns it.
    clock.advance(lease + timedelta(seconds=1))
    assert await reap_expired_leases(conn, clock.now()) == 1

    reclaimed = await fetch_row(conn, row.id)
    assert reclaimed.status is OutboxStatus.PENDING
    assert reclaimed.lease_owner is None and reclaimed.lease_expires_at is None

    # And it is claimable again, with the attempt counter carried forward.
    again = await claim_batch(conn, "healthy", 10, at=clock.now(), lease=lease)
    assert len(again) == 1
    assert again[0].id == row.id
    assert again[0].attempts == 2, "the attempt counter must survive a reclaim"

    # The attempt counter moved and the idempotency key did not.
    assert again[0].idempotency_key == row.idempotency_key


# ===========================================================================
# 4
# ===========================================================================
async def test_idempotency_key_stable_across_retries(conn: Any, gate: Gate) -> None:
    """Retries reuse the key. Re-decisions do not.

    M3 derives `certificate_id` as a UUIDv5 over the evaluation rather than
    generating a random one, which makes this property structural rather than
    something dispatch has to remember. So the test asserts the property
    directly instead of trusting the derivation: certify the same inputs twice
    and the key must be byte-identical.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    ctx = GateContext(
        claim_id=claim_id,
        subject_token=TOKEN,
        rail=Rail.CARD,
        claim_state=ClaimState.PLANNED,
        amount_paise=129_900,
        tz_basis=IST,
        consent={channel: ConsentState.GRANTED for channel in Channel},
    )

    first = gate.certify(ctx, ActionType.SMS, T0)
    second = gate.certify(ctx, ActionType.SMS, T0)
    assert first.certificate_id == second.certificate_id, (
        "the Gate is not deterministic, so the idempotency key cannot be stable"
    )

    key_a = idempotency_key(claim_id, ActionType.SMS, cycle_id, first.certificate_id)
    key_b = idempotency_key(claim_id, ActionType.SMS, cycle_id, second.certificate_id)
    assert key_a == key_b

    # The attempt counter is nowhere in it: claiming a row repeatedly moves
    # `attempts` and leaves the key untouched.
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    await commit_decision(conn, make_request(claim_id, cycle_id, certificate=first))
    clock = FrozenClock(T0)
    keys: set[str] = set()
    attempts: list[int] = []
    for _ in range(4):
        rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=timedelta(seconds=30))
        assert len(rows) == 1
        keys.add(rows[0].idempotency_key)
        attempts.append(rows[0].attempts)
        clock.advance(timedelta(seconds=31))
        await reap_expired_leases(conn, clock.now())

    assert keys == {key_a}, "the key changed across dispatch retries"
    assert attempts == [1, 2, 3, 4], "the attempt counter did not advance"

    # A genuine RE-DECISION must produce a different key, or a fresh
    # instruction would be deduplicated away as a retry of the old one.
    other_cycle = idempotency_key(claim_id, ActionType.SMS, uuid4(), first.certificate_id)
    assert other_cycle != key_a

    later = gate.certify(ctx, ActionType.SMS, T0 + timedelta(hours=3))
    assert later.certificate_id != first.certificate_id
    assert idempotency_key(claim_id, ActionType.SMS, cycle_id, later.certificate_id) != key_a

    # And a different action is a different key.
    assert idempotency_key(claim_id, ActionType.EMAIL, cycle_id, first.certificate_id) != key_a


# ===========================================================================
# 5 and 6 - the certificate window, with the assertion factored out
# ===========================================================================
async def assert_refuses_expired_certificate(
    dispatch_fn: Any,
    conn: Any,
    row: Any,
    provider: FakeProvider,
    at: datetime,
    *,
    label: str,
) -> Any:
    """THE ASSERTION THAT MATTERS, and it is checked before anything else.

    Deliberately makes no claim about the row's status, the ledger or the
    reservation first, so a dispatcher driven through it fails because it
    EXECUTED under an expired certificate rather than because it forgot to
    write a log line. `test_expired_certificate_gate_rejects_a_reckless_
    dispatcher` depends on that ordering.
    """
    result = await dispatch_fn(conn, row, channel_map(provider), at=at)
    assert provider.invocations == [], (
        f"EXPIRED CERTIFICATE EXECUTED by {label}: the provider was called "
        f"{len(provider.invocations)} time(s) for a certificate valid "
        f"[{row.cert_valid_from.isoformat()}, {row.cert_valid_until.isoformat()}] "
        f"at {at.isoformat()}. Stale authorisation is the whole reason certificate "
        "windows exist, and executing it silently corrupts the propensity log."
    )
    return result


async def _expired_row(conn: Any, *, minutes_past: int = 20) -> tuple[Any, UUID, UUID]:
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    await commit_decision(conn, make_request(claim_id, cycle_id))
    clock = FrozenClock(T0 + timedelta(minutes=minutes_past))
    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=TEST_LEASE)
    assert len(rows) == 1
    return rows[0], claim_id, cycle_id


async def test_expired_certificate_never_executes(conn: Any) -> None:
    """Nothing reaches the provider once the window has closed.

    Not "the row is marked cancelled" - that is the next test. This one is
    about the customer's phone not ringing.
    """
    row, claim_id, _cycle = await _expired_row(conn)
    provider = FakeProvider()
    at = T0 + timedelta(minutes=20)

    assert not row.certificate_covers(at)
    result = await assert_refuses_expired_certificate(
        dispatch, conn, row, provider, at, label="the real dispatcher"
    )
    assert result.outcome is DispatchOutcome.CERT_EXPIRED
    assert provider.effects == {}

    # One minute inside the window it does execute, so the refusal above is
    # the window doing its job rather than the dispatcher being broken.
    fresh_row, fresh_claim, _ = await _expired_row(conn, minutes_past=0)
    fresh_provider = FakeProvider()
    ok = await dispatch(conn, fresh_row, channel_map(fresh_provider), at=T0)
    assert ok.outcome is DispatchOutcome.SENT
    assert len(fresh_provider.invocations) == 1
    assert fresh_claim != claim_id


async def test_expired_certificate_cancels_and_requeues(conn: Any) -> None:
    """Cancel the row, record the abandonment, free the budget, re-decide.

    The claim goes back to the Allocator rather than executing, so the next
    cycle gives it a fresh certificate and a FRESH PROPENSITY. That is what
    keeps the M11 log describing actions that could actually have been taken.
    """
    row, claim_id, cycle_id = await _expired_row(conn)
    provider = FakeProvider()
    at = T0 + timedelta(minutes=20)

    before = await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)
    result = await dispatch(conn, row, channel_map(provider), at=at)

    assert result.outcome is DispatchOutcome.CERT_EXPIRED
    assert result.requeued is True

    settled = await fetch_row(conn, row.id)
    assert settled.status is OutboxStatus.CANCELLED
    assert settled.last_error == "CERT_EXPIRED"

    entries = await conn.fetch(
        """
        SELECT event_type, body FROM decision_ledger
         WHERE claim_id = $1 AND event_type = 'abandoned_unexecuted'
        """,
        claim_id,
    )
    assert len(entries) == 1, "no ABANDONED_UNEXECUTED entry was appended"

    # The budget was given back rather than consumed by an action that never ran.
    after = await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)
    assert after == before + cost_of(ActionType.SMS).contact
    live = await reservations.live_for(conn, claim_id)
    assert live == []

    # And nothing is left claimable, so no worker picks it up and tries again.
    assert await claim_batch(conn, "w2", 10, at=at, lease=TEST_LEASE) == []


# ===========================================================================
# 7
# ===========================================================================
async def test_reservation_released_on_terminal_and_on_timeout(conn: Any) -> None:
    """Both release paths, because forgetting either one starves the portfolio.

    A leaked reservation produces no error and no alert. The cycle simply has
    less budget than it believes, every cycle, and the cause is invisible in
    every dashboard - which is why the expiry sweep exists as well as the
    terminal release.
    """
    cycle_id = uuid4()
    await reservations.declare_caps(conn, cycle_id, {BudgetKey.CONTACT: 10})

    # -- the two tiers -----------------------------------------------------
    near, far = uuid4(), uuid4()
    await insert_claim(conn, near)
    await insert_claim(conn, far)

    near_key = "near-key"
    far_key = "far-key"
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=near,
        subject_token=TOKEN,
        cost=CostVector(contact=1),
        idempotency_key=near_key,
        at=T0,
        horizon=timedelta(minutes=5),
    )
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=far,
        subject_token=TOKEN,
        cost=CostVector(contact=1),
        idempotency_key=far_key,
        at=T0,
        horizon=timedelta(days=3),
    )

    # A near-horizon decision holds the budget; a far one only signals demand.
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 9
    assert await reservations.pipeline_demand(conn, cycle_id) == {BudgetKey.CONTACT: 1}

    # The soft one hardens at wake, and only then does it cost anything.
    await reservations.harden(conn, far_key, T0 + timedelta(days=3))
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 8
    assert await reservations.pipeline_demand(conn, cycle_id) == {}

    # -- release on terminal ------------------------------------------------
    await transition(conn, near, frm=ClaimState.PLANNED, to=ClaimState.IN_TREATMENT, at=T0)
    await transition(conn, near, frm=ClaimState.IN_TREATMENT, to=ClaimState.FORBORNE, at=T0)
    freed = await release_on_terminal(conn, near, at=T0, state=ClaimState.FORBORNE)
    assert freed == 1
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 9

    # -- release on timeout -------------------------------------------------
    stale = uuid4()
    await insert_claim(conn, stale)
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=stale,
        subject_token=TOKEN,
        cost=CostVector(contact=1),
        idempotency_key="stale-key",
        at=T0,
        horizon=timedelta(minutes=1),
        ttl=timedelta(hours=2),
    )
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 8

    # Not yet: the sweep must not take a reservation that is still valid.
    assert await reservations.expire_stale(conn, T0 + timedelta(hours=1)) == 0
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 8

    # Only the stale one. The far reservation was hardened at wake and its
    # expiry moved with it, so a sweep three hours after T0 does not touch it.
    swept = await reservations.expire_stale(conn, T0 + timedelta(hours=3))
    assert swept == 1, "the sweep took a reservation that had just been hardened"
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 9

    # A consumed reservation is NOT credited back - it was genuinely spent.
    spent = uuid4()
    await insert_claim(conn, spent)
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=spent,
        subject_token=TOKEN,
        cost=CostVector(contact=1),
        idempotency_key="spent-key",
        at=T0,
        horizon=timedelta(minutes=1),
    )
    await reservations.consume(conn, "spent-key", T0)
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 8

    # The cap refuses rather than overdrawing (GI-3).
    overdraw = uuid4()
    await insert_claim(conn, overdraw)
    with pytest.raises(reservations.BudgetExhausted):
        await reservations.reserve(
            conn,
            cycle_id=cycle_id,
            claim_id=overdraw,
            subject_token=TOKEN,
            cost=CostVector(contact=99),
            idempotency_key="overdraw-key",
            at=T0,
            horizon=timedelta(minutes=1),
        )


# ===========================================================================
# 8
# ===========================================================================
async def test_illegal_fsm_transition_rejected(conn: Any) -> None:
    """M1's table is the only table. The database applies it, it does not
    reinvent it."""
    claim_id = uuid4()
    await insert_claim(conn, claim_id, state=ClaimState.PLANNED)

    with pytest.raises(IllegalTransition):
        await transition(conn, claim_id, frm=ClaimState.PLANNED, to=ClaimState.RECOVERED, at=T0)
    assert await current_state(conn, claim_id) is ClaimState.PLANNED

    await transition(conn, claim_id, frm=ClaimState.PLANNED, to=ClaimState.IN_TREATMENT, at=T0)
    await transition(conn, claim_id, frm=ClaimState.IN_TREATMENT, to=ClaimState.FORBORNE, at=T0)

    # FORBORNE is absorbing. No expected-value argument reopens hardship.
    for target in (ClaimState.IN_TREATMENT, ClaimState.WRITTEN_OFF, ClaimState.RECOVERED):
        with pytest.raises(IllegalTransition, match="absorbing"):
            await transition(conn, claim_id, frm=ClaimState.FORBORNE, to=target, at=T0)
    assert await current_state(conn, claim_id) is ClaimState.FORBORNE

    # A stale `frm` is a concurrent mover, not a silent overwrite.
    other = uuid4()
    await insert_claim(conn, other, state=ClaimState.IN_TREATMENT)
    with pytest.raises(ConcurrentTransition):
        await transition(conn, other, frm=ClaimState.PLANNED, to=ClaimState.IN_TREATMENT, at=T0)
    assert await current_state(conn, other) is ClaimState.IN_TREATMENT

    with pytest.raises(ClaimNotFound):
        await transition(conn, uuid4(), frm=ClaimState.PLANNED, to=ClaimState.IN_TREATMENT, at=T0)

    # And commit_decision inherits all of it.
    cycle_id = uuid4()
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    with pytest.raises(IllegalTransition):
        await commit_decision(
            conn,
            replace(
                make_request(claim_id, cycle_id),
                from_state=ClaimState.FORBORNE,
                to_state=ClaimState.IN_TREATMENT,
            ),
        )


# ===========================================================================
# 9
# ===========================================================================
async def test_gateway_own_retry_counted_against_network_budget(conn: Any) -> None:
    """Their attempts count against our cap, because the penalty is ours.

    The gateway re-presents on its own schedule after a failure and does not
    ask. A counter that only knows about ARC's retries is wrong in the
    direction that gets the merchant fined: the cap is exceeded without ARC
    ever having issued the excess attempt.
    """
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    instrument = "card_tok_7f3a"
    cap = 4

    # Two of ours.
    for index in range(2):
        assert await reservations.record_network_attempt(
            conn,
            instrument_ref=instrument,
            claim_id=claim_id,
            subject_token=TOKEN,
            rail=Rail.CARD.value,
            attempted_at=T0 - timedelta(days=index + 1),
            initiated_by="arc",
            attempt_ref=f"arc-key-{index}",
        )

    counted = await reservations.network_attempts_in_window(conn, instrument, T0)
    assert counted == 2
    assert reservations.within_network_cap(counted, cap)

    # Two of theirs, which we never issued.
    for index in range(2):
        assert await reservations.record_network_attempt(
            conn,
            instrument_ref=instrument,
            claim_id=claim_id,
            subject_token=TOKEN,
            rail=Rail.CARD.value,
            attempted_at=T0 - timedelta(hours=index + 1),
            initiated_by="gateway",
            attempt_ref=f"gw-event-{index}",
        )

    counted = await reservations.network_attempts_in_window(conn, instrument, T0)
    assert counted == 4, "gateway attempts are not being counted"
    split = await reservations.network_attempts_by_initiator(conn, instrument, T0)
    assert split == {"arc": 2, "gateway": 2}

    # The cap is now reached, and it was reached by attempts we did not make.
    assert not reservations.within_network_cap(counted, cap), (
        "the cap allows a fifth attempt because it is only counting our own"
    )

    # A redelivered gateway webhook counts once, because they redeliver by design.
    assert not await reservations.record_network_attempt(
        conn,
        instrument_ref=instrument,
        claim_id=claim_id,
        subject_token=TOKEN,
        rail=Rail.CARD.value,
        attempted_at=T0 - timedelta(hours=1),
        initiated_by="gateway",
        attempt_ref="gw-event-0",
    )
    assert await reservations.network_attempts_in_window(conn, instrument, T0) == 4

    # The window is half-open and rolls forward.
    assert (
        await reservations.network_attempts_in_window(
            conn, instrument, T0, window=timedelta(hours=3)
        )
        == 2
    )


# ===========================================================================
# Proving the gate is not vacuously green
# ===========================================================================
async def test_expired_certificate_gate_rejects_a_reckless_dispatcher(conn: Any) -> None:
    """The falsifiability check.

    A dispatcher that skips the window check is exactly what "it is only four
    minutes past, just send it" produces. It must fail, and it must fail on the
    EXPIRED-CERTIFICATE assertion rather than on a missing ledger entry or an
    unreleased reservation - those are consequences, and a suite that caught
    only them would pass a dispatcher that executed stale authorisation and
    happened to tidy up afterwards.
    """
    row, _claim, _cycle = await _expired_row(conn)
    at = T0 + timedelta(minutes=20)

    async def reckless_dispatch(connection, outbox_row, channels, *, at, ledger=None):
        """Gate touchpoint 3, removed. Everything else identical."""
        channel = channels[outbox_row.channel]
        return await channel.send(outbox_row.payload, outbox_row.idempotency_key)

    provider = FakeProvider()
    with pytest.raises(AssertionError) as caught:
        await assert_refuses_expired_certificate(
            reckless_dispatch, conn, row, provider, at, label="a reckless dispatcher"
        )
    message = str(caught.value)
    assert "EXPIRED CERTIFICATE EXECUTED" in message, (
        f"the reckless dispatcher failed on the wrong assertion:\n{message}"
    )
    assert "propensity log" in message
    assert len(provider.invocations) == 1, "the reckless dispatcher did reach the provider"

    # The real dispatcher passes the identical assertion, so it is not
    # unpassable.
    fresh_row, _, _ = await _expired_row(conn)
    await assert_refuses_expired_certificate(
        dispatch, conn, fresh_row, FakeProvider(), at, label="the real dispatcher"
    )


async def test_commit_refuses_an_action_without_a_valid_allow(conn: Any) -> None:
    """GI-1 at the executor boundary: no effect without a live certificate."""
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)

    blocked = make_certificate(claim_id, ActionType.SMS, decision=Verdict.BLOCK)
    with pytest.raises(UncertifiedAction, match="not ALLOW"):
        await commit_decision(conn, make_request(claim_id, cycle_id, certificate=blocked))

    wrong_action = make_certificate(claim_id, ActionType.EMAIL)
    with pytest.raises(UncertifiedAction, match="authorises"):
        await commit_decision(
            conn,
            replace(
                make_request(claim_id, cycle_id, certificate=wrong_action),
                action=ActionType.SMS,
            ),
        )

    # A certificate that will already have expired at the planned moment is
    # refused at commit rather than being queued to fail later.
    stale = make_certificate(
        claim_id,
        ActionType.SMS,
        valid_from=T0 - timedelta(hours=2),
        valid_until=T0 - timedelta(hours=1),
    )
    with pytest.raises(UncertifiedAction, match="does not cover"):
        await commit_decision(conn, make_request(claim_id, cycle_id, certificate=stale))

    # Nothing was written by any of the three refusals.
    assert await current_state(conn, claim_id) is ClaimState.PLANNED
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM budget_reservations WHERE claim_id = $1", claim_id
        )
        == 0
    )


async def test_commit_is_idempotent_and_atomic_under_repetition(conn: Any) -> None:
    """A retried commit produces one row, one reservation, one held budget."""
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    request = make_request(claim_id, cycle_id)

    first = await commit_decision(conn, request)
    assert first.outbox_id is not None
    assert first.already_enqueued is False
    remaining_after_first = await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)

    # The FSM edge is gone now, so a naive retry fails loudly rather than
    # double-reserving. That is the correct outcome: the claim moved.
    with pytest.raises(ConcurrentTransition):
        await commit_decision(conn, request)

    # Enqueue itself is idempotent, which is what an Inngest step replay needs.
    assert (
        await enqueue(
            conn,
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action_type=ActionType.SMS,
            channel="sms",
            payload={"template": "x"},
            certificate_id=request.certificate.certificate_id,
            cert_valid_from=request.certificate.valid_from,
            cert_valid_until=request.certificate.valid_until,
            not_before=T0,
        )
        is None
    )
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 1
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == remaining_after_first

    stored = await by_idempotency_key(conn, first.idempotency_key)
    assert stored is not None and stored.claim_id == claim_id


async def test_retryable_error_reschedules_and_permanent_error_goes_to_dlq(
    conn: Any,
) -> None:
    """Retry the transient, bury the permanent, and free the budget either way."""
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(conn, cycle_id, DEFAULT_CAPS)
    await commit_decision(conn, make_request(claim_id, cycle_id))
    clock = FrozenClock(T0)

    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=TEST_LEASE)
    provider = FakeProvider(fail_with=RetryableError("provider 500"), fail_times=1)
    result = await dispatch(conn, rows[0], channel_map(provider), at=clock.now())
    assert result.outcome is DispatchOutcome.RESCHEDULED

    row = await fetch_row(conn, rows[0].id)
    assert row.status is OutboxStatus.PENDING
    assert row.not_before > T0, "a rescheduled row must back off"
    # Still reserved: the action has not been abandoned, only delayed. An SMS
    # costs a contact slot AND messaging spend, so it holds two dimensions.
    dimensions = sum(1 for amount in cost_of(ActionType.SMS).as_tuple() if amount)
    assert len(await reservations.live_for(conn, claim_id)) == dimensions == 2

    # Second attempt succeeds, and the key is unchanged across both.
    #
    # The advance is small on purpose. THE CERTIFICATE WINDOW BOUNDS THE WHOLE
    # RETRY CHAIN: backing off past `cert_valid_until` does not produce a late
    # send, it produces a cancellation and a re-decision. That is the correct
    # behaviour and it is asserted below, so the retry here stays inside the
    # window the Gate actually issued.
    clock.advance(timedelta(minutes=1))
    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=TEST_LEASE)
    assert rows[0].idempotency_key == row.idempotency_key
    assert rows[0].attempts == 2
    ok = await dispatch(conn, rows[0], channel_map(provider), at=clock.now())
    assert ok.outcome is DispatchOutcome.SENT
    assert len(provider.invocations) == 2
    assert provider.effects == {row.idempotency_key: 1}

    # Permanent failure on a different claim goes straight to the DLQ.
    dead_claim = uuid4()
    await insert_claim(conn, dead_claim)
    await commit_decision(conn, make_request(dead_claim, cycle_id))
    dead_rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=TEST_LEASE)
    dead_provider = FakeProvider(fail_with=PermanentError("number disconnected"), fail_times=99)
    dead = await dispatch(conn, dead_rows[0], channel_map(dead_provider), at=clock.now())
    assert dead.outcome is DispatchOutcome.DEAD
    assert (await fetch_row(conn, dead_rows[0].id)).status is OutboxStatus.DEAD
    assert await reservations.live_for(conn, dead_claim) == [], (
        "a dead dispatch must free its budget, or the cycle silently shrinks"
    )

    # A backoff that would land outside the certificate window does not send
    # late - it cancels and re-decides. The retry budget is bounded by the
    # authorisation, not the other way round.
    late_claim = uuid4()
    await insert_claim(conn, late_claim)
    await commit_decision(conn, make_request(late_claim, cycle_id))
    late_rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=TEST_LEASE)
    late_provider = FakeProvider()
    late = await dispatch(
        conn, late_rows[0], channel_map(late_provider), at=T0 + timedelta(minutes=45)
    )
    assert late.outcome is DispatchOutcome.CERT_EXPIRED
    assert late_provider.invocations == []


async def test_lease_outlasts_the_channel_timeout() -> None:
    """A lease shorter than the provider call hands the row to a second worker.

    At that point the only thing between the customer and two messages is the
    provider honouring the key, which is a second line of defence rather than
    the guarantee. So the default is asserted to have real headroom.
    """
    from arc.conductor.outbox import DEFAULT_LEASE

    plausible_channel_timeout = timedelta(seconds=30)
    assert 3 * plausible_channel_timeout <= DEFAULT_LEASE, (
        f"the default lease is {DEFAULT_LEASE}, which leaves no headroom over a "
        f"{plausible_channel_timeout} channel call"
    )


async def test_m9_report_card(pool: asyncpg.Pool, capsys: pytest.CaptureFixture[str]) -> None:
    """Run the concurrency scenario once more and print what it did."""
    started = time.perf_counter()
    provider, statuses = await _run_concurrency_round(pool, rows=400, workers=20, round_no=99)
    elapsed = time.perf_counter() - started

    lines = [
        "",
        "=" * 68,
        "M9 CONDUCTOR - 20 workers, one queue",
        "=" * 68,
        "  rows committed                 400",
        f"  wall time                      {elapsed:.2f}s",
        "",
        "  DISPATCH",
        f"    provider invocations         {len(provider.invocations)}",
        f"    distinct idempotency keys    {len(set(provider.invocations))}",
        f"    duplicate invocations        {provider.duplicate_invocations}",
        f"    provider-side deduplications {sum(n - 1 for n in provider.effects.values())}",
        "",
        "  OUTBOX",
        *[f"    {name:<28} {count}" for name, count in sorted(statuses.items())],
        "",
        "  GUARANTEES",
        "    exactly-once state transition  by the transaction",
        "    at-least-once dispatch         by lease and retry",
        "    effectively-once effect        by the idempotency key",
        "    exactly-once delivery          not claimed",
        "=" * 68,
    ]
    with capsys.disabled():
        print("\n".join(lines))
