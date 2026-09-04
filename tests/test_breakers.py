"""M13 acceptance gate - kill switch, circuit breakers, erasure.

    test_shadow_mode_schedules_nothing
    test_freeze_releases_reservations
    test_held_items_invalidated_not_executed_on_resume
    test_resume_ramp_5_25_60_100
    test_complaint_spike_trips_to_shadow
    test_veto_rate_above_2pct_trips_CB_VETO
    test_volume_surge_defers_excess

Plus the erasure orchestration M5 left unbuilt, whose end-to-end property is
M2's: the decision ledger chain must still verify after a subject's data has
been destroyed.

THE SUITE IS NOT VACUOUSLY GREEN. `test_a_resume_that_executes_held_work_is_
caught` plants a resume path that dispatches held items instead of returning
them, and asserts the check that matters fires - a held idempotency key
appearing as a sent outbox row - rather than some incidental difference in a
report object.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from arc.allocator.budgets import BudgetKey, cost_of
from arc.conductor import reservations
from arc.conductor.breakers import (
    SELF_MONITORING,
    SPECS,
    BreakerId,
    Comparison,
    Reading,
    evaluate,
    evaluate_all,
    evaluate_and_apply,
)
from arc.conductor.erasure import ErasureIncomplete, assert_swept, erase_subject, report_for
from arc.conductor.kill_switch import (
    RAMP,
    HeldWorkExecuted,
    Mode,
    ShadowIntent,
    advance_ramp,
    assert_nothing_executed,
    current_mode,
    freeze,
    held_keys,
    record_shadow_intent,
    resume,
    set_mode,
)
from arc.conductor.outbox import OutboxStatus, enqueue, idempotency_key, mark
from arc.core.ids import subject_token
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.events.names import EventName
from arc.events.runs import run_status, start_run
from arc.gate.context import ACTION_CHANNEL
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType
from arc.ledger.pii_guard import PIIDetected
from arc.ledger.subject_store import SubjectStore
from tests.conductor_db import scratch_database

PEPPER = b"m13-acceptance-gate-pepper-00000"
TOKEN = subject_token("+919812345671", pepper=PEPPER)
OTHER = subject_token("+919812345672", pepper=PEPPER)
T0 = datetime(2026, 4, 6, 9, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    """A scratch database for this module alone. Per-caller, per the M10 rule."""
    try:
        yield from scratch_database("breakers")
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover
        pytest.skip(f"postgres unavailable: {exc}")


@pytest.fixture
async def conn(dsn: str) -> AsyncIterator[asyncpg.Connection]:
    connection = await asyncpg.connect(dsn)
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


async def insert_claim(conn: Any, claim_id: UUID, *, token: str = TOKEN) -> None:
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
        129_900,
        1_500_000,
        ClaimType.CARD_DECLINE.value,
        Rail.CARD.value,
        T0 - timedelta(days=1),
        b"\x00" * 32,
        ClaimState.IN_TREATMENT.value,
    )


async def schedule(
    conn: Any,
    claim_id: UUID,
    cycle_id: UUID,
    *,
    token: str = TOKEN,
    action: ActionType = ActionType.SMS,
    reserve: bool = True,
) -> str:
    """One decision, reserved and enqueued, exactly as the Conductor would."""
    certificate_id = uuid4()
    key = idempotency_key(claim_id, action, cycle_id, certificate_id)
    if reserve:
        await reservations.reserve(
            conn,
            cycle_id=cycle_id,
            claim_id=claim_id,
            subject_token=token,
            cost=cost_of(action),
            idempotency_key=key,
            at=T0,
            horizon=timedelta(minutes=5),
        )
    await enqueue(
        conn,
        claim_id=claim_id,
        subject_token=token,
        cycle_id=cycle_id,
        action_type=action,
        channel=ACTION_CHANNEL[action].value,
        payload={},
        certificate_id=certificate_id,
        cert_valid_from=T0 - timedelta(minutes=10),
        cert_valid_until=T0 + timedelta(minutes=10),
        not_before=T0,
    )
    return key


async def declare(conn: Any, cycle_id: UUID, contact: int = 50) -> None:
    await reservations.declare_caps(
        conn, cycle_id, {BudgetKey.CONTACT: contact, BudgetKey.RUPEE: 500_000}
    )


# ---------------------------------------------------------------------------
# Gate test 1 - SHADOW
# ---------------------------------------------------------------------------
async def test_shadow_mode_schedules_nothing(conn: asyncpg.Connection) -> None:
    """No reservation, no outbox row, no backlog. Only a recorded intent.

    A shadow mode that queued work would accumulate a backlog for however long
    it was switched on, and the moment it switched off that backlog would go
    out at once - to people whose circumstances had moved on by exactly that
    long. So SHADOW writes nothing a worker can find.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare(conn, cycle_id)

    state = await set_mode(conn, Mode.SHADOW, at=T0, changed_by="test", reason="observing")
    assert state.mode is Mode.SHADOW
    assert not state.admits_new_work
    assert not state.reserves_budget
    assert not state.dispatches

    seq = await record_shadow_intent(
        conn,
        ShadowIntent(claim_id=claim_id, subject_token=TOKEN, action=ActionType.VOICE_CALL, at=T0),
    )
    assert seq > 0, "the intent was not recorded at all; shadow must still observe"

    # NOTHING SCHEDULED, NOTHING RESERVED.
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM budget_reservations WHERE claim_id = $1", claim_id
        )
        == 0
    )
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 50, (
        "shadow mode consumed budget"
    )

    # The intent IS on the record, marked as never scheduled.
    body = await conn.fetchval(
        "SELECT body FROM decision_ledger WHERE claim_id = $1 ORDER BY seq DESC LIMIT 1",
        claim_id,
    )
    assert "shadow" in str(body) and "voice_call" in str(body)
    assert '"scheduled": false' in str(body).replace("False", "false").lower()


async def test_drain_still_dispatches_what_is_already_authorised(
    conn: asyncpg.Connection,
) -> None:
    """DRAIN is the one mode that keeps dispatching, and that is its meaning.

    Nothing new is admitted, but work already certified and reserved finishes.
    Confusing DRAIN with FREEZE throws away authorised work that was about to
    complete, which is a worse outcome than either mode intends.
    """
    state = await set_mode(conn, Mode.DRAIN, at=T0, changed_by="test", reason="deploying")
    assert not state.admits_new_work
    assert state.dispatches, "DRAIN stopped dispatching, which is what FREEZE is for"
    assert not state.reserves_budget


# ---------------------------------------------------------------------------
# Gate tests 2 and 3 - FREEZE and resume
# ---------------------------------------------------------------------------
async def test_freeze_releases_reservations(conn: asyncpg.Connection) -> None:
    """A frozen system holding budget starves the portfolio for nothing.

    Safe to release precisely because long-horizon work never held a hard
    reservation in the first place - the two-tier rule at M9 paying off here.
    """
    cycle_id = uuid4()
    await declare(conn, cycle_id, contact=50)
    claims = [uuid4() for _ in range(3)]
    for claim_id in claims:
        await insert_claim(conn, claim_id)
        await schedule(conn, claim_id, cycle_id)

    before = await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)
    assert before == 47, f"three contacts should have been held, remaining is {before}"

    report = await freeze(conn, at=T0, changed_by="operator", reason="complaint spike")

    assert report.held == 3
    assert report.outbox_cancelled == 3
    assert report.reservations_released >= 3
    after = await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT)
    assert after == 50, f"the freeze did not give the budget back; remaining is {after}"

    assert (await current_mode(conn)).mode is Mode.FREEZE
    # Nothing is left for a worker to pick up.
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE status = 'pending'") == 0, (
        "a pending row survived the freeze"
    )


async def test_held_items_invalidated_not_executed_on_resume(
    conn: asyncpg.Connection,
) -> None:
    """The world changed during the freeze. Held decisions are re-decided.

    Two independent reasons, either sufficient. The freeze happened because
    something was wrong, so executing decisions made before it acts on the very
    state that made freezing necessary. And the certificates have expired
    anyway - a freeze long enough to matter outlasts a certificate window.
    """
    cycle_id = uuid4()
    await declare(conn, cycle_id)
    claims = [uuid4() for _ in range(4)]
    for claim_id in claims:
        await insert_claim(conn, claim_id)
        await schedule(conn, claim_id, cycle_id)

    await freeze(conn, at=T0, changed_by="operator", reason="opt-out spike")
    keys = await held_keys(conn)
    assert len(keys) == 4

    report = await resume(conn, at=T0 + timedelta(hours=6), changed_by="operator")

    assert report.invalidated == 4
    assert report.requeued == 4
    assert report.executed == 0, "the resume path reports having executed something"

    # THE CHECK THAT MATTERS: no held key was dispatched.
    await assert_nothing_executed(conn, keys)

    # Each one went back to the Allocator, on the record.
    for claim_id in claims:
        body = await conn.fetchval(
            """
            SELECT body FROM decision_ledger
             WHERE claim_id = $1 AND event_type = 'abandoned_unexecuted'
             ORDER BY seq DESC LIMIT 1
            """,
            claim_id,
        )
        assert body is not None, f"{claim_id} was not requeued"
        assert "RESUME_HELD_INVALIDATED" in str(body)
        assert "requeued_for_allocation" in str(body)

    assert await held_keys(conn) == [], "held work was not released after resume"
    assert (await current_mode(conn)).mode is Mode.NORMAL


async def test_a_resume_that_executes_held_work_is_caught(conn: asyncpg.Connection) -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    A resume path that dispatches held items instead of returning them is
    planted here, through the real outbox. The assertion that catches it is the
    one that matters - a held idempotency key appearing as a sent row - not a
    difference in some report object the bad path would also be writing.
    """
    cycle_id = uuid4()
    await declare(conn, cycle_id)
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    await schedule(conn, claim_id, cycle_id)

    await freeze(conn, at=T0, changed_by="operator", reason="test")
    keys = await held_keys(conn)
    assert keys, "nothing was held, so the plant would prove nothing"

    # The defect: resurrect the held row and dispatch it.
    row_id = await conn.fetchval(
        "SELECT outbox_id FROM held_work WHERE idempotency_key = $1", keys[0]
    )
    await conn.execute("UPDATE outbox SET status = 'pending' WHERE id = $1", row_id)
    await mark(conn, row_id, OutboxStatus.SENT, error="resume dispatched a held row")

    with pytest.raises(HeldWorkExecuted) as caught:
        await assert_nothing_executed(conn, keys)

    message = str(caught.value)
    assert "held by a freeze was dispatched" in message
    assert keys[0] in message, "the failure does not name which item was executed"
    assert "certificates have expired" in message, (
        "the failure explains that work was dispatched but not why that is wrong"
    )


async def test_resume_ramp_5_25_60_100(conn: asyncpg.Connection) -> None:
    """Coming back at full volume is itself a volume surge.

    A burst on resume trips CB-VOLUME, which freezes the system again. The
    ramp is slow enough that the breaker sees a ramp rather than a spike.
    """
    assert RAMP == (0.05, 0.25, 0.60, 1.00)

    await freeze(conn, at=T0, changed_by="operator", reason="test")
    report = await resume(conn, at=T0 + timedelta(hours=1), changed_by="operator")
    assert report.ramp_step == 0

    trailing_median = 1_000
    observed: list[int] = []
    for step in range(len(RAMP)):
        state = await current_mode(conn)
        assert state.ramp_step == step
        observed.append(state.admission_cap(trailing_median))
        await advance_ramp(conn, at=T0 + timedelta(hours=2 + step))

    assert observed == [50, 250, 600, 1000], f"the ramp admitted {observed}"

    # At the top the ramp CLEARS rather than sticking at 100%, so a later
    # cycle is not still described as ramping.
    final = await current_mode(conn)
    assert final.ramp_step is None, "the ramp never finished"
    assert final.ramp_fraction == 1.0
    assert final.admission_cap(trailing_median) == trailing_median

    # And a frozen system admits nothing regardless of where the ramp sits.
    await set_mode(conn, Mode.FREEZE, at=T0, changed_by="t", reason="t", ramp_step=3)
    assert (await current_mode(conn)).admission_cap(trailing_median) == 0


# ---------------------------------------------------------------------------
# Gate tests 5, 6, 7 - the breakers
# ---------------------------------------------------------------------------
def test_all_ten_breakers_exist_with_a_rationale() -> None:
    """Ten, from spec 7.6, three of which watch the system's own machinery."""
    assert len(SPECS) == 10, f"{len(SPECS)} breakers, expected ten"
    assert set(SPECS) == set(BreakerId)
    assert {
        BreakerId.VETO,
        BreakerId.DEGRADED,
        BreakerId.COHORT_BLIND,
    } == SELF_MONITORING
    for spec in SPECS.values():
        assert spec.rationale.strip(), f"{spec.breaker_id} has no rationale"
        assert spec.threshold > 0


async def test_complaint_spike_trips_to_shadow(conn: asyncpg.Connection) -> None:
    """Recovery bought with complaints is a cost deferred, not a win.

    Trips to SHADOW rather than off: shadow keeps L0-L5 running and the ledger
    filling, so the diagnosis of whatever tripped it is being recorded while it
    is tripped. A system that goes dark on detecting a problem destroys the
    evidence about the problem.
    """
    await set_mode(conn, Mode.NORMAL, at=T0, changed_by="test", reason="baseline")

    verdicts, newly = await evaluate_and_apply(
        conn,
        [
            # Trailing median 2.0 per thousand; observed 4.1 is above 1.5x.
            Reading(BreakerId.COMPLAINT, observed=4.1, baseline=2.0, sample=900),
            Reading(BreakerId.OPTOUT, observed=1.9, baseline=2.0, sample=900),
        ],
        at=T0,
    )

    assert BreakerId.COMPLAINT in newly
    assert BreakerId.OPTOUT not in newly, "a within-tolerance rate tripped"
    assert (await current_mode(conn)).mode is Mode.SHADOW, "the trip did not stop the system"

    # EVERY reading is evaluated - no short-circuit on the first trip. An
    # operator needs to know everything that is wrong, not the first thing.
    assert len(verdicts) == 2

    state = await conn.fetchrow(
        "SELECT tripped, observed, tripped_at, detail FROM breaker_state WHERE breaker_id = $1",
        BreakerId.COMPLAINT.value,
    )
    assert state["tripped"] and state["tripped_at"] == T0
    assert "baseline" in state["detail"], "the trip did not record what it was judged against"


async def test_veto_rate_above_2pct_trips_CB_VETO(conn: asyncpg.Connection) -> None:
    """A self-monitoring breaker. It measures the machinery, not the customer.

    `project` and `certify` share one registry and one evaluator, so only
    RUNTIME-class rules can fire after allocation. A veto rate above two
    percent does not mean the Gate is strict - it means the eligibility
    projection is broken and the Allocator is optimising over a candidate set
    it is not allowed to have.
    """
    await set_mode(conn, Mode.NORMAL, at=T0, changed_by="test", reason="baseline")

    below = evaluate(Reading(BreakerId.VETO, observed=0.015, sample=500))
    assert not below.tripped, "1.5% veto rate tripped a 2% breaker"

    _, newly = await evaluate_and_apply(
        conn, [Reading(BreakerId.VETO, observed=0.031, sample=500)], at=T0
    )
    assert BreakerId.VETO in newly
    assert (await current_mode(conn)).mode is Mode.SHADOW

    verdict = evaluate(Reading(BreakerId.VETO, observed=0.031, sample=500))
    assert verdict.self_monitoring, "CB-VETO is not marked as self-monitoring"
    assert SPECS[BreakerId.VETO].comparison is Comparison.ABSOLUTE_SHARE, (
        "the veto rate is an absolute share, not a multiple of a baseline"
    )


def test_cohort_blindspot_surfaces_the_outage_m6_could_not_detect() -> None:
    """CB-COHORT-BLIND is where an honest limitation becomes a measurement.

    M6's hierarchical back-off cannot always find power - the forty-minute
    outage on a thin issuer is the case it legitimately misses. This breaker is
    where that miss SURFACES, rather than passing as a clean NORMAL. An
    unmeasured blind spot is a defect; a measured one is a known limitation,
    and this is the difference between them.
    """
    spec = SPECS[BreakerId.COHORT_BLIND]
    assert spec.self_monitoring
    assert spec.comparison is Comparison.ABSOLUTE_SHARE
    assert spec.threshold == 0.40

    fine = evaluate(Reading(BreakerId.COHORT_BLIND, observed=0.33, sample=400))
    assert not fine.tripped

    blind = evaluate(Reading(BreakerId.COHORT_BLIND, observed=0.52, sample=400))
    assert blind.tripped
    assert blind.self_monitoring
    assert "cohort_blindspot_share" in blind.detail


def test_a_ratio_breaker_without_a_baseline_does_not_trip() -> None:
    """A fresh deployment has no trailing median.

    Firing every ratio breaker on cycle one would make the system unusable at
    exactly the moment it is being watched most closely, so an absent baseline
    is "not evaluated" rather than "assume the worst".
    """
    verdict = evaluate(Reading(BreakerId.COMPLAINT, observed=99.0, baseline=None, sample=500))
    assert not verdict.tripped
    assert "no baseline" in verdict.detail


def test_a_tiny_sample_does_not_trip_a_rate() -> None:
    """One complaint in three contacts is 333 per thousand and means nothing."""
    verdict = evaluate(Reading(BreakerId.COMPLAINT, observed=333.0, baseline=2.0, sample=3))
    assert not verdict.tripped
    assert "below" in verdict.detail


async def test_volume_surge_defers_excess(conn: asyncpg.Connection) -> None:
    """A spike is either a bug upstream or a burst nobody authorised.

    The excess is deferred rather than sent: admission is capped at the ramp's
    share of the trailing median, and everything above that waits for the next
    cycle instead of going out at once.
    """
    await set_mode(conn, Mode.NORMAL, at=T0, changed_by="test", reason="baseline")
    trailing_median = 1_000

    _, newly = await evaluate_and_apply(
        conn,
        [Reading(BreakerId.VOLUME, observed=4_200, baseline=trailing_median, sample=4_200)],
        at=T0,
    )
    assert BreakerId.VOLUME in newly, "a 4.2x volume spike did not trip CB-VOLUME"
    assert (await current_mode(conn)).mode is Mode.SHADOW

    # Coming back, admission is capped and the excess is deferred, not dropped.
    await freeze(conn, at=T0 + timedelta(minutes=5), changed_by="op", reason="volume")
    await resume(conn, at=T0 + timedelta(hours=1), changed_by="op")

    state = await current_mode(conn)
    cap = state.admission_cap(trailing_median)
    assert cap == 50, f"first ramp step admitted {cap}, not 5% of the trailing median"

    offered = 4_200
    deferred = offered - cap
    assert deferred == 4_150, "the excess above the cap is not accounted for"
    # Deferred, not discarded: the cap rises on the next rung.
    await advance_ramp(conn, at=T0 + timedelta(hours=2))
    assert (await current_mode(conn)).admission_cap(trailing_median) == 250


async def test_breakers_do_not_re_trip_an_already_open_breaker(
    conn: asyncpg.Connection,
) -> None:
    """`tripped_at` marks the transition, not every evaluation since."""
    await set_mode(conn, Mode.NORMAL, at=T0, changed_by="test", reason="baseline")
    reading = Reading(BreakerId.CHANNEL_FAIL, observed=0.31, sample=500)

    _, first = await evaluate_and_apply(conn, [reading], at=T0)
    assert first == [BreakerId.CHANNEL_FAIL]

    _, second = await evaluate_and_apply(conn, [reading], at=T0 + timedelta(hours=6))
    assert second == [], "an already-open breaker reported as newly tripped"

    at = await conn.fetchval(
        "SELECT tripped_at FROM breaker_state WHERE breaker_id = $1",
        BreakerId.CHANNEL_FAIL.value,
    )
    assert at == T0, "the trip time moved on a later evaluation"

    # Clearing records when it cleared.
    await evaluate_and_apply(
        conn,
        [Reading(BreakerId.CHANNEL_FAIL, observed=0.01, sample=500)],
        at=T0 + timedelta(hours=8),
    )
    row = await conn.fetchrow(
        "SELECT tripped, cleared_at FROM breaker_state WHERE breaker_id = $1",
        BreakerId.CHANNEL_FAIL.value,
    )
    assert not row["tripped"] and row["cleared_at"] == T0 + timedelta(hours=8)


def test_every_reading_is_evaluated_not_short_circuited() -> None:
    """The same reasoning as the Gate evaluating all rules."""
    readings = [
        Reading(BreakerId.COMPLAINT, observed=99.0, baseline=1.0, sample=500),
        Reading(BreakerId.OPTOUT, observed=99.0, baseline=1.0, sample=500),
        Reading(BreakerId.DEGRADED, observed=0.9, sample=500),
    ]
    verdicts = evaluate_all(readings)
    assert len(verdicts) == 3
    assert all(v.tripped for v in verdicts)
    assert [v.breaker_id for v in verdicts] == [r.breaker_id for r in readings]


# ---------------------------------------------------------------------------
# Erasure - the orchestration M5 left unbuilt
# ---------------------------------------------------------------------------
async def _archive(conn: Any, token: str, *, count: int = 3) -> None:
    """Raw deliveries, archived before parse, tagged with the subject."""
    for index in range(count):
        await conn.execute(
            """
            INSERT INTO raw_events
                (source, payload_hash, signature, body, received_at,
                 signature_valid, event_id, subject_token)
            VALUES ($1, $2, $3, $4, $5, TRUE, $6, $7)
            """,
            "razorpay",
            bytes([index]) * 32,
            "sig",
            f'{{"contact":"+91981234567{index}","name":"Priya Sharma"}}'.encode(),
            T0 - timedelta(days=2),
            f"evt_{token[-6:]}_{index}",
            token,
        )


async def test_erasure_sweeps_every_store_and_leaves_the_chain_verifiable(
    conn: asyncpg.Connection,
) -> None:
    """The end-to-end property, and it is M2's: the chain still verifies.

    An immutable hash-chained ledger and a right to erasure are opposed only if
    the ledger contains personal data. It does not - the chain covers
    pseudonymous tokens and structured fields - so erasure destroys the key,
    the subject's rows become unreadable, and the chain stays intact because it
    never covered plaintext.
    """
    ledger = DecisionLedger()
    store = SubjectStore(ledger=ledger)
    cycle_id = uuid4()
    await declare(conn, cycle_id)

    claim_id, other_claim = uuid4(), uuid4()
    await insert_claim(conn, claim_id, token=TOKEN)
    await insert_claim(conn, other_claim, token=OTHER)

    await store.put(conn, TOKEN, {"name": "Priya Sharma", "phone": "+919812345671"})
    await store.put(conn, OTHER, {"name": "Rohan Mehta", "phone": "+919812345672"})
    await _archive(conn, TOKEN, count=3)
    await _archive(conn, OTHER, count=2)
    await schedule(conn, claim_id, cycle_id, token=TOKEN)
    await schedule(conn, other_claim, cycle_id, token=OTHER)

    run = await start_run(
        conn,
        function_id="salary-aligned-retry",
        at=T0,
        claim_id=claim_id,
        subject_token=TOKEN,
    )

    # A real chain to break. Without prior entries the tombstone would be the
    # first link and "the chain still verifies" would be trivially true.
    for target, token in ((claim_id, TOKEN), (other_claim, OTHER)):
        await ledger.append(
            conn,
            LedgerEntry(
                event_type=LedgerEventType.DECISION,
                occurred_at=T0,
                claim_id=target,
                subject_token=token,
                payload={"intended_action": ActionType.SMS.value, "pi_intended": 0.31},
            ),
        )

    head_before = await conn.fetchval("SELECT coalesce(max(seq), 0) FROM decision_ledger")
    assert head_before >= 1, "the fixture wrote no ledger entries to chain"
    assert await ledger.verify_chain(conn, 1, head_before)

    report = await erase_subject(
        conn, TOKEN, at=T0 + timedelta(hours=1), requested_by="role:data-protection-officer"
    )

    # 1. Subject store: key destroyed, rows unreadable.
    assert report.subject_refs_destroyed >= 1
    assert await store.is_shredded(conn, TOKEN)

    # 2. THE ARCHIVE. The store that gets forgotten.
    assert report.archive_rows_purged == 3
    assert (
        await conn.fetchval("SELECT count(*) FROM raw_events WHERE subject_token = $1", TOKEN)
    ) == 0
    bodies = await conn.fetch(
        "SELECT body FROM raw_events WHERE event_id LIKE $1", f"evt_{TOKEN[-6:]}%"
    )
    assert all(bytes(row["body"]) == b"" for row in bodies), (
        "the raw archive still holds the payload; it is a complete copy of what the "
        "subject store shredded, so an erasure that misses it has erased nothing"
    )

    # 3. Scheduled work stopped.
    assert report.outbox_rows_cancelled == 1
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM outbox WHERE subject_token = $1 AND status IN "
            "('pending','in_flight')",
            TOKEN,
        )
        == 0
    )

    # 4. The durable run cancelled, and the event M12 subscribes to emitted.
    assert report.runs_cancelled == 1
    status = await run_status(conn, run.run_id)
    assert status["status"] == "cancelled"
    assert status["cancelled_by"] == EventName.SUBJECT_ERASURE.value
    emitted = await conn.fetchval(
        "SELECT count(*) FROM durable_events WHERE name = $1 AND match_key = $2",
        EventName.SUBJECT_ERASURE.value,
        TOKEN,
    )
    assert emitted == 1, "the erasure event M12's cancelOn consumes was not emitted"

    # 5. THE TOMBSTONE, and THE CHAIN STILL VERIFIES. M2's property, end to end.
    head_after = await conn.fetchval("SELECT coalesce(max(seq), 0) FROM decision_ledger")
    assert head_after > head_before, "no tombstone was appended"
    assert await ledger.verify_chain(conn, 1, head_after), (
        "the ledger chain broke when a subject was erased. The chain covers "
        "pseudonymous tokens and structured fields only; if erasure can break it, "
        "something personal was hash-chained into it"
    )
    tombstone = await conn.fetchrow(
        """
        SELECT body FROM decision_ledger
         WHERE event_type = 'tombstone' AND subject_token = $1 ORDER BY seq DESC LIMIT 1
        """,
        TOKEN,
    )
    assert tombstone is not None, "the erasure was not recorded"
    assert "erasure_request" in str(tombstone["body"])
    assert "role:data-protection-officer" in str(tombstone["body"])

    # 6. The other subject is untouched. An erasure that swept a neighbour
    #    would be a data loss incident wearing a compliance label.
    assert not await store.is_shredded(conn, OTHER)
    assert (
        await conn.fetchval("SELECT count(*) FROM raw_events WHERE subject_token = $1", OTHER)
    ) == 2
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM outbox WHERE subject_token = $1 AND status = 'pending'",
            OTHER,
        )
        == 1
    )

    # 7. The register survives the data it describes - itself an obligation.
    records = await report_for(conn, TOKEN)
    assert len(records) == 1
    assert records[0]["completed_at"] is not None
    assert records[0]["tombstone_seq"] == report.tombstone_seq


async def test_erasure_verification_catches_a_missed_archive_row(
    conn: asyncpg.Connection,
) -> None:
    """The sweep is checked by reading the stores back, not by its own counts.

    A sweep with a wrong WHERE clause also reports a count. This asserts the
    verification reads the archive independently and refuses a partial erasure.
    """
    await conn.execute(
        """
        INSERT INTO raw_events
            (source, payload_hash, signature, body, received_at, signature_valid,
             event_id, subject_token)
        VALUES ('razorpay', $1, 'sig', $2, $3, TRUE, 'evt_missed', $4)
        """,
        b"\x09" * 32,
        b'{"name":"Priya Sharma"}',
        T0,
        TOKEN,
    )
    with pytest.raises(ErasureIncomplete) as caught:
        await assert_swept(conn, TOKEN)
    assert "raw archive" in str(caught.value)
    assert "erased nothing" in str(caught.value)


async def test_erasure_is_idempotent(conn: asyncpg.Connection) -> None:
    """A repeated request is honoured, not an error.

    A data principal who asks twice gets two records of having asked, and the
    second sweep finds nothing left to destroy. Failing the second request
    would leave a compliance obligation looking unmet.
    """
    store = SubjectStore(ledger=DecisionLedger())
    await store.put(conn, TOKEN, {"name": "Priya Sharma"})
    await _archive(conn, TOKEN, count=2)

    first = await erase_subject(conn, TOKEN, at=T0, requested_by="dpo")
    second = await erase_subject(conn, TOKEN, at=T0 + timedelta(days=1), requested_by="dpo-again")

    assert first.archive_rows_purged == 2
    assert second.archive_rows_purged == 0, "the second sweep found data the first missed"
    assert len(await report_for(conn, TOKEN)) == 2, "the repeat request was not recorded"
    assert await DecisionLedger().verify_chain(
        conn, 1, await conn.fetchval("SELECT coalesce(max(seq), 1) FROM decision_ledger")
    )


async def test_erasure_refuses_to_chain_the_requesters_own_pii(
    conn: asyncpg.Connection,
) -> None:
    """An erasure that recorded the requester's email would create a new
    erasure obligation in the one store that cannot honour one.

    The tombstone goes into the hash-chained ledger, so `requested_by` has to
    be a role or a pseudonymous operator reference. The PII write-guard refuses
    the append rather than trusting the caller, and because the whole sweep is
    one transaction the refusal rolls back the erasure rather than leaving a
    subject half-destroyed.
    """
    store = SubjectStore(ledger=DecisionLedger())
    await store.put(conn, TOKEN, {"name": "Priya Sharma"})
    await _archive(conn, TOKEN, count=2)

    with pytest.raises(PIIDetected) as caught:
        await erase_subject(conn, TOKEN, at=T0, requested_by="dpo@example.test")
    assert "email" in str(caught.value)

    # AND THE ROLLBACK HELD. A refused tombstone must not leave the archive
    # swept and the key destroyed with no record that it happened.
    assert not await store.is_shredded(conn, TOKEN), "the key was destroyed anyway"
    assert (
        await conn.fetchval("SELECT count(*) FROM raw_events WHERE subject_token = $1", TOKEN)
    ) == 2, "the archive was swept by an erasure that then failed to record itself"
    assert await report_for(conn, TOKEN) == []

    # The same request with a role reference succeeds.
    report = await erase_subject(conn, TOKEN, at=T0, requested_by="role:dpo")
    assert report.archive_rows_purged == 2
