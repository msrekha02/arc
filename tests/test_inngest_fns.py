"""M12 acceptance gate - durable functions.

    test_wake_re_gates_before_enqueue
    test_hardship_event_cancels_mid_sleep
    test_recovered_event_cancels_mid_sleep
    test_defer_bounded_to_3_hops_then_requeues
    test_ptp_wait_resolves_on_payment
    test_ptp_timeout_records_BROKEN_and_requeues
    test_ptp_tracker_does_not_choose_next_action
    test_soft_intent_hard_reserved_at_wake

TIME IS A PARAMETER HERE TOO. A three-day sleep is driven by a `ManualClock`
that `sleep_until` advances, so the run genuinely resumes at the later moment -
every rule the Gate evaluates on wake sees that moment - and the test takes
microseconds. Nothing in this file reads a wall clock, which is the same
constraint the rest of the repo runs under and the reason any of this is
replayable.

THE SUITE IS NOT VACUOUSLY GREEN. `test_a_function_that_skips_the_wake_gate_is_
caught` plants a durable function that enqueues without certifying and asserts
the check that matters catches it - the absence of a certificate on the outbox
row, not some incidental difference in the result object.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from arc.allocator.budgets import cost_of
from arc.conductor import reservations
from arc.conductor.outbox import OutboxStatus, idempotency_key
from arc.core.ids import subject_token
from arc.core.money import paise
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.events.bus import emit
from arc.events.names import (
    CANCEL_ON,
    EventName,
    IncompleteCancellation,
    assert_cancels_on_every_stop,
)
from arc.events.runs import run_status, start_run
from arc.gate.context import ACTION_CHANNEL, Channel, ConsentState, GateContext
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict
from arc.inngest_fns.gated_enqueue import (
    MAX_DEFER_HOPS,
    Outcome,
    gated_enqueue,
)
from arc.inngest_fns.ptp_tracker import (
    Promise,
    PromiseOutcome,
    PromiseResult,
    classify,
    promise_to_pay_tracker,
)
from arc.inngest_fns.runtime import ManualClock, RunCancelled, Step
from arc.inngest_fns.salary_retry import RetryRequest, salary_aligned_retry
from tests.conductor_db import scratch_database

PEPPER = b"m12-acceptance-gate-pepper-00000"
TOKEN = subject_token("+919812345670", pepper=PEPPER)
T0 = datetime(2026, 3, 16, 9, 0, tzinfo=UTC)
PAYDAY = T0 + timedelta(days=3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    """A scratch database for this module alone. Per-caller, per the M10 rule."""
    try:
        yield from scratch_database("inngest_fns")
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


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------
class ScriptedGate:
    """A Gate whose verdict per call is scripted, and which counts its calls.

    The call count is the point of several tests: `gatedEnqueue` must certify on
    EVERY wake, so a function that woke three times and certified twice has a
    fast path it must not have.
    """

    def __init__(self, script: list[tuple[Verdict, datetime | None]]) -> None:
        self.script = list(script)
        self.calls: list[tuple[ActionType, datetime]] = []

    def certify(self, ctx: GateContext, action: ActionType, at: datetime) -> Certificate:
        self.calls.append((action, at))
        verdict, until = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        return Certificate(
            certificate_id=uuid4(),
            decision=verdict,
            valid_from=at - timedelta(minutes=10),
            valid_until=at + timedelta(minutes=10),
            evaluated_rules=(),
            blocking_rule_ids=("TEST-RULE",) if verdict is not Verdict.ALLOW else (),
            defer_until=until,
            rule_registry_version="test-registry-1",
            action=action,
            issued_at=at,
            claim_id=ctx.claim_id,
        )


class Contexts:
    """A context source that rebuilds at the moment it is asked for.

    Deliberately not a cache. The whole contract of `gatedEnqueue` is that it
    fetches fresh state at the wake, and a source that returned a snapshot
    taken at plan time would defeat it while every test still passed.
    """

    def __init__(self, claim_id: UUID, *, missing: bool = False) -> None:
        self.claim_id = claim_id
        self.missing = missing
        self.asked_at: list[datetime] = []

    async def context_for(self, conn: Any, claim_id: UUID, *, at: datetime) -> GateContext | None:
        self.asked_at.append(at)
        if self.missing:
            return None
        return GateContext(
            claim_id=claim_id,
            subject_token=TOKEN,
            rail=Rail.CARD,
            claim_state=ClaimState.IN_TREATMENT,
            amount_paise=paise(129_900),
            consent=dict.fromkeys(Channel, ConsentState.GRANTED),
        )


async def insert_claim(
    conn: Any, claim_id: UUID, *, state: ClaimState = ClaimState.IN_TREATMENT
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
        TOKEN,
        129_900,
        1_500_000,
        ClaimType.CARD_DECLINE.value,
        Rail.CARD.value,
        T0 - timedelta(days=1),
        b"\x00" * 32,
        state.value,
    )


async def declare_budget(conn: Any, cycle_id: UUID, amount: int = 100) -> None:
    from arc.allocator.budgets import BudgetKey

    await reservations.declare_caps(
        conn, cycle_id, {BudgetKey.CONTACT: amount, BudgetKey.RUPEE: amount * 1000}
    )


# ---------------------------------------------------------------------------
# Gate test 1 - every wake re-certifies
# ---------------------------------------------------------------------------
async def test_wake_re_gates_before_enqueue(conn: asyncpg.Connection) -> None:
    """GATE TOUCHPOINT 4, at the moment of the wake and not of the plan.

    The certificate that authorised the plan expired while the function slept.
    That is what certificate windows are for, so the wake must produce a new
    one - evaluated against the state and the moment it woke into.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)

    gate = ScriptedGate([(Verdict.ALLOW, None)])
    contexts = Contexts(claim_id)
    clock = ManualClock(T0)

    outcome = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
        ),
        gate=gate,
        contexts=contexts,
        clock=clock,
    )

    assert outcome.result.outcome is Outcome.ENQUEUED
    assert len(gate.calls) == 1, "the wake did not certify exactly once"

    # THE MOMENT MATTERS. Certified at the wake, not at the plan.
    _, certified_at = gate.calls[0]
    assert certified_at == PAYDAY, (
        f"certified at {certified_at} but the run woke at {PAYDAY}; a rule evaluated "
        f"at plan time cannot account for three days of sleep"
    )
    assert contexts.asked_at == [PAYDAY], "claim state was not re-fetched at the wake"

    # And the outbox row carries the certificate the WAKE issued.
    row = await conn.fetchrow(
        "SELECT certificate_id, status FROM outbox WHERE claim_id = $1", claim_id
    )
    assert row is not None, "an ALLOW at the wake did not reach the outbox"
    assert row["certificate_id"] == outcome.result.certificate_id
    assert row["status"] == OutboxStatus.PENDING.value


async def test_a_function_that_skips_the_wake_gate_is_caught(conn: asyncpg.Connection) -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    A durable function that sleeps and then enqueues WITHOUT certifying is
    planted here, going through the real outbox. The check that catches it is
    the one that matters: the row it wrote carries no certificate the Gate
    issued at the wake, so `gate.calls` is empty and the certificate on the row
    is one nobody re-validated.
    """
    from arc.conductor.outbox import enqueue

    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    gate = ScriptedGate([(Verdict.ALLOW, None)])
    clock = ManualClock(T0)
    run = await start_run(
        conn,
        function_id="ungated-retry",
        at=clock.now(),
        claim_id=claim_id,
        subject_token=TOKEN,
    )
    step = Step(conn, run, clock)

    # The defect: sleep, then enqueue directly. No re-fetch, no certify.
    stale_certificate = uuid4()
    await step.sleep_until("wait-for-payday", PAYDAY)
    await step.run_step(
        "enqueue-without-gating",
        lambda: enqueue(
            conn,
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action_type=ActionType.SMS,
            channel=ACTION_CHANNEL[ActionType.SMS].value,
            payload={},
            certificate_id=stale_certificate,
            cert_valid_from=T0 - timedelta(minutes=10),
            cert_valid_until=T0 + timedelta(minutes=10),
            not_before=PAYDAY,
        ),
    )

    assert gate.calls == [], "the planted function was supposed to skip the Gate"

    row = await conn.fetchrow(
        "SELECT certificate_id, cert_valid_until FROM outbox WHERE claim_id = $1", claim_id
    )
    assert row is not None
    # The assertion that matters: the row is authorised by a certificate that
    # had already expired by the time the function woke.
    assert row["cert_valid_until"] < PAYDAY, (
        "the planted defect did not actually produce a stale authorisation, so this "
        "test would not have caught the real thing"
    )
    assert row["certificate_id"] == stale_certificate

    # And the same wake through the real path refuses to reuse it.
    contexts = Contexts(claim_id)
    honest = await gated_enqueue(
        Step(conn, run, ManualClock(PAYDAY)),
        conn,
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=uuid4(),
        action=ActionType.SMS,
        gate=gate,
        contexts=contexts,
        at=PAYDAY,
    )
    assert len(gate.calls) == 1, "the real path did not certify at the wake"
    assert honest.certificate_id != stale_certificate


async def test_terminal_claim_stops_without_certifying(conn: asyncpg.Connection) -> None:
    """FORBORNE is absorbing. A sleeping run that wakes onto it stops.

    Checked before the Gate is consulted, because there is no action to certify
    for a claim that has reached a terminal state and asking would invite an
    ALLOW for one.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id, state=ClaimState.FORBORNE)
    gate = ScriptedGate([(Verdict.ALLOW, None)])

    result = await gated_enqueue(
        Step(
            conn,
            await start_run(conn, function_id="t", at=T0, claim_id=claim_id, subject_token=TOKEN),
            ManualClock(PAYDAY),
        ),
        conn,
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=cycle_id,
        action=ActionType.SMS,
        gate=gate,
        contexts=Contexts(claim_id),
        at=PAYDAY,
    )
    assert result.outcome is Outcome.TERMINATED
    assert result.state is ClaimState.FORBORNE
    assert gate.calls == [], "a terminal claim was certified for an action"


# ---------------------------------------------------------------------------
# Gate tests 2 and 3 - cancellation mid-sleep
# ---------------------------------------------------------------------------
async def test_hardship_event_cancels_mid_sleep(conn: asyncpg.Connection) -> None:
    """A hardship signal kills a sleeping run. Nothing polls for it.

    The signal lands on day two of a three-day sleep. There is no timer
    checking for it and no wake-and-check step - the subscription is evaluated
    at the step boundary, so the run stops at the instant it would have woken
    and never reaches the Gate at all.

    HARDSHIP IS KEYED BY SUBJECT, NOT CLAIM, and that is load-bearing: a person
    in distress holding three claims must have all three stopped, and matching
    on the claim would have left the other two running.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)
    gate = ScriptedGate([(Verdict.ALLOW, None)])

    # Mid-sleep: after the run started, before payday.
    await emit(
        conn,
        EventName.SUBJECT_HARDSHIP,
        at=T0 + timedelta(days=2),
        subject_token=TOKEN,
        payload={"signal": "bereavement"},
    )

    with pytest.raises(RunCancelled) as caught:
        await salary_aligned_retry(
            conn,
            RetryRequest(
                claim_id=claim_id,
                subject_token=TOKEN,
                cycle_id=cycle_id,
                action=ActionType.SMS,
                planned_at=PAYDAY,
            ),
            gate=gate,
            contexts=Contexts(claim_id),
            clock=ManualClock(T0),
        )

    assert caught.value.event is EventName.SUBJECT_HARDSHIP
    assert gate.calls == [], "a cancelled run still asked the Gate for authorisation"
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0, (
        "a cancelled run reached the outbox"
    )

    status = await conn.fetchrow(
        "SELECT status, cancelled_by FROM durable_runs WHERE claim_id = $1", claim_id
    )
    assert status["status"] == "cancelled"
    assert status["cancelled_by"] == EventName.SUBJECT_HARDSHIP.value, (
        "the run does not record WHICH signal stopped it"
    )


async def test_recovered_event_cancels_mid_sleep(conn: asyncpg.Connection) -> None:
    """They already paid. Chasing them anyway is the worst outcome available.

    `claim.recovered` is keyed by CLAIM, unlike hardship: one claim recovering
    says nothing about the subject's others, and cancelling those would stop
    treatment on money still outstanding.
    """
    claim_id, other_claim, cycle_id = uuid4(), uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await insert_claim(conn, other_claim)
    await declare_budget(conn, cycle_id)
    gate = ScriptedGate([(Verdict.ALLOW, None)])

    await emit(
        conn,
        EventName.CLAIM_RECOVERED,
        at=T0 + timedelta(days=1),
        claim_id=claim_id,
        subject_token=TOKEN,
    )

    with pytest.raises(RunCancelled) as caught:
        await salary_aligned_retry(
            conn,
            RetryRequest(
                claim_id=claim_id,
                subject_token=TOKEN,
                cycle_id=cycle_id,
                action=ActionType.SMS,
                planned_at=PAYDAY,
            ),
            gate=gate,
            contexts=Contexts(claim_id),
            clock=ManualClock(T0),
        )
    assert caught.value.event is EventName.CLAIM_RECOVERED

    # The subject's OTHER claim is untouched by a claim-keyed cancellation.
    other = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=other_claim,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
        ),
        gate=gate,
        contexts=Contexts(other_claim),
        clock=ManualClock(T0),
    )
    assert other.result.outcome is Outcome.ENQUEUED, (
        "recovering one claim cancelled treatment on another claim of the same subject"
    )


@pytest.mark.parametrize(
    "event",
    sorted(CANCEL_ON),
    ids=lambda e: e.value,
)
async def test_every_behavioural_stop_cancels_a_sleeping_run(
    conn: asyncpg.Connection, event: EventName
) -> None:
    """All six, not just the two the gate names.

    A function that cancelled on five of six would pass a suite that tested
    two, and the gap would only appear when the sixth signal arrived.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)

    await emit(
        conn,
        event,
        at=T0 + timedelta(days=1),
        claim_id=claim_id,
        subject_token=TOKEN,
        tenant_id="default",
    )

    with pytest.raises(RunCancelled) as caught:
        await salary_aligned_retry(
            conn,
            RetryRequest(
                claim_id=claim_id,
                subject_token=TOKEN,
                cycle_id=cycle_id,
                action=ActionType.SMS,
                planned_at=PAYDAY,
            ),
            gate=ScriptedGate([(Verdict.ALLOW, None)]),
            contexts=Contexts(claim_id),
            clock=ManualClock(T0),
        )
    assert caught.value.event is event


def test_a_function_missing_a_stop_cannot_be_declared() -> None:
    """The cancellation set is checked at import, not discovered in production."""
    with pytest.raises(IncompleteCancellation) as caught:
        assert_cancels_on_every_stop("partial-function", CANCEL_ON - {EventName.SUBJECT_HARDSHIP})
    assert "subject.hardship" in str(caught.value)


# ---------------------------------------------------------------------------
# Gate test 4 - DEFER is bounded, BLOCK is not retryable
# ---------------------------------------------------------------------------
async def test_defer_bounded_to_3_hops_then_requeues(conn: asyncpg.Connection) -> None:
    """Three hops, then a fresh decision. Not a fourth hop on a stale one.

    Each deferral re-uses a decision the Allocator made for a different moment.
    One or two is a cooldown expiring. A fourth means the world has moved far
    enough that the decision itself is stale, so the claim goes back for a new
    one with a new propensity.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)

    # Deferred forever. The bound, not the Gate, is what stops this.
    later = [(Verdict.DEFER, PAYDAY + timedelta(hours=6 * (hop + 1))) for hop in range(10)]
    gate = ScriptedGate(later)

    outcome = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
        ),
        gate=gate,
        contexts=Contexts(claim_id),
        clock=ManualClock(T0),
    )

    assert outcome.hops == MAX_DEFER_HOPS, f"deferred {outcome.hops} times, not {MAX_DEFER_HOPS}"
    assert len(gate.calls) == MAX_DEFER_HOPS + 1, (
        "each hop must re-certify; the count is one initial call plus one per hop"
    )
    assert outcome.requeued, "deferrals were exhausted without returning the claim"
    assert outcome.result.outcome is Outcome.REQUEUED
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0

    # The requeue is on the record as a fresh-decision request.
    entries = await conn.fetch(
        "SELECT body FROM decision_ledger WHERE claim_id = $1 ORDER BY seq", claim_id
    )
    assert any("DEFER_BUDGET_EXHAUSTED" in str(row["body"]) for row in entries)


async def test_block_is_not_retryable_and_terminates_the_path(
    conn: asyncpg.Connection,
) -> None:
    """A freeze with no computable end is a BLOCK, and BLOCK does not sleep.

    THE CONSEQUENCE OF M3'S LATTICE, HANDLED EXPLICITLY. An issuer outage whose
    resolution time nobody knows produces a DEFER with no timestamp, which M3
    resolves to BLOCK precisely because there is nothing to sleep until.
    Treating it as retryable would mean the scheduler inventing a duration the
    Gate declined to name.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)

    # BLOCK, and no defer_until - exactly what an unresolvable freeze produces.
    gate = ScriptedGate([(Verdict.BLOCK, None)])

    outcome = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
        ),
        gate=gate,
        contexts=Contexts(claim_id),
        clock=ManualClock(T0),
    )

    assert outcome.result.outcome is Outcome.BLOCKED
    assert not outcome.result.is_retryable, "a BLOCK was treated as retryable"
    assert outcome.hops == 0, "a BLOCK entered the defer loop"
    assert len(gate.calls) == 1, "a BLOCK was re-certified as though it might change"
    assert outcome.requeued, "a blocked claim was not returned to the Allocator"
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0

    veto = await conn.fetchrow(
        """
        SELECT body FROM decision_ledger
         WHERE claim_id = $1 AND event_type = 'gate_veto' ORDER BY seq LIMIT 1
        """,
        claim_id,
    )
    assert veto is not None, "the veto was not recorded"
    assert '"retryable": false' in str(veto["body"]).replace("False", "false").lower()


async def test_a_defer_without_a_timestamp_is_treated_as_a_block(
    conn: asyncpg.Connection,
) -> None:
    """Belt and braces on the one thing that would stall a run forever.

    M3's lattice already refuses to emit a DEFER with no timestamp. If one ever
    reached here it would go to `sleepUntil` as None and the run would never
    resume, so `gatedEnqueue` degrades it to BLOCK rather than trusting the
    upstream guarantee.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    gate = ScriptedGate([(Verdict.DEFER, None)])

    result = await gated_enqueue(
        Step(
            conn,
            await start_run(conn, function_id="t", at=T0, claim_id=claim_id, subject_token=TOKEN),
            ManualClock(PAYDAY),
        ),
        conn,
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=cycle_id,
        action=ActionType.SMS,
        gate=gate,
        contexts=Contexts(claim_id),
        at=PAYDAY,
    )
    assert result.outcome is Outcome.BLOCKED
    assert not result.is_retryable


async def test_missing_context_fails_closed(conn: asyncpg.Connection) -> None:
    """GI-5. No context is not permission to proceed."""
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    gate = ScriptedGate([(Verdict.ALLOW, None)])

    result = await gated_enqueue(
        Step(
            conn,
            await start_run(conn, function_id="t", at=T0, claim_id=claim_id, subject_token=TOKEN),
            ManualClock(PAYDAY),
        ),
        conn,
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=cycle_id,
        action=ActionType.SMS,
        gate=gate,
        contexts=Contexts(claim_id, missing=True),
        at=PAYDAY,
    )
    assert result.outcome is Outcome.BLOCKED
    assert result.blocking_rule_ids == ("CONTEXT-UNAVAILABLE",)
    assert gate.calls == [], "the Gate was asked about a claim with no state"


# ---------------------------------------------------------------------------
# Gate tests 5, 6, 7 - the promise-to-pay tracker
# ---------------------------------------------------------------------------
async def test_ptp_wait_resolves_on_payment(conn: asyncpg.Connection) -> None:
    """The promise was kept. Recorded as KEPT, with the amount that arrived."""
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    promise = Promise(
        claim_id=claim_id,
        subject_token=TOKEN,
        promise_date=T0 + timedelta(days=5),
        amount_paise=paise(129_900),
    )
    await emit(
        conn,
        EventName.PAYMENT_RECEIVED,
        at=T0 + timedelta(days=4),
        subject_token=TOKEN,
        payload={"amount_paise": 129_900},
    )

    result = await promise_to_pay_tracker(conn, promise, clock=ManualClock(T0))

    assert result.outcome is PromiseOutcome.KEPT
    assert int(result.paid_paise) == 129_900
    assert not result.requeued, "a kept promise was sent back to the Allocator"

    entry = await conn.fetchrow(
        """
        SELECT body FROM decision_ledger
         WHERE claim_id = $1 AND event_type = 'promise_outcome' ORDER BY seq LIMIT 1
        """,
        claim_id,
    )
    assert entry is not None and "kept" in str(entry["body"])


async def test_ptp_timeout_records_BROKEN_and_requeues(conn: asyncpg.Connection) -> None:
    """The deadline passed with no payment. BROKEN, and back to the Allocator."""
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    promise = Promise(
        claim_id=claim_id,
        subject_token=TOKEN,
        promise_date=T0 + timedelta(days=5),
        amount_paise=paise(129_900),
    )

    result = await promise_to_pay_tracker(conn, promise, clock=ManualClock(T0))

    assert result.outcome is PromiseOutcome.BROKEN
    assert int(result.paid_paise) == 0
    assert result.requeued

    payloads = [
        str(row["body"])
        for row in await conn.fetch(
            "SELECT body FROM decision_ledger WHERE claim_id = $1 ORDER BY seq", claim_id
        )
    ]
    assert any("broken" in p for p in payloads)
    assert any("PTP_BROKEN" in p for p in payloads), "the requeue reason was not recorded"


async def test_ptp_tracker_does_not_choose_next_action(conn: asyncpg.Connection) -> None:
    """Escalation authority belongs to L4, and this is where that is enforced.

    Two checks, because one of them can be worked around. The RESULT TYPE has
    nowhere to put an action - so a tracker that chose one could not return it.
    And the run writes NO outbox row, so it cannot have taken one either.

    The off-policy consequence is why this matters more than it looks: an
    action chosen here would have no logged propensity, because no distribution
    was sampled, and every importance ratio touching it would divide by zero.
    """
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    promise = Promise(
        claim_id=claim_id,
        subject_token=TOKEN,
        promise_date=T0 + timedelta(days=5),
        amount_paise=paise(129_900),
    )

    result = await promise_to_pay_tracker(conn, promise, clock=ManualClock(T0))
    assert result.outcome is PromiseOutcome.BROKEN

    # 1. Structural: there is nowhere in the result to name an action.
    fields = set(PromiseResult.__dataclass_fields__)
    assert not fields & {"next_action", "action", "escalate_to", "tier"}, (
        f"PromiseResult can carry a next action: {sorted(fields)}. Escalation "
        f"authority belongs to the Allocator and this type must not be able to "
        f"express a decision it does not have"
    )

    # 2. Behavioural: nothing was scheduled.
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0, (
        "the promise tracker scheduled an action after a broken promise"
    )
    # It handed the claim back instead.
    assert result.requeued


def test_unresolved_promise_is_censored_not_broken() -> None:
    """A promise dated the twentieth is not broken on the eighteenth.

    Coding unresolved as broken is what biases a promise-to-pay model
    pessimistic, and M7's Model C is fitted on exactly these records.
    """
    promise = Promise(
        claim_id=uuid4(),
        subject_token=TOKEN,
        promise_date=T0 + timedelta(days=5),
        amount_paise=paise(1),
    )
    before = T0 + timedelta(days=3)
    after = promise.deadline + timedelta(hours=1)

    assert classify(promise, paid_at=None, at=before) is PromiseOutcome.UNRESOLVED
    assert classify(promise, paid_at=None, at=after) is PromiseOutcome.BROKEN
    assert classify(promise, paid_at=before, at=after) is PromiseOutcome.KEPT
    # Paid, but after the grace ran out.
    assert classify(promise, paid_at=after, at=after) is PromiseOutcome.BROKEN


# ---------------------------------------------------------------------------
# Gate test 8 - soft intent hardens at wake, not stale
# ---------------------------------------------------------------------------
async def test_soft_intent_hard_reserved_at_wake(conn: asyncpg.Connection) -> None:
    """The two-tier rule, completed. And the expiry is reset when it hardens.

    A soft intent recorded on Monday carries an expiry a couple of hours after
    Monday. Hardening it on Thursday without moving that expiry produces a hold
    that is BORN STALE: the next sweep frees it while the dispatch it belongs
    to is still pending, and the budget is handed to another decision while
    this one is in flight.
    """
    from arc.allocator.budgets import BudgetKey

    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id, amount=10)

    key = idempotency_key(claim_id, ActionType.SMS, cycle_id, uuid4())

    # Monday: a long-horizon decision takes a SOFT intent, not a hard hold.
    # A long horizon is what makes this SOFT rather than hard - the two-tier
    # rule keyed on the gap between decision and execution, not on a flag.
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=claim_id,
        subject_token=TOKEN,
        cost=cost_of(ActionType.SMS),
        idempotency_key=key,
        at=T0,
        horizon=PAYDAY - T0,
    )
    soft = await reservations.live_for(conn, claim_id)
    # One row per priced dimension the action touches - an SMS costs a contact
    # slot and some rupees - and ALL of them are soft. A mixed set would mean
    # the tier was decided per dimension rather than per decision.
    assert soft, "no intent was recorded at all"
    assert {r.status for r in soft} == {reservations.ReservationStatus.SOFT}
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 10, (
        "a soft intent consumed cap; it is pipeline demand, not a hold"
    )

    # Thursday: the run wakes and gatedEnqueue hardens inside its transaction.
    outcome = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
            reservation_key=key,
        ),
        gate=ScriptedGate([(Verdict.ALLOW, None)]),
        contexts=Contexts(claim_id),
        clock=ManualClock(T0),
    )
    assert outcome.result.outcome is Outcome.ENQUEUED
    assert outcome.result.hardened == len(soft), (
        "the wake hardened some dimensions and not others; the hold is all or nothing"
    )

    rows = await conn.fetch(
        """
        SELECT status, reserved_at, expires_at FROM budget_reservations
         WHERE idempotency_key = $1
        """,
        key,
    )
    assert {r["status"] for r in rows} == {"hard"}
    hard = rows[0]
    assert await reservations.remaining(conn, cycle_id, BudgetKey.CONTACT) == 9, (
        "hardening did not consume cap"
    )

    # THE EXPIRY MOVED. Born at the wake, not at the plan.
    assert hard["reserved_at"] == PAYDAY, "the hold records the plan time, not the wake"
    assert hard["expires_at"] > PAYDAY, (
        f"the hardened hold expires at {hard['expires_at']}, which is not after the "
        f"wake at {PAYDAY}. It was born stale and the next sweep will free it while "
        f"the dispatch it belongs to is still pending"
    )
    # And it is not already expired at the moment it was created.
    swept = await reservations.expire_stale(conn, PAYDAY)
    assert swept == 0, "the freshly hardened hold was swept away immediately"


async def test_budget_gone_at_wake_does_not_execute(conn: asyncpg.Connection) -> None:
    """The budget assumed three days ago may be gone. Failing is correct.

    The claim returns to the Allocator for a fresh decision against the world
    as it is now, rather than executing against the world as it was.
    """
    from arc.allocator.budgets import BudgetKey

    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await reservations.declare_caps(
        conn, cycle_id, {BudgetKey.CONTACT: 1, BudgetKey.RUPEE: 100_000}
    )

    key = idempotency_key(claim_id, ActionType.SMS, cycle_id, uuid4())
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=claim_id,
        subject_token=TOKEN,
        cost=cost_of(ActionType.SMS),
        idempotency_key=key,
        at=T0,
        horizon=PAYDAY - T0,
    )
    # Somebody else took the last slot while this run slept. A SHORT horizon,
    # so it hard-reserves immediately and the cap is genuinely gone.
    await reservations.reserve(
        conn,
        cycle_id=cycle_id,
        claim_id=uuid4(),
        subject_token=TOKEN,
        cost=cost_of(ActionType.SMS),
        idempotency_key="other-holder",
        at=T0 + timedelta(days=1),
        horizon=timedelta(minutes=1),
    )

    with pytest.raises(Exception) as caught:
        await gated_enqueue(
            Step(
                conn,
                await start_run(
                    conn, function_id="t", at=T0, claim_id=claim_id, subject_token=TOKEN
                ),
                ManualClock(PAYDAY),
            ),
            conn,
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            gate=ScriptedGate([(Verdict.ALLOW, None)]),
            contexts=Contexts(claim_id),
            at=PAYDAY,
            reservation_key=key,
        )
    # The step FAILS rather than proceeding, and the run is marked failed. The
    # message is the one that matters: the budget was there at plan time and is
    # not now, so the claim goes back rather than executing on a stale hold.
    assert "is not now" in str(caught.value), caught.value
    assert "stale hold" in str(caught.value)
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 0


# ---------------------------------------------------------------------------
# Replay: a step runs once, ever
# ---------------------------------------------------------------------------
async def test_replayed_steps_do_not_run_twice(conn: asyncpg.Connection) -> None:
    """A durable function is replayed from the top, not resumed mid-line.

    Without memoisation, waking from a sleep re-runs `gatedEnqueue`, issues a
    second certificate for the same wake, and puts a second outbox row under a
    different key. The primary key on (run_id, step_id) is the whole guarantee.
    """
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)
    gate = ScriptedGate([(Verdict.ALLOW, None)])
    contexts = Contexts(claim_id)
    run = await start_run(
        conn,
        function_id="salary-aligned-retry",
        at=T0,
        claim_id=claim_id,
        subject_token=TOKEN,
    )

    request = RetryRequest(
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=cycle_id,
        action=ActionType.SMS,
        planned_at=PAYDAY,
    )
    first = await salary_aligned_retry(
        conn,
        request,
        gate=gate,
        contexts=contexts,
        clock=ManualClock(T0),
        step=Step(conn, run, ManualClock(T0)),
    )
    second_step = Step(conn, run, ManualClock(T0))
    second = await salary_aligned_retry(
        conn, request, gate=gate, contexts=contexts, clock=ManualClock(T0), step=second_step
    )

    assert first.result.outcome is second.result.outcome is Outcome.ENQUEUED
    assert first.result.certificate_id == second.result.certificate_id, (
        "the replay issued a second certificate for the same wake"
    )
    assert len(gate.calls) == 1, "the replay re-certified an already-completed step"
    assert second_step.replayed, "no step was recognised as already complete"
    assert await conn.fetchval("SELECT count(*) FROM outbox WHERE claim_id = $1", claim_id) == 1


async def test_run_status_records_the_outcome(conn: asyncpg.Connection) -> None:
    """A run that finished says how. Reconstructing it from steps is guesswork."""
    claim_id, cycle_id = uuid4(), uuid4()
    await insert_claim(conn, claim_id)
    await declare_budget(conn, cycle_id)
    outcome = await salary_aligned_retry(
        conn,
        RetryRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=ActionType.SMS,
            planned_at=PAYDAY,
        ),
        gate=ScriptedGate([(Verdict.ALLOW, None)]),
        contexts=Contexts(claim_id),
        clock=ManualClock(T0),
    )
    status = await run_status(conn, outcome.run.run_id)
    assert status["status"] == "completed"
    assert status["outcome"] == Outcome.ENQUEUED.value
