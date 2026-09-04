"""M15 acceptance gate - the full adversarial suite.

    python -m arc.demo.run --adversarial

THE ELEVEN THE BUILD DOCUMENT NAMES, plus three the build itself surfaced.
Every one goes through the component it attacks rather than a mock of it: a
suite that asserts a mock said no proves the mock said no, and what a reviewer
is being asked to believe is that the Gate, the ledger and the Conductor say no.

THE ODD ONE OUT IS THE LLM-DISABLED RUN, which must SUCCEED. A refusal there
would be the failure. The system has to be complete with the model off,
degrading in message quality and never in correctness or compliance.

WHY THIS SUITE IS WORTH RUNNING LIVE. Each line names what was attempted, that
it was refused, and which rule or assertion refused it. A refusal nobody can
attribute is indistinguishable from a bug that happened to help, and a suite
whose output is a row of dots proves the same thing while demonstrating none of
it.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from arc.allocator.budgets import BudgetKey, cost_of
from arc.conductor import reservations
from arc.conductor.kill_switch import Mode as KillMode
from arc.conductor.kill_switch import (
    assert_nothing_executed,
    current_mode,
    freeze,
    held_keys,
    resume,
    set_mode,
)
from arc.conductor.outbox import OutboxStatus, claim_batch, enqueue, idempotency_key
from arc.core.ids import subject_token
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.demo.attacks import ATTACKS, run_attack
from arc.demo.harness import adversarial_lines, llm_disabled_pipeline_lines
from arc.gate.context import ACTION_CHANNEL
from arc.llm_service import GroundingFacts, LlmClient, canned, llm_enabled, validate
from tests.conductor_db import scratch_database

PEPPER = b"m15-adversarial-suite-pepper-001"
TOKEN = subject_token("+919812345699", pepper=PEPPER)
T0 = datetime(2026, 6, 1, 9, 0, tzinfo=UTC)

# The eleven the build document lists, by the phrase each attack is described
# with. Held as data so a rename that quietly drops one fails here.
BUILD_DOC_ATTACKS: tuple[str, ...] = (
    "19:01",
    "16th retry",
    "forborne",
    "cooldown",
    "name into the ledger",
    "expired certificate",
    "no certificate",
    "wrong amount",
)


@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    try:
        yield from scratch_database("adversarial")
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover
        pytest.skip(f"postgres unavailable: {exc}")


@pytest.fixture
async def pool(dsn: str) -> AsyncIterator[asyncpg.Pool]:
    created = await asyncpg.create_pool(dsn, min_size=4, max_size=56)
    assert created is not None
    try:
        yield created
    finally:
        await created.close()


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


# ---------------------------------------------------------------------------
# The suite as a whole
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("attack", ATTACKS, ids=lambda a: a.description)
def test_every_attack_is_refused(attack) -> None:
    """Each one, through the real component, with the refusal attributed."""
    outcome = run_attack(attack)
    assert outcome.refused, f"{attack.description!r} SUCCEEDED. {outcome.refused_by}"
    assert "NOTHING" not in outcome.refused_by, (
        f"{attack.description!r} was refused but nothing was named as refusing it"
    )
    assert outcome.refused_by.strip(), f"{attack.description!r} named no refuser"


def test_every_attack_the_build_document_names_is_present() -> None:
    """A rename that quietly drops an attack fails here rather than passing."""
    described = " | ".join(a.description for a in ATTACKS).lower()
    missing = [phrase for phrase in BUILD_DOC_ATTACKS if phrase not in described]
    assert not missing, f"the suite no longer attempts: {missing}"


def test_the_three_defects_the_build_surfaced_are_attacked() -> None:
    """Guards for real defects, demonstrated rather than merely present.

    Each of these was a live bug found while building - a replayed step that
    issued a second certificate, an erasure tombstone carrying the requester's
    own email, a batch screen that reported zero outages on a batch containing
    two. A guard nobody demonstrates is a guard nobody has reason to trust.
    """
    described = " | ".join(a.description for a in ATTACKS).lower()
    for phrase in ("2nd certificate", "dpo's pii", "hiding a detected outage"):
        assert phrase in described, f"the suite does not attempt: {phrase}"


def test_the_output_names_a_rule_for_every_refusal() -> None:
    """Readable output is the deliverable, not a side effect."""
    lines = adversarial_lines()
    body = "\n".join(lines)

    assert "attempted" in body and "refused by" in body
    assert f"{len(ATTACKS)} of {len(ATTACKS)} attacks refused." in body
    assert "AN ATTACK SUCCEEDED" not in body

    for rule_id in ("TIME-WINDOW", "ABS-FORBORNE", "ABS-CONSENT", "NET-CAT1", "ABS-MINOR"):
        assert rule_id in body, f"{rule_id} does not appear as a refusing rule"
    for mechanism in ("PII write-guard", "GI-1", "GI-9", "validator/groundedness"):
        assert mechanism in body, f"{mechanism} does not appear as a refusing mechanism"


# ---------------------------------------------------------------------------
# The two that need real concurrency
# ---------------------------------------------------------------------------
async def test_double_dispatch_under_fifty_concurrent_workers(pool: asyncpg.Pool) -> None:
    """Fifty workers, one queue, zero rows claimed twice.

    `SKIP LOCKED` is only meaningful between COMMITTED sessions, so this uses a
    real pool rather than one transaction. The assertion is on what the
    CONDUCTOR handed out, not on what a provider then deduplicated: a provider
    honouring idempotency keys would absorb a double dispatch and the defect
    would never surface.
    """
    rows = 400
    workers = 50
    cycle_id = uuid4()

    async with pool.acquire() as setup:
        # A committing test in a shared scratch database. Start from a known
        # empty queue rather than from whatever an earlier test left, or the
        # counts below measure the suite's history instead of this run.
        await setup.execute("TRUNCATE outbox, held_work")
        await reservations.declare_caps(
            setup, cycle_id, {BudgetKey.CONTACT: rows * 2, BudgetKey.RUPEE: rows * 10_000}
        )
        for index in range(rows):
            claim_id = uuid4()
            await insert_claim(setup, claim_id)
            await enqueue(
                setup,
                claim_id=claim_id,
                subject_token=TOKEN,
                cycle_id=cycle_id,
                action_type=ActionType.SMS,
                channel=ACTION_CHANNEL[ActionType.SMS].value,
                payload={"n": index},
                certificate_id=uuid4(),
                cert_valid_from=T0 - timedelta(minutes=10),
                cert_valid_until=T0 + timedelta(minutes=10),
                not_before=T0 - timedelta(minutes=1),
            )

    claimed: list[list[int]] = []

    async def worker(name: int) -> None:
        async with pool.acquire() as connection:
            mine: list[int] = []
            while True:
                batch = await claim_batch(
                    connection, f"w{name}", 8, at=T0, lease=timedelta(minutes=2)
                )
                if not batch:
                    break
                mine.extend(row.id for row in batch)
            claimed.append(mine)

    await asyncio.gather(*(worker(n) for n in range(workers)))

    handed_out = [row_id for worker_rows in claimed for row_id in worker_rows]
    duplicates = len(handed_out) - len(set(handed_out))
    assert duplicates == 0, (
        f"{duplicates} row(s) were handed to more than one worker. The Conductor "
        f"guarantees exactly-once STATE TRANSITION, and a provider absorbing the "
        f"second send would hide this rather than fix it"
    )
    assert len(handed_out) == rows, f"{len(handed_out)} of {rows} rows were dispatched"

    async with pool.acquire() as check:
        await check.execute("DELETE FROM outbox WHERE cycle_id = $1", cycle_id)


async def test_a_thundering_herd_on_resume_is_refused(conn: asyncpg.Connection) -> None:
    """Everything held by a freeze, released at once on resume.

    Two separate refusals, and the test needs both. Nothing held may be
    DISPATCHED - the decisions predate the freeze and their certificates have
    expired. And nothing may be admitted above the ramp's first rung, which is
    what stops the burst that would trip CB-VOLUME and freeze the system again.
    """
    cycle_id = uuid4()
    # Same reason as the concurrency test above: the freeze is global by
    # design, so the queue has to start empty for `held == 60` to mean what it
    # says. This connection is inside a transaction that rolls back.
    await conn.execute("TRUNCATE outbox, held_work")
    await reservations.declare_caps(
        conn, cycle_id, {BudgetKey.CONTACT: 500, BudgetKey.RUPEE: 5_000_000}
    )
    for _ in range(60):
        claim_id = uuid4()
        await insert_claim(conn, claim_id)
        key = idempotency_key(claim_id, ActionType.SMS, cycle_id, uuid4())
        await reservations.reserve(
            conn,
            cycle_id=cycle_id,
            claim_id=claim_id,
            subject_token=TOKEN,
            cost=cost_of(ActionType.SMS),
            idempotency_key=key,
            at=T0,
            horizon=timedelta(minutes=5),
        )
        await enqueue(
            conn,
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action_type=ActionType.SMS,
            channel=ACTION_CHANNEL[ActionType.SMS].value,
            payload={},
            certificate_id=uuid4(),
            cert_valid_from=T0 - timedelta(minutes=10),
            cert_valid_until=T0 + timedelta(minutes=10),
            not_before=T0,
        )

    report = await freeze(conn, at=T0, changed_by="operator", reason="complaint spike")
    assert report.held == 60
    keys = await held_keys(conn)

    resumed = await resume(conn, at=T0 + timedelta(hours=8), changed_by="operator")

    # 1. NOTHING WAS EXECUTED. Read from the outbox, not from the report.
    await assert_nothing_executed(conn, keys)
    assert resumed.executed == 0
    assert resumed.invalidated == 60 and resumed.requeued == 60

    pending = await conn.fetchval(
        "SELECT count(*) FROM outbox WHERE status = $1", OutboxStatus.PENDING.value
    )
    assert pending == 0, f"{pending} held rows became dispatchable again on resume"

    # 2. AND THE RAMP HELD. Sixty items were waiting; the first rung admits
    #    five percent of the trailing median, not all sixty.
    state = await current_mode(conn)
    assert state.mode is KillMode.NORMAL
    assert state.ramp_step == 0
    admitted = state.admission_cap(60)
    assert admitted == 3, (
        f"the first rung admitted {admitted} of 60 held items. Coming back at full "
        f"volume is itself a volume surge, which trips CB-VOLUME and freezes again"
    )


async def test_a_resume_that_dispatched_would_be_caught(conn: asyncpg.Connection) -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    The check above passes trivially if nothing was ever held. This plants a
    dispatched held row and confirms the same assertion fires, naming the item.
    """
    from arc.conductor.kill_switch import HeldWorkExecuted
    from arc.conductor.outbox import mark

    await conn.execute("TRUNCATE outbox, held_work")
    cycle_id = uuid4()
    await reservations.declare_caps(conn, cycle_id, {BudgetKey.CONTACT: 10})
    claim_id = uuid4()
    await insert_claim(conn, claim_id)
    await enqueue(
        conn,
        claim_id=claim_id,
        subject_token=TOKEN,
        cycle_id=cycle_id,
        action_type=ActionType.SMS,
        channel=ACTION_CHANNEL[ActionType.SMS].value,
        payload={},
        certificate_id=uuid4(),
        cert_valid_from=T0 - timedelta(minutes=10),
        cert_valid_until=T0 + timedelta(minutes=10),
        not_before=T0,
    )
    await freeze(conn, at=T0, changed_by="operator", reason="test")
    keys = await held_keys(conn)
    assert keys

    row_id = await conn.fetchval(
        "SELECT outbox_id FROM held_work WHERE idempotency_key = $1", keys[0]
    )
    await mark(conn, row_id, OutboxStatus.SENT, error="resume dispatched a held row")

    with pytest.raises(HeldWorkExecuted) as caught:
        await assert_nothing_executed(conn, keys)
    assert keys[0] in str(caught.value)
    assert "certificates have expired" in str(caught.value)

    await set_mode(conn, KillMode.NORMAL, at=T0, changed_by="t", reason="reset")


# ---------------------------------------------------------------------------
# The one that must succeed
# ---------------------------------------------------------------------------
def test_full_pipeline_runs_with_the_llm_disabled() -> None:
    """The attack that must SUCCEED. A refusal here is the failure.

    The system has to be complete with the model off. Message quality degrades
    to the canned template; correctness and compliance do not move.
    """
    previous = os.environ.get("LLM_ENABLED")
    os.environ["LLM_ENABLED"] = "false"
    try:
        assert not llm_enabled()
        lines = llm_disabled_pipeline_lines(seed=1, size=300, cycles=2)
    finally:
        if previous is None:
            os.environ.pop("LLM_ENABLED", None)
        else:
            os.environ["LLM_ENABLED"] = previous

    body = "\n".join(lines)
    assert "COMPLETED" in body, f"the pipeline did not complete:\n{body}"
    assert "grounded=True" in body, "the canned message failed its own validator"
    assert "REFUSED" not in body

    # The pipeline did real work rather than completing by doing nothing.
    numbers = [line for line in lines if "decisions" in line or "recovered" in line]
    assert len(numbers) == 2
    assert not any(line.strip().endswith(" 0") for line in numbers), (
        f"the pipeline completed without deciding anything:\n{body}"
    )


def test_the_canned_fallback_passes_the_same_validator_a_model_faces() -> None:
    """A fallback that could not survive validation would hide a second failure."""
    facts = GroundingFacts(
        amount="Rs 1,299.00",
        due_date="12 May 2026",
        plan_name="Pro Monthly",
        merchant="Acme",
    )
    fallback = canned("dunning_v1", facts)
    verdict = validate(fallback, facts)
    assert verdict.accepted, f"the canned template is not valid output: {verdict.detail}"
    assert facts.amount in fallback.text, "the fallback does not quote the source amount"
    assert "STOP" in fallback.text, "no opt-out mechanism in the message"


def test_a_disabled_client_never_reaches_a_model() -> None:
    """With the flag off there is no path to a provider, injected or not."""
    calls: list[str] = []

    def should_not_run(task, prompt):  # pragma: no cover - asserted not to run
        calls.append(prompt)
        raise AssertionError("the model was called with LLM_ENABLED=false")

    previous = os.environ.get("LLM_ENABLED")
    os.environ["LLM_ENABLED"] = "false"
    try:
        client = LlmClient(invoke=should_not_run)
        assert not client.enabled
        facts = GroundingFacts(
            amount="Rs 1,299.00",
            due_date="12 May 2026",
            plan_name="Pro Monthly",
            merchant="Acme",
        )
        message, verdict = client.compose_message(template_id="dunning_v1", facts=facts)
    finally:
        if previous is None:
            os.environ.pop("LLM_ENABLED", None)
        else:
            os.environ["LLM_ENABLED"] = previous

    assert calls == [], "the client called the model while disabled"
    assert verdict.accepted and facts.amount in message.text
    # And the call is still on the record, marked as having taken the fallback.
    assert client.log and client.log[-1].verdict == "disabled"
