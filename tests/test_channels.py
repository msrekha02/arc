"""M10 acceptance gate: effectors that decide nothing.

The five named tests are:

    test_channels_contain_no_decision_logic
    test_duplicate_idempotency_key_deduped_by_provider
    test_provider_500_triggers_retry_not_dead
    test_provider_permanent_error_goes_to_dlq
    test_all_failure_modes_reported_structurally

THE AST SCAN READS STRING LITERALS, not just identifiers. M5's adapter scan
found the reason: `payload["amount_paise"] > 100000` reaches a domain concept
through a subscript, and a walk that only collected `ast.Name` and
`ast.Attribute` would wave it through while flagging the far more obvious
`claim.amount_paise`. Both spellings are the same defect.

ANYTHING THAT COMMITS USES A SCRATCH DATABASE. M9's load test wrote twenty
thousand append-only ledger rows into the shared development database and broke
M2's gate, which asserts the ledger starts empty. The rows could not be deleted
afterwards, because the ledger refuses UPDATE and DELETE by design. So the two
tests here that drive M9's dispatcher build their own database and drop it.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest
from arc.allocator.budgets import BudgetKey
from arc.channels import (
    ACTION_REVERSIBILITY,
    CONSENT_AFFECTING_OUTCOMES,
    DEFAULT_MIX,
    EFFECTOR_CHANNELS,
    PROVIDER_STATUSES,
    STATUS_TO_OUTCOME,
    SUCCESSFUL_OUTCOMES,
    AlwaysOutcome,
    ChannelOutcome,
    ChannelResult,
    Effector,
    Reversibility,
    UnknownProviderStatus,
    build_channels,
    channel_reversibility,
    coverage,
    fake_channels,
    outcome_for,
    reversibility_of,
)
from arc.conductor import reservations
from arc.conductor.commit import CommitRequest, commit_decision
from arc.conductor.outbox import MAX_ATTEMPTS, OutboxStatus, claim_batch, fetch_row
from arc.conductor.worker import DispatchOutcome, dispatch
from arc.core.ids import subject_token
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.gate.context import ACTION_CHANNEL, Channel
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict
from tests.conductor_db import scratch_database

REPO_ROOT = Path(__file__).resolve().parents[1]
CHANNEL_DIR = REPO_ROOT / "arc" / "channels"

TOKEN = subject_token("+919876543210", pepper=b"m10-acceptance-gate-pepper-0000")
T0 = datetime(2026, 3, 17, 10, 0, tzinfo=UTC)
LEASE = timedelta(minutes=5)


# ===========================================================================
# 1 - the AST scan
# ===========================================================================
# Domain concepts an effector must never branch on. Carrying a value through to
# the provider is transport; changing behaviour because of one is policy, and
# policy in the last layer before the world executes without having passed the
# Gate and cannot be replayed from the ledger.
FORBIDDEN_BRANCH_NAMES = frozenset(
    {
        "state",
        "claim_state",
        "ClaimState",
        "claim",
        "Claim",
        "claim_id",
        "cause",
        "Cause",
        "CauseLayer",
        "CauseLabel",
        "claim_type",
        "ClaimType",
        "amount",
        "amount_paise",
        "ltv",
        "ltv_remaining_paise",
        "arm",
        "Arm",
        "confidence",
        "subject",
        "subject_token",
        "consent",
        "ConsentState",
        "tenure_days",
        "prior_bounces_90d",
    }
)

FORBIDDEN_IMPORTS = frozenset(
    {
        "Claim",
        "ClaimState",
        "ClaimType",
        "Cause",
        "CauseLabel",
        "CauseLayer",
        "Arm",
        "ConsentState",
        "GateContext",
    }
)


def _channel_modules() -> list[Path]:
    return sorted(path for path in CHANNEL_DIR.glob("*.py") if path.name != "__init__.py")


def _names_in(node: ast.AST) -> set[str]:
    """Names, attributes AND string literals anywhere in an expression.

    The string literals are the point. `payload["amount_paise"] > 100000`
    reaches the same domain concept a subscript at a time, and a scan that only
    collected identifiers would pass it while catching the obvious spelling.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.add(child.value)
    return found


def branch_violations(path: Path) -> list[str]:
    """Every branch in one module whose condition mentions a domain concept."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences: list[str] = []

    for node in ast.walk(tree):
        tests: list[ast.AST] = []
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            tests.append(node.test)
        elif isinstance(node, ast.Match):
            tests.append(node.subject)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)
        elif isinstance(node, ast.Assert):
            tests.append(node.test)

        for test in tests:
            leaked = _names_in(test) & FORBIDDEN_BRANCH_NAMES
            if leaked:
                offences.append(
                    f"{path.name}:{getattr(node, 'lineno', 0)} branches on "
                    f"{', '.join(sorted(leaked))}"
                )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    offences.append(f"{path.name}:{node.lineno} imports {alias.name}")
    return offences


def test_channels_contain_no_decision_logic() -> None:
    """A channel effects. It does not decide.

    The effector is the last code before the world, so a rule that lives here
    runs without having passed the Gate, cannot be reconstructed from the
    ledger, and cannot be tested without a provider fixture. A compliance rule
    that only fires inside an SMS client is one nobody can audit.
    """
    modules = _channel_modules()
    assert len(modules) >= 4, "the channel scan found almost nothing; the walk is broken"

    offences = [offence for path in modules for offence in branch_violations(path)]
    assert not offences, "channels must not decide:\n" + "\n".join(f"  {o}" for o in offences)

    # The mapping from a provider's vocabulary to ours is a dict lookup rather
    # than a chain of conditionals, so there is no branch for a rule to grow in.
    effectors = (CHANNEL_DIR / "effectors.py").read_text(encoding="utf-8")
    send = ast.parse(effectors)
    bodies = [
        node
        for node in ast.walk(send)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "send"
    ]
    assert bodies, "no send() found to inspect"
    for body in bodies:
        conditionals = [
            node for node in ast.walk(body) if isinstance(node, (ast.If, ast.IfExp, ast.Match))
        ]
        assert not conditionals, (
            f"send() contains {len(conditionals)} conditional(s); the status mapping "
            "is a table lookup precisely so that it has none"
        )


def test_the_channel_scan_catches_a_planted_decision(tmp_path: Path) -> None:
    """The scan is worth nothing unless a real violation trips it.

    Four spellings of the same defect, because a scan that catches only the
    obvious one is a scan that passes the version somebody actually writes.
    """
    planted = {
        "attribute": "async def send(self, claim, key):\n"
        "    if claim.state == 'in_treatment':\n"
        "        return 'escalate'\n"
        "    return 'send'\n",
        "subscript": "async def send(self, payload, key):\n"
        "    if payload['amount_paise'] > 100000:\n"
        "        return 'voice'\n"
        "    return 'sms'\n",
        "ternary": "async def send(self, payload, key):\n"
        "    return 'a' if payload['cause'] == 'issuer_outage' else 'b'\n",
        "comprehension": "def pick(rows):\n    return [r for r in rows if r['confidence'] > 0.8]\n",
    }
    for label, source in planted.items():
        path = tmp_path / f"{label}.py"
        path.write_text(source, encoding="utf-8")
        assert branch_violations(path), f"the scan missed a decision planted as {label}:\n{source}"

    # A real module, planted into the package's own shape, is caught the same way.
    sneaky = tmp_path / "sneaky_channel.py"
    sneaky.write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass\n"
        "class SneakyEffector:\n"
        "    async def send(self, payload, idempotency_key):\n"
        "        # Looks like routing. Is policy.\n"
        "        if payload['claim_state'] == 'escalated':\n"
        "            return {'outcome': 'delivered', 'via': 'voice'}\n"
        "        return {'outcome': 'delivered', 'via': 'sms'}\n",
        encoding="utf-8",
    )
    offences = branch_violations(sneaky)
    assert offences, "a channel branching on claim state was not caught"
    assert "claim_state" in offences[0]


def test_the_channel_scan_allows_transport(tmp_path: Path) -> None:
    """Reading a provider's response and carrying a payload through is not a
    decision, or the scan would forbid the job itself."""
    honest = tmp_path / "honest.py"
    honest.write_text(
        "STATUS = {'accepted': 'delivered'}\n\n\n"
        "async def send(provider, payload, idempotency_key):\n"
        "    response = await provider.deliver(payload, idempotency_key)\n"
        "    if response.status not in STATUS:\n"
        "        raise ValueError(response.status)\n"
        "    return STATUS[response.status]\n",
        encoding="utf-8",
    )
    assert branch_violations(honest) == []


def test_channels_do_not_import_the_domain() -> None:
    """No `Claim`, no `Cause`, no `GateContext`. What it cannot see it cannot
    branch on, which is a stronger guarantee than promising not to."""
    for path in _channel_modules():
        offences = [o for o in branch_violations(path) if "imports" in o]
        assert not offences, "\n".join(offences)


# ===========================================================================
# 2 - idempotency
# ===========================================================================
async def test_duplicate_idempotency_key_deduped_by_provider() -> None:
    """The same key twice reaches the customer once.

    THIS IS THE SECOND LINE OF DEFENCE, NOT THE FIRST. The Conductor's job is
    to never present a key twice; this is what happens when something upstream
    fails anyway. Both numbers are kept, because they answer different
    questions: `invocations` is what the Conductor did, `effects` is what the
    customer got.
    """
    channels, provider = fake_channels(seed="dedupe")
    payload = {"template": "utility_reminder_v3"}

    first = await channels["sms"].send(payload, "key-alpha")
    second = await channels["sms"].send(payload, "key-alpha")

    assert first.deduplicated is False
    assert second.deduplicated is True
    assert second.outcome is first.outcome
    assert second.provider_reference == first.provider_reference

    # Two calls, one effect. The Conductor's failure stays visible.
    assert len(provider.invocations) == 2
    assert provider.distinct_keys == 1
    assert provider.duplicate_invocations == 1
    assert len(provider.effects) == 1

    # A different key is a different message, and crosses channels cleanly.
    other = await channels["email"].send(payload, "key-beta")
    assert other.deduplicated is False
    assert len(provider.effects) == 2

    # Deduplication is by key alone: the same key on another channel is still
    # the same instruction, which is what makes a re-decision need a new key.
    crossed = await channels["whatsapp"].send(payload, "key-alpha")
    assert crossed.deduplicated is True


# ===========================================================================
# 3 and 4 - the retry and dead-letter paths, driven through M9
# ===========================================================================
@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    """A scratch database, so this milestone cannot dirty the shared ledger."""
    try:
        yield from scratch_database("channels")
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


class FrozenClock:
    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, delta: timedelta) -> None:
        self._at += delta


CAPS: dict[BudgetKey, int] = dict.fromkeys(
    (k for k in BudgetKey if k is not BudgetKey.EXPLORE), 100_000
)


def certificate_for(
    claim_id: UUID,
    action: ActionType = ActionType.SMS,
    *,
    valid_from: datetime = T0 - timedelta(minutes=30),
    valid_until: datetime = T0 + timedelta(minutes=30),
) -> Certificate:
    return Certificate(
        certificate_id=uuid4(),
        decision=Verdict.ALLOW,
        valid_from=valid_from,
        valid_until=valid_until,
        evaluated_rules=(),
        blocking_rule_ids=(),
        defer_until=None,
        rule_registry_version="m10-test",
        action=action,
        issued_at=valid_from,
        claim_id=claim_id,
    )


async def seed_one(
    conn: Any,
    *,
    action: ActionType = ActionType.SMS,
    certificate: Certificate | None = None,
) -> tuple[UUID, UUID]:
    claim_id, cycle_id = uuid4(), uuid4()
    await conn.execute(
        """
        INSERT INTO claims
            (claim_id, subject_token, amount_paise, ltv_remaining_paise,
             claim_type, rail, detected_at, evidence_hash, state)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
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
    await reservations.declare_caps(conn, cycle_id, CAPS)
    await commit_decision(
        conn,
        CommitRequest(
            claim_id=claim_id,
            subject_token=TOKEN,
            cycle_id=cycle_id,
            action=action,
            certificate=certificate or certificate_for(claim_id, action),
            decision_time=T0,
            planned_execution_time=T0,
            pi_intended=0.3,
            shadow_prices={},
            payload={"template": "utility_reminder_v3"},
        ),
    )
    return claim_id, cycle_id


async def test_provider_500_triggers_retry_not_dead(conn: Any) -> None:
    """A transient failure comes back later. It does not die and it does not
    silently succeed.

    The provider fails every key for its first attempt and then behaves, so the
    retry path is exercised deterministically rather than by hoping a sampled
    error rate lands on the row under test.
    """
    claim_id, cycle_id = await seed_one(conn)
    clock = FrozenClock(T0)
    channels, provider = fake_channels(
        seed="retry", transient_failure_rate=1.0, transient_attempts=1
    )

    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=LEASE)
    assert len(rows) == 1
    first = await dispatch(conn, rows[0], channels, at=clock.now())

    assert first.outcome is DispatchOutcome.RESCHEDULED, (
        "a 500 must come back for another attempt, not go to the dead-letter queue"
    )
    row = await fetch_row(conn, rows[0].id)
    assert row.status is OutboxStatus.PENDING
    assert row.not_before > T0, "a rescheduled row must back off"
    assert row.attempts == 1
    assert "500" in (row.last_error or "")

    # The budget is still held: the action is delayed, not abandoned.
    assert await reservations.live_for(conn, claim_id) != []

    # Second attempt succeeds, on the same key.
    clock.advance(timedelta(seconds=30))
    again = await claim_batch(conn, "w", 10, at=clock.now(), lease=LEASE)
    assert again[0].idempotency_key == rows[0].idempotency_key
    assert again[0].attempts == 2
    second = await dispatch(conn, again[0], channels, at=clock.now())

    assert second.outcome is DispatchOutcome.SENT
    assert (await fetch_row(conn, rows[0].id)).status is OutboxStatus.SENT
    assert len(provider.invocations) == 2
    assert provider.distinct_keys == 1
    # One effect: the failed attempt never reached the customer.
    assert len(provider.effects) == 1
    assert await reservations.live_for(conn, claim_id) == [], (
        "a sent action consumes its reservation"
    )
    assert cycle_id is not None


async def test_provider_permanent_error_goes_to_dlq(conn: Any) -> None:
    """A permanent failure is buried immediately and its budget is freed.

    Retrying an unroutable address burns attempts against a network cap and
    accomplishes nothing, which is why the two error classes are distinct types
    rather than one exception with a flag.
    """
    claim_id, _cycle = await seed_one(conn)
    clock = FrozenClock(T0)
    channels, provider = fake_channels(seed="dlq", permanent_failure_rate=1.0)

    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=LEASE)
    result = await dispatch(conn, rows[0], channels, at=clock.now())

    assert result.outcome is DispatchOutcome.DEAD
    row = await fetch_row(conn, rows[0].id)
    assert row.status is OutboxStatus.DEAD
    assert "permanently" in (row.last_error or "")
    assert row.attempts == 1, "a permanent error must not be retried first"

    assert await reservations.live_for(conn, claim_id) == [], (
        "a dead dispatch must free its budget, or the cycle silently shrinks"
    )
    assert len(provider.invocations) == 1
    assert provider.effects == {}

    # Nothing is left claimable, so no worker picks it up again.
    assert await claim_batch(conn, "w2", 10, at=clock.now(), lease=LEASE) == []


async def test_backoff_respects_the_certificate_window(conn: Any) -> None:
    """A backoff landing past the window cancels and re-decides, never sends late.

    The retry budget is bounded by the authorisation, not the other way round.
    An action rescheduled past `cert_valid_until` has lost the permission it was
    granted, and the correct answer is a fresh decision with a fresh propensity
    rather than a late send against stale authorisation.
    """
    claim_id, _cycle = await seed_one(
        conn,
        certificate=certificate_for(
            uuid4(),
            valid_from=T0 - timedelta(seconds=5),
            valid_until=T0 + timedelta(seconds=5),
        ),
    )
    clock = FrozenClock(T0)
    channels, provider = fake_channels(
        seed="window", transient_failure_rate=1.0, transient_attempts=3
    )

    rows = await claim_batch(conn, "w", 10, at=clock.now(), lease=LEASE)
    first = await dispatch(conn, rows[0], channels, at=clock.now())
    assert first.outcome is DispatchOutcome.RESCHEDULED
    assert len(provider.invocations) == 1

    # The backoff pushes the next attempt outside the window the Gate issued.
    clock.advance(timedelta(seconds=20))
    again = await claim_batch(conn, "w", 10, at=clock.now(), lease=LEASE)
    second = await dispatch(conn, again[0], channels, at=clock.now())

    assert second.outcome is DispatchOutcome.CERT_EXPIRED
    assert second.requeued is True
    assert len(provider.invocations) == 1, (
        "the provider was reached after the certificate expired; a backoff must "
        "not turn into a late send"
    )
    assert (await fetch_row(conn, rows[0].id)).status is OutboxStatus.CANCELLED
    assert await reservations.live_for(conn, claim_id) == []


# ===========================================================================
# 5 - structured outcomes
# ===========================================================================
async def test_all_failure_modes_reported_structurally() -> None:
    """Seven outcomes, each reachable, none collapsed into a boolean.

    M7 trains on the difference between a bounce and a wrong number; M11's
    guardrails have no source for the opt-out rate without `opted_out`. A
    channel that returned success-or-failure would destroy both, and neither
    loss would be visible until the milestone that needed the signal.
    """
    assert len(ChannelOutcome) == 7
    assert {str(o) for o in ChannelOutcome} == {
        "delivered",
        "read",
        "replied",
        "bounced",
        "wrong_number",
        "opted_out",
        "failed",
    }

    # Every outcome is reachable through a real channel, not just declared.
    for outcome in ChannelOutcome:
        channels = build_channels(AlwaysOutcome(outcome))
        result = await channels["sms"].send({"template": "x"}, f"key-{outcome}")
        assert isinstance(result, ChannelResult)
        assert result.outcome is outcome
        assert result.channel is Channel.SMS
        assert result.metadata["provider_status"] in PROVIDER_STATUSES

    # Every provider status maps somewhere, and the map is total in both
    # directions - no status without an outcome, no outcome unreachable.
    assert set(STATUS_TO_OUTCOME) == set(PROVIDER_STATUSES)
    assert set(STATUS_TO_OUTCOME.values()) == set(ChannelOutcome)

    # The distinctions that matter are kept apart rather than merged.
    assert STATUS_TO_OUTCOME["undeliverable"] is ChannelOutcome.BOUNCED
    assert STATUS_TO_OUTCOME["reached_third_party"] is ChannelOutcome.WRONG_NUMBER
    assert ChannelOutcome.WRONG_NUMBER in CONSENT_AFFECTING_OUTCOMES
    assert ChannelOutcome.OPTED_OUT in CONSENT_AFFECTING_OUTCOMES
    assert ChannelOutcome.FAILED not in CONSENT_AFFECTING_OUTCOMES
    assert {
        ChannelOutcome.DELIVERED,
        ChannelOutcome.READ,
        ChannelOutcome.REPLIED,
    } == SUCCESSFUL_OUTCOMES

    # And the default traffic mix actually produces the tail, so the guardrail
    # metrics at M11 have a source rather than a permanent zero.
    channels, provider = fake_channels(seed="mix")
    for index in range(3_000):
        await channels["whatsapp"].send({"t": "x"}, f"mix-key-{index:05d}")
    seen = {outcome_for(status) for status in provider.status_counts()}
    assert seen == set(ChannelOutcome), (
        f"the default mix never produced {set(ChannelOutcome) - seen}"
    )


async def test_unknown_provider_status_fails_closed() -> None:
    """A status nobody mapped raises. It does not quietly become `failed`.

    An unrecognised state is a gap between the vendor's vocabulary and ours,
    not a known failure. Folding it into the residual bucket is how a delivery
    state the provider added last week disappears from every guardrail for a
    year (GI-5).
    """
    with pytest.raises(UnknownProviderStatus, match="no outcome in the closed set"):
        outcome_for("quantum_superposition")

    class RogueProvider:
        async def deliver(self, channel, payload, idempotency_key):
            from arc.channels.provider import ProviderResponse

            return ProviderResponse(status="invented_last_tuesday", reference="r")

    channels = build_channels(RogueProvider())
    with pytest.raises(UnknownProviderStatus):
        await channels["sms"].send({"t": "x"}, "rogue-key")


# ===========================================================================
# Reversibility
# ===========================================================================
def test_reversibility_declared_per_action() -> None:
    """The Conductor is told what it can take back, for every action.

    Declared per action rather than per channel because that is where the real
    difference is: a retry and a card-updater refresh both travel the silent
    rail, and only one of them is a debit queued for presentation.
    """
    assert set(ACTION_REVERSIBILITY) == set(ActionType), (
        f"actions without a declared reversibility: "
        f"{sorted(set(ActionType) - set(ACTION_REVERSIBILITY))}"
    )

    # Sending a message is irreversible; a scheduled retry is not.
    assert reversibility_of(ActionType.SMS) is Reversibility.IRREVERSIBLE
    assert reversibility_of(ActionType.VOICE_CALL) is Reversibility.IRREVERSIBLE
    assert reversibility_of(ActionType.STATUTORY_NOTICE) is Reversibility.IRREVERSIBLE
    assert reversibility_of(ActionType.RETRY) is Reversibility.CANCELLABLE
    assert reversibility_of(ActionType.RAIL_FALLBACK) is Reversibility.CANCELLABLE
    assert reversibility_of(ActionType.PAYMENT_LINK) is Reversibility.REVOCABLE

    # Every effector carries the value derived from its actions. One source of
    # truth, not two that can disagree.
    channels, _ = fake_channels(seed="rev")
    for name, effector in channels.items():
        channel = Channel(name)
        assert effector.reversibility is channel_reversibility(channel)
        actions = [a for a, c in ACTION_CHANNEL.items() if c is channel]
        assert actions
        assert {reversibility_of(a) for a in actions} == {effector.reversibility}

    # A channel whose actions disagreed would raise rather than one answer
    # quietly winning, which is the only way two sources of truth ever behave.
    source = inspect.getsource(channel_reversibility)
    assert "raise ValueError" in source


def test_every_carrying_action_has_an_effector() -> None:
    """No authorised action can be undeliverable for want of a channel."""
    channels, _ = fake_channels(seed="cover")
    assert coverage(channels) == []

    # `do_nothing` has no effector on purpose: there is nothing to dispatch.
    assert Channel.NONE not in EFFECTOR_CHANNELS
    assert ACTION_CHANNEL[ActionType.DO_NOTHING] is Channel.NONE
    assert Channel.NONE.value not in channels

    # And a missing channel is a dead row rather than a silent no-op, which is
    # the correct answer to "authorised, with no way to perform it".
    assert set(channels) == {c.value for c in EFFECTOR_CHANNELS}


def test_provider_failure_rates_are_reproducible_under_concurrency() -> None:
    """Which key fails is a property of the key, not of the schedule.

    Sampling from a generator on each call would make the failing set depend on
    the order twenty workers happened to arrive, so the same seed would produce
    a different run every time and M9's concurrency gate would flake for
    reasons that had nothing to do with `SKIP LOCKED`.
    """
    import asyncio

    keys = [f"repro-{i:04d}" for i in range(400)]

    async def run(order: list[str]) -> set[str]:
        channels, provider = fake_channels(seed="repro", transient_failure_rate=0.3)
        failed: set[str] = set()
        for key in order:
            try:
                await channels["sms"].send({"t": "x"}, key)
            except Exception:  # noqa: BLE001 - the identity of the failure is the point
                failed.add(key)
        return failed

    forward = asyncio.run(run(keys))
    backward = asyncio.run(run(list(reversed(keys))))

    assert forward == backward, (
        "the failing set changed when the arrival order did; the provider is "
        "sampling rather than deriving from the key"
    )
    assert 0.2 < len(forward) / len(keys) < 0.4, (
        f"{len(forward)} of {len(keys)} keys failed; the configured rate is not honoured"
    )


def test_m10_report_card(capsys: pytest.CaptureFixture[str]) -> None:
    """Print what the effector layer looks like. Run with `-s` to see it."""
    import asyncio

    channels, provider = fake_channels(seed="report")

    async def traffic() -> None:
        for index in range(2_000):
            await channels["whatsapp"].send({"t": "x"}, f"report-{index:05d}")

    asyncio.run(traffic())
    counts = provider.status_counts()
    total = sum(counts.values())

    lines = [
        "",
        "=" * 68,
        "M10 CHANNELS - dumb effectors, structured outcomes",
        "=" * 68,
        f"  effectors                      {len(channels)}",
        f"  actions covered                {len(ACTION_REVERSIBILITY)} of {len(ActionType)}",
        f"  coverage gaps                  {coverage(channels) or 'none'}",
        "",
        "  REVERSIBILITY",
        *[
            f"    {name:<14} {effector.reversibility}"
            for name, effector in sorted(channels.items())
        ],
        "",
        f"  OUTCOME MIX over {total} sends",
        *[
            f"    {str(outcome_for(status)):<14} {count:>5}  ({100 * count / total:.1f}%)"
            for status, count in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
        "",
        "  DECISION LOGIC",
        "    conditionals in send()         0",
        "    domain imports                 0",
        "=" * 68,
    ]
    with capsys.disabled():
        print("\n".join(lines))

    assert set(DEFAULT_MIX) and isinstance(channels["sms"], Effector)
    assert MAX_ATTEMPTS > 1
