"""M3 acceptance gate: the twelve adversarial tests, plus purity enforcement.

These tests search for an input that breaks a rule rather than confirming that
one obvious input is refused. The Gate is the only component whose failure is
silent and total, so a suite that only walks the happy path is worth nothing.

    test_gate_is_pure
    test_cannot_produce_call_at_1901
    test_cannot_produce_16th_retry
    test_cannot_contact_forborne_subject
    test_cannot_contact_inside_cooldown
    test_cannot_contact_without_consent
    test_lattice_most_restrictive_wins
    test_all_rules_evaluated_not_short_circuited
    test_unknown_input_fails_closed
    test_project_and_certify_agree_on_static_rules
    test_draft_rule_never_rendered_as_statutory
    test_certificate_expires
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from arc.core.ids import subject_token
from arc.core.money import paise
from arc.core.time_authority import TimezoneBasis, TzBasisKind, to_local
from arc.core.types import ActionType, ClaimState, Rail
from arc.gate.checks import CHECKS
from arc.gate.context import (
    CONTACT_CHANNELS,
    Channel,
    ConsentState,
    ContactEvent,
    ContactOutcome,
    DeclineCategory,
    GateContext,
    RetryEvent,
    RetryInitiator,
    SubjectFlags,
    TargetRelationship,
)
from arc.gate.evaluator import Gate, render_rule_mix, statutory_rules
from arc.gate.lattice import Verdict, most_restrictive, rank, resolve
from arc.gate.registry import (
    ALL_CLASSES,
    NON_BINDING_STATUSES,
    PROJECT_CLASSES,
    RuleBasis,
    RuleClass,
    RuleRegistry,
    RuleStatus,
    load_registry,
    parse_rule,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PEPPER = b"m3-acceptance-gate-pepper-000000"
TOKEN = subject_token("+919876543210", pepper=PEPPER)
KOLKATA = TimezoneBasis(kind=TzBasisKind.DECLARED, zone="Asia/Kolkata")

# Tuesday. 06:30 UTC is 12:00 IST, the middle of the contact window.
AT = datetime(2026, 3, 17, 6, 30, tzinfo=UTC)

CONTACT_ACTIONS = tuple(
    action
    for action in ActionType
    if action is not ActionType.DO_NOTHING
    and Channel(
        {
            ActionType.RETRY: "silent",
            ActionType.CARD_UPDATER: "silent",
            ActionType.MANDATE_RE_REGISTER: "silent",
            ActionType.RAIL_FALLBACK: "silent",
            ActionType.WHATSAPP_UTILITY: "whatsapp",
            ActionType.SMS: "sms",
            ActionType.EMAIL: "email",
            ActionType.PAYMENT_LINK: "payment_link",
            ActionType.VOICE_CALL: "voice",
            ActionType.INSTALMENT_OFFER: "instalment",
            ActionType.HUMAN_HANDOFF: "human",
            ActionType.STATUTORY_NOTICE: "postal",
        }[action]
    )
    in CONTACT_CHANNELS
)


@pytest.fixture(scope="module")
def gate() -> Gate:
    return Gate(load_registry())


def permissive(**overrides: Any) -> GateContext:
    """A context every rule allows, so a test can vary exactly one thing."""
    base: dict[str, Any] = {
        "claim_id": uuid4(),
        "subject_token": TOKEN,
        "rail": Rail.CARD,
        "claim_state": ClaimState.PLANNED,
        "amount_paise": paise(129900),
        "target": TargetRelationship.OBLIGOR,
        "tz_basis": KOLKATA,
        "mandate_cap_paise": paise(500000),
        "predebit_notice_at": AT - timedelta(hours=25),
        "decline_category": DeclineCategory.SOFT,
        "advice_code": None,
        "consent": {channel: ConsentState.GRANTED for channel in Channel},
        "opted_out_channels": frozenset(),
        "quiet_hours": None,
        "bank_holidays": frozenset(),
        "contacts": (),
        "retries": (),
        "certificate_valid_until": None,
        "flags": SubjectFlags(identity_verified=True),
    }
    base.update(overrides)
    return GateContext(**base)


def test_permissive_context_is_actually_permissive(gate: Gate) -> None:
    """Guards every other test in this file: if the baseline is refused, the
    adversarial tests pass for the wrong reason."""
    for action in ActionType:
        certificate = gate.certify(permissive(), action, AT)
        assert certificate.decision is Verdict.ALLOW, (
            f"{action} refused in the baseline by {certificate.blocking_rule_ids}"
        )


# ---------------------------------------------------------------------------
# 1
# ---------------------------------------------------------------------------
def test_gate_is_pure(gate: Gate) -> None:
    """Identical inputs, one thousand times, one distinct result."""
    ctx = permissive(
        contacts=(ContactEvent(AT - timedelta(hours=2), Channel.SMS, ContactOutcome.DELIVERED),),
        retries=(RetryEvent(AT - timedelta(days=1), Rail.CARD),),
    )

    results = {gate.certify(ctx, ActionType.VOICE_CALL, AT) for _ in range(1000)}
    assert len(results) == 1, "certify is not a function of its inputs"

    # Including the certificate id, which is derived rather than generated.
    only = next(iter(results))
    assert only.certificate_id == gate.certify(ctx, ActionType.VOICE_CALL, AT).certificate_id

    # And the same holds for the pruning call site.
    masks = {frozenset(gate.project(ctx, list(ActionType), AT)) for _ in range(200)}
    assert len(masks) == 1


def test_gate_performs_no_io_and_reads_no_clock() -> None:
    """AST enforcement of what the docstrings promise.

    Only `registry.py` may touch a filesystem, and it does so at load time. The
    evaluation path imports nothing that could reach a database, a model, an
    LLM, or a random number generator.
    """
    forbidden_everywhere = {
        "arc.llm_service",
        "arc.models",
        "arc.ledger",
        "asyncpg",
        "psycopg",
        "psycopg2",
        "sqlalchemy",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "random",
        "secrets",
    }
    forbidden_outside_registry = {"pathlib", "yaml", "os", "io", "shutil", "tempfile"}
    clock_calls = {"now", "utcnow", "today", "time_ns", "monotonic"}

    modules = sorted((REPO_ROOT / "arc" / "gate").rglob("*.py"))
    assert len(modules) >= 5

    problems: list[str] = []
    for path in modules:
        rel = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        banned = forbidden_everywhere | (
            set() if path.name == "registry.py" else forbidden_outside_registry
        )

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in banned or alias.name in banned:
                        problems.append(f"{rel}:{node.lineno} imports {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
                if node.module in banned or root in banned:
                    problems.append(f"{rel}:{node.lineno} imports from {node.module}")
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in clock_calls
            ):
                problems.append(f"{rel}:{node.lineno} calls .{node.func.attr}()")

    assert not problems, "the Gate is not pure:\n" + "\n".join(problems)


# ---------------------------------------------------------------------------
# 2
# ---------------------------------------------------------------------------
def test_cannot_produce_call_at_1901(gate: Gate) -> None:
    """Search a whole day, minute by minute, for a voice call outside hours."""
    ctx = permissive()
    start = datetime(2026, 3, 17, 0, 0, tzinfo=UTC)

    allowed_at: list[datetime] = []
    for minute in range(24 * 60):
        moment = start + timedelta(minutes=minute)
        certificate = gate.certify(ctx, ActionType.VOICE_CALL, moment)
        if certificate.decision is not Verdict.ALLOW:
            continue

        allowed_at.append(moment)
        local = to_local(moment, KOLKATA).time()
        assert time(8, 0) <= local < time(19, 0), f"ALLOW at local {local}"

        # The certificate window must not authorise a moment the Gate refuses.
        for edge in (certificate.valid_from, certificate.valid_until):
            edge_local = to_local(edge, KOLKATA).time()
            assert time(8, 0) <= edge_local < time(19, 0), (
                f"certificate issued at local {local} reaches local {edge_local}"
            )

    assert allowed_at, "nothing was allowed all day; the search proved nothing"

    # The specific attack: authorise at 18:58 local, execute at 19:01 local.
    at_1858 = datetime(2026, 3, 17, 13, 28, tzinfo=UTC)
    assert to_local(at_1858, KOLKATA).time() == time(18, 58)
    certificate = gate.certify(ctx, ActionType.VOICE_CALL, at_1858)
    assert certificate.decision is Verdict.ALLOW

    at_1901 = datetime(2026, 3, 17, 13, 31, tzinfo=UTC)
    assert to_local(at_1901, KOLKATA).time() == time(19, 1)
    assert certificate.authorises(ActionType.VOICE_CALL, at_1901) is False
    assert gate.certify(ctx, ActionType.VOICE_CALL, at_1901).decision is Verdict.DEFER


def test_out_of_hours_defer_carries_a_usable_timestamp(gate: Gate) -> None:
    """A DEFER with no timestamp is a BLOCK wearing the wrong label."""
    after_hours = datetime(2026, 3, 17, 16, 0, tzinfo=UTC)  # 21:30 IST
    certificate = gate.certify(permissive(), ActionType.SMS, after_hours)

    assert certificate.decision is Verdict.DEFER
    assert certificate.defer_until is not None
    assert certificate.defer_until > after_hours
    assert to_local(certificate.defer_until, KOLKATA).time() == time(8, 0)

    # And the deferred moment is genuinely eligible, not merely later.
    assert gate.certify(permissive(), ActionType.SMS, certificate.defer_until).decision is (
        Verdict.ALLOW
    )


def test_every_defer_anywhere_carries_a_timestamp(gate: Gate) -> None:
    """Swept across contexts, no DEFER is ever emitted without a wake time."""
    contexts = [
        permissive(),
        permissive(quiet_hours=(time(21, 0), time(7, 0))),
        permissive(
            flags=SubjectFlags(
                identity_verified=True, ptp_active=True, ptp_freeze_until=AT + timedelta(days=3)
            )
        ),
        permissive(
            flags=SubjectFlags(
                identity_verified=True,
                payment_pending=True,
                payment_pending_until=AT + timedelta(hours=48),
            )
        ),
        permissive(
            flags=SubjectFlags(
                identity_verified=True,
                issuer_degraded=True,
                issuer_degraded_until=AT + timedelta(hours=1),
            )
        ),
        permissive(
            contacts=tuple(
                ContactEvent(AT - timedelta(hours=h), Channel.SMS, ContactOutcome.DELIVERED)
                for h in (1, 5, 30)
            )
        ),
        permissive(retries=(RetryEvent(AT - timedelta(hours=3), Rail.CARD),)),
        permissive(bank_holidays=frozenset({AT.date()})),
    ]
    for offset in (0, 8, 14, 20):
        moment = datetime(2026, 3, 17, offset, 0, tzinfo=UTC)
        for ctx in contexts:
            for action in ActionType:
                evaluation = gate.evaluate(ctx, action, moment, classes=ALL_CLASSES)
                for verdict in evaluation.verdicts:
                    if verdict.verdict is Verdict.DEFER:
                        assert verdict.defer_until is not None, verdict.rule_id
                        assert verdict.defer_until > moment, verdict.rule_id
                if evaluation.decision is Verdict.DEFER:
                    assert evaluation.defer_until is not None


# ---------------------------------------------------------------------------
# 3
# ---------------------------------------------------------------------------
def test_cannot_produce_16th_retry(gate: Gate) -> None:
    """Sweep the attempt count and find the first refusal, rather than assume it."""
    cap = gate.registry["NET-RETRY-30D"].params["max_attempts"]
    assert cap == 15

    first_refused: int | None = None
    for prior in range(0, 21):
        # Spaced beyond the daily limit so only the 30-day cap can bind.
        retries = tuple(
            RetryEvent(AT - timedelta(hours=25 * (n + 1)), Rail.CARD) for n in range(prior)
        )
        decision = gate.certify(permissive(retries=retries), ActionType.RETRY, AT).decision
        if decision is not Verdict.ALLOW and first_refused is None:
            first_refused = prior
        assert (decision is Verdict.ALLOW) == (prior < cap), f"{prior} prior attempts"

    assert first_refused == cap, "the 16th presentment is the first one refused"

    blocked = gate.certify(
        permissive(
            retries=tuple(
                RetryEvent(AT - timedelta(hours=25 * (n + 1)), Rail.CARD) for n in range(15)
            )
        ),
        ActionType.RETRY,
        AT,
    )
    assert "NET-RETRY-30D" in blocked.blocking_rule_ids


def test_gateway_initiated_retries_count_against_the_cap(gate: Gate) -> None:
    """Otherwise the counter is wrong in the direction that breaches the cap."""
    retries = tuple(
        RetryEvent(AT - timedelta(hours=25 * (n + 1)), Rail.CARD, RetryInitiator.GATEWAY)
        for n in range(15)
    )
    certificate = gate.certify(permissive(retries=retries), ActionType.RETRY, AT)
    assert certificate.decision is not Verdict.ALLOW
    assert "NET-RETRY-30D" in certificate.blocking_rule_ids


def test_hard_decline_is_permanent_not_merely_capped(gate: Gate) -> None:
    for category in (
        DeclineCategory.LOST_OR_STOLEN,
        DeclineCategory.ACCOUNT_CLOSED,
        DeclineCategory.STOP_PAYMENT,
    ):
        certificate = gate.certify(permissive(decline_category=category), ActionType.RETRY, AT)
        assert certificate.decision is Verdict.BLOCK_PERMANENT
        assert "NET-CAT1" in certificate.blocking_rule_ids

    certificate = gate.certify(permissive(advice_code="MAC03"), ActionType.RETRY, AT)
    assert certificate.decision is Verdict.BLOCK_PERMANENT
    assert "NET-MAC03" in certificate.blocking_rule_ids


# ---------------------------------------------------------------------------
# 4
# ---------------------------------------------------------------------------
def test_cannot_contact_forborne_subject(gate: Gate) -> None:
    """Search every action, every hour of a week, for one that gets through."""
    ctx = permissive(flags=SubjectFlags(identity_verified=True, forborne=True))

    escapes: list[str] = []
    for hour in range(24 * 7):
        moment = AT + timedelta(hours=hour)
        eligible = gate.project(ctx, list(ActionType), moment)
        for action in ActionType:
            if action is ActionType.DO_NOTHING:
                continue
            certificate = gate.certify(ctx, action, moment)
            if certificate.decision is Verdict.ALLOW:
                escapes.append(f"{action} at {moment.isoformat()}")
            if action in eligible:
                escapes.append(f"{action} survived project() at {moment.isoformat()}")

    assert not escapes, "FORBORNE subject reachable:\n" + "\n".join(escapes[:10])

    # Absorbing means permanent, and it covers silent rail actions too, because
    # forbearance means collection stops rather than continuing quietly.
    for action in (ActionType.VOICE_CALL, ActionType.RETRY, ActionType.CARD_UPDATER):
        certificate = gate.certify(ctx, action, AT)
        assert certificate.decision is Verdict.BLOCK_PERMANENT
        assert "ABS-FORBORNE" in certificate.blocking_rule_ids

    # do_nothing stays available; the Allocator always needs it.
    assert gate.certify(ctx, ActionType.DO_NOTHING, AT).decision is Verdict.ALLOW


def test_hardship_and_erasure_are_also_permanent(gate: Gate) -> None:
    for flag, rule_id in (("hardship", "FRZ-HARDSHIP"), ("erasure_requested", "FRZ-ERASURE")):
        flags = SubjectFlags(identity_verified=True, **{flag: True})
        certificate = gate.certify(permissive(flags=flags), ActionType.VOICE_CALL, AT)
        assert certificate.decision is Verdict.BLOCK_PERMANENT
        assert rule_id in certificate.blocking_rule_ids


# ---------------------------------------------------------------------------
# 5
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("action", "channel", "cooldown_hours"),
    [
        (ActionType.VOICE_CALL, Channel.VOICE, 48),
        (ActionType.WHATSAPP_UTILITY, Channel.WHATSAPP, 24),
        (ActionType.SMS, Channel.SMS, 24),
        (ActionType.EMAIL, Channel.EMAIL, 12),
    ],
)
def test_cannot_contact_inside_cooldown(
    gate: Gate, action: ActionType, channel: Channel, cooldown_hours: int
) -> None:
    """Sweep the gap in fifteen-minute steps looking for an early ALLOW."""
    ctx = permissive(contacts=(ContactEvent(AT, channel, ContactOutcome.DELIVERED),))
    cooldown = timedelta(hours=cooldown_hours)

    # From one minute after the previous contact rather than from zero: a
    # contact at the same instant is simultaneous, not inside the cooldown.
    # The sweep runs past 24h so it clears FREQ-24H, which binds harder than
    # the shorter channel cooldowns and would otherwise hide every ALLOW.
    gaps = [timedelta(minutes=1)] + [
        timedelta(minutes=15 * quarter) for quarter in range(1, (cooldown_hours + 28) * 4)
    ]

    violations: list[str] = []
    allowed_after: list[timedelta] = []
    for gap in gaps:
        if gate.certify(ctx, action, AT + gap).decision is Verdict.ALLOW:
            allowed_after.append(gap)
            if gap < cooldown:
                violations.append(f"{action} allowed {gap} after the last {channel}")

    assert not violations, chr(10).join(violations[:5])
    assert allowed_after, "nothing was ever allowed; the sweep proved nothing"
    assert min(allowed_after) >= cooldown


def test_connected_voice_has_a_longer_cooldown_than_unanswered(gate: Gate) -> None:
    """If you spoke to them you have your answer; calling back is pressure."""
    at_72h = AT + timedelta(hours=72)

    unanswered = permissive(contacts=(ContactEvent(AT, Channel.VOICE, ContactOutcome.NO_ANSWER),))
    connected = permissive(contacts=(ContactEvent(AT, Channel.VOICE, ContactOutcome.CONNECTED),))

    assert gate.certify(unanswered, ActionType.VOICE_CALL, at_72h).decision is Verdict.ALLOW

    refused = gate.certify(connected, ActionType.VOICE_CALL, at_72h)
    assert refused.decision is Verdict.DEFER
    assert "CD-VOICE-CONNECTED" in refused.blocking_rule_ids


def test_cross_channel_cooldown_stops_a_burst(gate: Gate) -> None:
    """One WhatsApp, one SMS and one email each pass their own cooldown."""
    ctx = permissive(contacts=(ContactEvent(AT, Channel.WHATSAPP, ContactOutcome.DELIVERED),))
    ten_minutes_later = AT + timedelta(minutes=10)

    certificate = gate.certify(ctx, ActionType.SMS, ten_minutes_later)
    assert certificate.decision is Verdict.DEFER
    assert "CD-CROSS" in certificate.blocking_rule_ids


def test_frequency_caps_are_counted_at_the_subject(gate: Gate) -> None:
    """Three claims must not buy three times the contact."""
    contacts = tuple(
        ContactEvent(AT - timedelta(days=d), Channel.SMS, ContactOutcome.DELIVERED)
        for d in (1, 2, 3)
    )
    certificate = gate.certify(permissive(contacts=contacts), ActionType.EMAIL, AT)
    assert certificate.decision is Verdict.DEFER
    assert "FREQ-7D" in certificate.blocking_rule_ids


# ---------------------------------------------------------------------------
# 6
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "state", [ConsentState.UNKNOWN, ConsentState.WITHDRAWN, ConsentState.NEVER_GIVEN]
)
def test_cannot_contact_without_consent(gate: Gate, state: ConsentState) -> None:
    for action in CONTACT_ACTIONS:
        channel = permissive().channel_for(action)
        consent = {c: ConsentState.GRANTED for c in Channel}
        consent[channel] = state

        certificate = gate.certify(permissive(consent=consent), action, AT)
        if channel in {Channel.HUMAN, Channel.POSTAL}:
            continue  # ABS-CONSENT does not scope to these
        assert certificate.decision is not Verdict.ALLOW, f"{action} with consent {state}"
        assert "ABS-CONSENT" in certificate.blocking_rule_ids


def test_a_missing_consent_record_is_not_consent(gate: Gate) -> None:
    """An empty map must not read as permission."""
    certificate = gate.certify(permissive(consent={}), ActionType.SMS, AT)
    assert certificate.decision is Verdict.BLOCK
    assert "ABS-CONSENT" in certificate.blocking_rule_ids


def test_opt_out_is_permanent_for_that_channel_only(gate: Gate) -> None:
    ctx = permissive(opted_out_channels=frozenset({Channel.VOICE}))

    refused = gate.certify(ctx, ActionType.VOICE_CALL, AT)
    assert refused.decision is Verdict.BLOCK_PERMANENT
    assert "ABS-OPTOUT" in refused.blocking_rule_ids

    assert gate.certify(ctx, ActionType.EMAIL, AT).decision is Verdict.ALLOW


# ---------------------------------------------------------------------------
# 7
# ---------------------------------------------------------------------------
def test_lattice_most_restrictive_wins(gate: Gate) -> None:
    assert rank(Verdict.BLOCK_PERMANENT) > rank(Verdict.BLOCK) > rank(Verdict.DEFER)
    assert rank(Verdict.DEFER) > rank(Verdict.ALLOW)

    for verdict in Verdict:
        assert most_restrictive([Verdict.ALLOW, verdict]) is verdict
        assert most_restrictive([verdict, Verdict.BLOCK_PERMANENT]) is Verdict.BLOCK_PERMANENT

    # A DEFER waits for the LATEST deferring rule, not the earliest.
    early, late = AT + timedelta(hours=1), AT + timedelta(hours=9)
    resolution = resolve([(Verdict.DEFER, early), (Verdict.DEFER, late), (Verdict.ALLOW, None)])
    assert resolution.decision is Verdict.DEFER
    assert resolution.defer_until == late

    # End to end: a deferring rule and a blocking rule together give BLOCK.
    both = permissive(
        contacts=(ContactEvent(AT, Channel.VOICE, ContactOutcome.NO_ANSWER),),
        flags=SubjectFlags(identity_verified=True, complaint_open=True),
    )
    certificate = gate.certify(both, ActionType.VOICE_CALL, AT + timedelta(hours=1))
    assert certificate.decision is Verdict.BLOCK
    assert certificate.defer_until is None

    # Add a permanent one and it dominates both.
    worse = dataclasses.replace(
        both, flags=SubjectFlags(identity_verified=True, complaint_open=True, forborne=True)
    )
    assert gate.certify(worse, ActionType.VOICE_CALL, AT).decision is Verdict.BLOCK_PERMANENT


# ---------------------------------------------------------------------------
# 8
# ---------------------------------------------------------------------------
def test_all_rules_evaluated_not_short_circuited(gate: Gate) -> None:
    """A refusal early in the registry must not hide a violation later in it."""
    total = len(gate.registry)
    assert total == 33

    # ABS-FORBORNE sorts near the front; FREQ-30D is in the other family.
    ctx = permissive(
        flags=SubjectFlags(identity_verified=True, forborne=True),
        contacts=tuple(
            ContactEvent(AT - timedelta(days=d), Channel.SMS, ContactOutcome.DELIVERED)
            for d in range(1, 10)
        ),
    )
    certificate = gate.certify(ctx, ActionType.VOICE_CALL, AT)

    assert certificate.decision is Verdict.BLOCK_PERMANENT
    assert len(certificate.evaluated_rules) == total
    assert len({v.rule_id for v in certificate.evaluated_rules}) == total

    reported = {v.rule_id: v for v in certificate.evaluated_rules}
    assert reported["ABS-FORBORNE"].verdict is Verdict.BLOCK_PERMANENT
    # The later rule still has a real verdict even though the outcome was
    # already decided by the earlier one.
    assert reported["FREQ-7D"].verdict is Verdict.DEFER
    assert reported["FREQ-7D"].applicable is True
    assert reported["FREQ-7D"].detail

    # Every rule reports whether it was in scope, so the audit trail can show
    # what was considered and not merely what refused.
    assert all(isinstance(v.applicable, bool) for v in certificate.evaluated_rules)
    assert any(v.applicable is False for v in certificate.evaluated_rules)

    # An ALLOW carries the full list too.
    clean = gate.certify(permissive(), ActionType.SMS, AT)
    assert clean.decision is Verdict.ALLOW
    assert len(clean.evaluated_rules) == total
    assert clean.blocking_rule_ids == ()


# ---------------------------------------------------------------------------
# 9
# ---------------------------------------------------------------------------
def test_unknown_input_fails_closed(gate: Gate) -> None:
    unresolved_tz = gate.certify(permissive(tz_basis=None), ActionType.VOICE_CALL, AT)
    assert unresolved_tz.decision is Verdict.BLOCK
    assert "TIME-WINDOW" in unresolved_tz.blocking_rule_ids

    unknown_target = gate.certify(permissive(target=TargetRelationship.UNKNOWN), ActionType.SMS, AT)
    assert unknown_target.decision is Verdict.BLOCK_PERMANENT
    assert "ABS-THIRD-PARTY" in unknown_target.blocking_rule_ids

    unknown_cap = gate.certify(permissive(mandate_cap_paise=None), ActionType.RETRY, AT)
    assert unknown_cap.decision is Verdict.BLOCK
    assert "NET-MANDATE-CAP" in unknown_cap.blocking_rule_ids

    no_notice = gate.certify(permissive(predebit_notice_at=None), ActionType.RETRY, AT)
    assert no_notice.decision is Verdict.BLOCK
    assert "NET-PREDEBIT" in no_notice.blocking_rule_ids

    # A freeze with no known end is a BLOCK, never an unbounded DEFER.
    open_ended = gate.certify(
        permissive(flags=SubjectFlags(identity_verified=True, issuer_degraded=True)),
        ActionType.SMS,
        AT,
    )
    assert open_ended.decision is Verdict.BLOCK
    assert "FRZ-ISSUER" in open_ended.blocking_rule_ids


def test_a_rule_that_throws_blocks_rather_than_passing(gate: Gate) -> None:
    """A broken rule must not become a permissive one."""
    broken = RuleRegistry(
        [
            parse_rule(
                {
                    "id": "TEST-BROKEN",
                    "class": "INVARIANT",
                    "scope": ["sms"],
                    "basis": "policy_choice",
                    "status": "in_force",
                    "check": "subject_flag",
                    "params": {"flag": "no_such_flag"},
                    "on_violation": "ALLOW",
                    "rationale": "deliberately references a field that does not exist",
                }
            )
        ]
    )
    verdict = Gate(broken).certify(permissive(), ActionType.SMS, AT)
    assert verdict.decision is Verdict.BLOCK
    assert "TEST-BROKEN" in verdict.blocking_rule_ids
    assert "failing closed" in verdict.evaluated_rules[0].detail


def test_a_registry_naming_an_unknown_check_will_not_load() -> None:
    """Nothing runs ungated because a rule silently did nothing."""
    registry = RuleRegistry(
        [
            parse_rule(
                {
                    "id": "TEST-GHOST",
                    "class": "INVARIANT",
                    "scope": ["sms"],
                    "basis": "policy_choice",
                    "status": "in_force",
                    "check": "no_such_check",
                    "params": {},
                    "on_violation": "BLOCK",
                    "rationale": "names a check that does not exist",
                }
            )
        ]
    )
    with pytest.raises(KeyError):
        Gate(registry)


# ---------------------------------------------------------------------------
# 10
# ---------------------------------------------------------------------------
def test_project_and_certify_agree_on_static_rules(gate: Gate) -> None:
    """One evaluator, two call sites. INVARIANT verdicts must be identical."""
    contexts = [
        permissive(),
        permissive(flags=SubjectFlags(identity_verified=True, forborne=True)),
        permissive(flags=SubjectFlags(identity_verified=False)),
        permissive(target=TargetRelationship.THIRD_PARTY),
        permissive(target=TargetRelationship.EMPLOYER),
        permissive(consent={}),
        permissive(opted_out_channels=frozenset({Channel.VOICE, Channel.SMS})),
        permissive(decline_category=DeclineCategory.LOST_OR_STOLEN),
        permissive(advice_code="MAC03"),
        permissive(mandate_cap_paise=paise(1)),
        permissive(flags=SubjectFlags(identity_verified=True, disputed=True)),
        permissive(flags=SubjectFlags(identity_verified=True, minor=True)),
        permissive(flags=SubjectFlags(identity_verified=True, complaint_open=True)),
    ]
    moments = [AT + timedelta(hours=h) for h in (0, 5, 11, 19, 26, 73)]

    compared = 0
    for ctx in contexts:
        for moment in moments:
            for action in ActionType:
                projected = gate.evaluate(ctx, action, moment, classes=PROJECT_CLASSES)
                certified = gate.evaluate(ctx, action, moment, classes=ALL_CLASSES)

                left = {
                    v.rule_id: (v.verdict, v.defer_until, v.applicable)
                    for v in projected.verdicts
                    if v.rule_class is RuleClass.INVARIANT
                }
                right = {
                    v.rule_id: (v.verdict, v.defer_until, v.applicable)
                    for v in certified.verdicts
                    if v.rule_class is RuleClass.INVARIANT
                }
                assert left == right, f"drift on {action} at {moment}"
                compared += len(left)

    assert compared > 5000, "the comparison covered too little to mean anything"


def test_project_never_consults_runtime_rules(gate: Gate) -> None:
    """RUNTIME rules are not decidable at plan time and must not prune."""
    runtime_ids = {r.id for r in gate.registry if r.rule_class is RuleClass.RUNTIME}
    assert runtime_ids == {"TIME-CERT-WINDOW"}

    projected = gate.evaluate(permissive(), ActionType.SMS, AT, classes=PROJECT_CLASSES)
    assert runtime_ids.isdisjoint({v.rule_id for v in projected.verdicts})

    certified = gate.certify(permissive(), ActionType.SMS, AT)
    assert runtime_ids <= {v.rule_id for v in certified.evaluated_rules}


def test_project_cannot_authorise(gate: Gate) -> None:
    """It returns a mask of action types, never a certificate."""
    eligible = gate.project(permissive(), list(ActionType), AT)
    assert isinstance(eligible, set)
    assert all(isinstance(action, ActionType) for action in eligible)
    assert not hasattr(eligible, "certificate_id")


def test_there_is_one_evaluator_not_two(gate: Gate) -> None:
    """Both call sites reach the same function, filtered differently."""
    source = (REPO_ROOT / "arc" / "gate" / "evaluator.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    gate_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Gate"
    )
    methods = {
        node.name: node
        for node in gate_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for entry_point in ("project", "certify", "project_evaluations"):
        calls = {
            node.func.attr
            for node in ast.walk(methods[entry_point])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "evaluate" in calls, f"{entry_point} does not go through evaluate()"


# ---------------------------------------------------------------------------
# 11
# ---------------------------------------------------------------------------
def test_draft_rule_never_rendered_as_statutory(gate: Gate) -> None:
    registry = gate.registry
    non_binding = [rule for rule in registry if rule.status in NON_BINDING_STATUSES]
    assert non_binding, "no draft or advisory rule exists, so this proves nothing"

    for rule in non_binding:
        label = rule.force_label()
        assert "statutory" not in label.lower(), f"{rule.id}: {label}"
        assert rule.is_binding_law() is False
        assert rule.id not in {r.id for r in statutory_rules(registry)}

    # The sharp case the registry does not currently contain: a rule whose
    # basis IS statutory but whose instrument is still draft.
    sharp = parse_rule(
        {
            "id": "TEST-DRAFT-STATUTE",
            "class": "INVARIANT",
            "scope": ["sms"],
            "basis": "statutory",
            "status": "draft",
            "check": "channel_consent",
            "params": {},
            "on_violation": "BLOCK",
            "rationale": "a statutory instrument still at consultation stage",
        }
    )
    assert "statutory" not in sharp.force_label().lower()
    assert sharp.is_binding_law() is False

    # No rendering path leaks it either.
    ctx = permissive(flags=SubjectFlags(identity_verified=True))
    certificate = gate.certify(ctx, ActionType.VOICE_CALL, AT)
    audit = certificate.to_audit_dict(registry)
    rendered = {row["rule_id"]: row for row in audit["evaluated_rules"]}

    for rule in non_binding:
        row = rendered[rule.id]
        assert "statutory" not in row["force"].lower()
        assert row["binding_law"] is False

    mix = "\n".join(render_rule_mix(registry))
    for rule in non_binding:
        line = next(ln for ln in mix.splitlines() if ln.strip().startswith(rule.id))
        assert "statutory" not in line.lower()
        assert " ours " in line


def test_the_rule_mix_is_honest(gate: Gate) -> None:
    """The compliance panel shows the real split, not a clean table of law."""
    summary = gate.registry.summary()
    assert summary["total"] == 33
    assert summary["status:contested"] >= 1
    assert summary["status:draft"] >= 1
    assert summary["status:advisory"] >= 1
    assert summary["basis:policy_choice"] >= 15
    assert summary["stricter_than_binding_minimum"] >= 20

    # Every frequency and cooldown interval is our own choice, because no
    # instrument specifies a number.
    for rule in gate.registry:
        if rule.id.startswith(("CD-", "FREQ-")):
            assert rule.basis is RuleBasis.POLICY_CHOICE, rule.id

    assert gate.registry["NET-RETRY-30D"].status is RuleStatus.CONTESTED


def test_every_rule_declares_basis_status_and_a_citation(gate: Gate) -> None:
    for rule in gate.registry:
        assert isinstance(rule.basis, RuleBasis), rule.id
        assert isinstance(rule.status, RuleStatus), rule.id
        assert rule.rationale.strip(), rule.id
        assert rule.informed_by, f"{rule.id} cites nothing"
        for citation in rule.informed_by:
            assert citation.instrument.strip()
            assert citation.force.strip()


def test_a_future_dated_obligation_is_not_a_current_one(gate: Gate) -> None:
    future = parse_rule(
        {
            "id": "TEST-FUTURE",
            "class": "INVARIANT",
            "scope": ["sms"],
            "basis": "statutory",
            "status": "in_force",
            "in_force_from": datetime(2030, 1, 1).date(),
            "check": "channel_opted_out",
            "params": {},
            "on_violation": "BLOCK_PERMANENT",
            "rationale": "commences in 2030",
        }
    )
    small = Gate(RuleRegistry([future]))
    ctx = permissive(opted_out_channels=frozenset({Channel.SMS}))

    assert small.certify(ctx, ActionType.SMS, AT).decision is Verdict.ALLOW
    later = datetime(2030, 6, 1, 6, 30, tzinfo=UTC)
    assert small.certify(ctx, ActionType.SMS, later).decision is Verdict.BLOCK_PERMANENT


# ---------------------------------------------------------------------------
# 12
# ---------------------------------------------------------------------------
def test_certificate_expires(gate: Gate) -> None:
    certificate = gate.certify(permissive(), ActionType.SMS, AT)
    assert certificate.decision is Verdict.ALLOW

    assert certificate.is_valid_at(AT) is True
    assert certificate.is_valid_at(certificate.valid_until) is True
    assert certificate.is_valid_at(certificate.valid_until + timedelta(seconds=1)) is False
    assert certificate.is_valid_at(certificate.valid_from - timedelta(seconds=1)) is False

    # It authorises one action, not a class of them.
    assert certificate.authorises(ActionType.SMS, AT) is True
    assert certificate.authorises(ActionType.VOICE_CALL, AT) is False

    # A refusal is never valid, whatever the window says.
    refused = gate.certify(
        permissive(flags=SubjectFlags(identity_verified=True, forborne=True)),
        ActionType.SMS,
        AT,
    )
    assert refused.is_valid_at(AT) is False

    # Touchpoint 3: an expired certificate presented at dispatch blocks, so the
    # dispatcher re-decides rather than executing stale authorisation.
    stale = gate.certify(
        permissive(certificate_valid_until=AT - timedelta(minutes=1)), ActionType.SMS, AT
    )
    assert stale.decision is Verdict.BLOCK
    assert "TIME-CERT-WINDOW" in stale.blocking_rule_ids


def test_certificate_pins_the_registry_version(gate: Gate) -> None:
    certificate = gate.certify(permissive(), ActionType.SMS, AT)
    assert certificate.rule_registry_version == gate.registry.version
    assert certificate.rule_registry_version.startswith("rr-")

    # The version is derived from rule content, so editing a threshold moves it
    # and a replay cannot silently re-decide under different rules.
    edited = [
        parse_rule(
            {
                "id": rule.id,
                "class": rule.rule_class.value,
                "scope": sorted(c.value for c in rule.scope),
                "actions": sorted(a.value for a in rule.actions) if rule.actions else None,
                "basis": rule.basis.value,
                "status": rule.status.value,
                "check": rule.check,
                "params": (
                    {**rule.params, "hours": 999} if rule.id == "CD-VOICE" else dict(rule.params)
                ),
                "on_violation": rule.on_violation.value.upper(),
                "rationale": rule.rationale,
            }
        )
        for rule in gate.registry
    ]
    assert RuleRegistry(edited).version != gate.registry.version


# ---------------------------------------------------------------------------
# Registry and check hygiene
# ---------------------------------------------------------------------------
def test_every_check_named_by_a_rule_exists(gate: Gate) -> None:
    for rule in gate.registry:
        assert rule.check in CHECKS, f"{rule.id} names {rule.check}"


def test_no_check_is_dead_code(gate: Gate) -> None:
    """An unused check is a rule someone forgot to write."""
    used = {rule.check for rule in gate.registry}
    assert set(CHECKS) == used, f"unused: {sorted(set(CHECKS) - used)}"


def test_the_registry_has_the_expected_families(gate: Gate) -> None:
    counts: dict[str, int] = {}
    for rule in gate.registry:
        counts[rule.id.split("-")[0]] = counts.get(rule.id.split("-")[0], 0) + 1
    assert counts == {"ABS": 7, "FRZ": 7, "NET": 6, "TIME": 4, "CD": 6, "FREQ": 3}


def test_an_empty_registry_is_refused() -> None:
    """An empty registry would allow everything, which is the worst failure."""
    from arc.gate.registry import RegistryError

    with pytest.raises(RegistryError):
        RuleRegistry([])
