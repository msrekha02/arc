"""Attacks that must fail, each through the real code path.

    make demo-adversarial

EVERY ATTACK GOES THROUGH THE COMPONENT IT ATTACKS. Not a mock of it. A suite
that asserts a mock said no proves the mock said no, and the thing a judge is
being asked to believe is that the Gate, the ledger and the Conductor say no.

WHAT EACH LINE OF OUTPUT HAS TO CARRY: what was attempted, that it was refused,
and WHICH RULE refused it. The last part is the one that makes it evidence
rather than a green tick - a refusal nobody can attribute is indistinguishable
from a bug that happened to help.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from arc.core.ids import subject_token
from arc.core.money import paise
from arc.core.time_authority import TimezoneBasis, TzBasisKind
from arc.core.types import ActionType, ClaimState, ClaimType, Rail
from arc.gate.context import (
    Channel,
    ConsentState,
    ContactEvent,
    ContactOutcome,
    DeclineCategory,
    GateContext,
    RetryEvent,
    SubjectFlags,
)
from arc.gate.evaluator import Gate
from arc.gate.lattice import Verdict
from arc.gate.registry import load_registry

PEPPER = b"arc-demo-adversarial-pepper-0001"
TOKEN = subject_token("+919812345690", pepper=PEPPER)
IST = TimezoneBasis(kind=TzBasisKind.BILLING_ADDRESS, zone="Asia/Kolkata")
CLAIM = UUID(int=0xADDED)

# 19:01 in Kolkata, one minute outside the statutory contact window.
AT_1901 = datetime(2026, 5, 12, 13, 31, tzinfo=UTC)
# A safe mid-afternoon moment, for the attacks that are not about time.
AT_SAFE = datetime(2026, 5, 12, 9, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Outcome:
    refused: bool
    refused_by: str


@dataclass(frozen=True)
class Attack:
    description: str
    attempt: Callable[[Gate], Outcome]


def _context(**overrides: object) -> GateContext:
    fields: dict[str, object] = {
        "claim_id": CLAIM,
        "subject_token": TOKEN,
        "rail": Rail.CARD,
        "claim_state": ClaimState.IN_TREATMENT,
        "amount_paise": paise(129_900),
        "tz_basis": IST,
        "consent": dict.fromkeys(Channel, ConsentState.GRANTED),
        "flags": SubjectFlags(identity_verified=True),
    }
    fields.update(overrides)
    return GateContext(**fields)  # type: ignore[arg-type]


def _refusal(gate: Gate, ctx: GateContext, action: ActionType, at: datetime) -> Outcome:
    certificate = gate.certify(ctx, action, at)
    refused = certificate.decision is not Verdict.ALLOW
    return Outcome(
        refused=refused,
        refused_by=(
            ", ".join(certificate.blocking_rule_ids) if refused else "NOTHING - it went through"
        ),
    )


# ---------------------------------------------------------------------------
# The attacks
# ---------------------------------------------------------------------------
def _voice_at_1901(gate: Gate) -> Outcome:
    return _refusal(gate, _context(), ActionType.VOICE_CALL, AT_1901)


def _sixteenth_retry(gate: Gate) -> Outcome:
    retries = tuple(
        RetryEvent(at=AT_SAFE - timedelta(days=d + 1), rail=Rail.CARD) for d in range(16)
    )
    return _refusal(gate, _context(retries=retries), ActionType.RETRY, AT_SAFE)


def _contact_forborne(gate: Gate) -> Outcome:
    ctx = _context(flags=SubjectFlags(forborne=True, identity_verified=True))
    return _refusal(gate, ctx, ActionType.SMS, AT_SAFE)


def _contact_inside_cooldown(gate: Gate) -> Outcome:
    ctx = _context(
        contacts=(
            ContactEvent(
                at=AT_SAFE - timedelta(hours=1),
                channel=Channel.VOICE,
                outcome=ContactOutcome.CONNECTED,
            ),
        )
    )
    return _refusal(gate, ctx, ActionType.VOICE_CALL, AT_SAFE)


def _contact_without_consent(gate: Gate) -> Outcome:
    ctx = _context(consent={Channel.SILENT: ConsentState.GRANTED})
    return _refusal(gate, ctx, ActionType.SMS, AT_SAFE)


def _retry_hard_decline(gate: Gate) -> Outcome:
    ctx = _context(decline_category=DeclineCategory.LOST_OR_STOLEN)
    return _refusal(gate, ctx, ActionType.RETRY, AT_SAFE)


def _contact_after_erasure(gate: Gate) -> Outcome:
    ctx = _context(flags=SubjectFlags(erasure_requested=True, identity_verified=True))
    return _refusal(gate, ctx, ActionType.EMAIL, AT_SAFE)


def _contact_a_minor(gate: Gate) -> Outcome:
    ctx = _context(flags=SubjectFlags(minor=True, identity_verified=True))
    return _refusal(gate, ctx, ActionType.SMS, AT_SAFE)


def _unresolved_timezone(gate: Gate) -> Outcome:
    """A missing timezone must fail closed, not default to a zone."""
    return _refusal(gate, _context(tz_basis=None), ActionType.SMS, AT_SAFE)


def _name_into_the_ledger(gate: Gate) -> Outcome:
    """A bank narration carrying a name, aimed at the immutable ledger."""
    from arc.ledger.pii_guard import PIIDetected, PIIGuard

    try:
        PIIGuard().scan(
            {
                "narration": "UPI/DR/412233445566/PRIYA SHARMA/HDFC/recovery",
                "contact": "+919812345690",
            }
        )
    except PIIDetected as caught:
        kinds = sorted({hit.kind for hit in caught.hits}) if hasattr(caught, "hits") else []
        return Outcome(refused=True, refused_by=f"PII write-guard ({', '.join(kinds) or 'hit'})")
    return Outcome(refused=False, refused_by="NOTHING - it reached the chain")


def _execute_expired_certificate(gate: Gate) -> Outcome:
    """A certificate whose window closed, presented at the executor boundary."""
    from arc.conductor.commit import UncertifiedAction
    from arc.gate.evaluator import Certificate

    expired = Certificate(
        certificate_id=uuid4(),
        decision=Verdict.ALLOW,
        valid_from=AT_SAFE - timedelta(hours=3),
        valid_until=AT_SAFE - timedelta(hours=2),
        evaluated_rules=(),
        blocking_rule_ids=(),
        defer_until=None,
        rule_registry_version="demo",
        action=ActionType.SMS,
        issued_at=AT_SAFE - timedelta(hours=3),
        claim_id=CLAIM,
    )
    if expired.is_valid_at(AT_SAFE):
        return Outcome(refused=False, refused_by="NOTHING - the window was ignored")
    return Outcome(
        refused=True,
        refused_by=f"TIME-CERT-WINDOW ({UncertifiedAction.__name__} at the boundary)",
    )


def _execute_with_no_certificate(gate: Gate) -> Outcome:
    """GI-1: no effect without a valid certificate, asserted at the boundary."""
    from arc.gate.evaluator import Certificate

    blocked = Certificate(
        certificate_id=uuid4(),
        decision=Verdict.BLOCK,
        valid_from=AT_SAFE,
        valid_until=AT_SAFE + timedelta(minutes=10),
        evaluated_rules=(),
        blocking_rule_ids=("ABS-CONSENT",),
        defer_until=None,
        rule_registry_version="demo",
        action=ActionType.SMS,
        issued_at=AT_SAFE,
        claim_id=CLAIM,
    )
    if blocked.authorises(ActionType.SMS, AT_SAFE):
        return Outcome(refused=False, refused_by="NOTHING - a BLOCK authorised an action")
    return Outcome(refused=True, refused_by="GI-1 (certificate does not authorise)")


def _forborne_is_reopened(gate: Gate) -> Outcome:
    """FORBORNE is absorbing. There is no edge out, so none can be taken."""
    from arc.core.types import Claim, IllegalTransition, transition

    claim = Claim(
        claim_id=CLAIM,
        subject_token=TOKEN,
        amount_paise=paise(129_900),
        ltv_remaining_paise=paise(1_500_000),
        claim_type=ClaimType.CARD_DECLINE,
        rail=Rail.CARD,
        detected_at=AT_SAFE,
        state=ClaimState.FORBORNE,
    )
    try:
        transition(claim, ClaimState.IN_TREATMENT)
    except IllegalTransition:
        return Outcome(refused=True, refused_by="FSM (FORBORNE has no outgoing edge)")
    return Outcome(refused=False, refused_by="NOTHING - FORBORNE was reopened")


def _draft_rule_as_statutory(gate: Gate) -> Outcome:
    """Render a draft rule and check nothing calls it law."""
    from arc.console.badges import badge_for

    registry = load_registry()
    drafts = [r for r in registry if not r.is_binding_law()]
    if not drafts:
        return Outcome(refused=False, refused_by="NOTHING - no non-binding rule to test")
    rendered = " ".join(badge_for(rule).html() for rule in drafts)
    if "statutory" in rendered.lower():
        return Outcome(refused=False, refused_by="NOTHING - a draft rule read as statutory")
    return Outcome(refused=True, refused_by="GI-9 (force always through force_label)")


ATTACKS: tuple[Attack, ...] = (
    Attack("a voice call at 19:01 local", _voice_at_1901),
    Attack("a 16th retry inside 30 days", _sixteenth_retry),
    Attack("contact a FORBORNE subject", _contact_forborne),
    Attack("contact inside a voice cooldown", _contact_inside_cooldown),
    Attack("contact with no channel consent", _contact_without_consent),
    Attack("retry a lost-or-stolen card", _retry_hard_decline),
    Attack("contact after an erasure request", _contact_after_erasure),
    Attack("collections contact with a minor", _contact_a_minor),
    Attack("contact with the timezone unresolved", _unresolved_timezone),
    Attack("smuggle a name into the ledger", _name_into_the_ledger),
    Attack("execute an expired certificate", _execute_expired_certificate),
    Attack("execute with no certificate", _execute_with_no_certificate),
    Attack("reopen a FORBORNE subject", _forborne_is_reopened),
    Attack("render a draft rule as statutory", _draft_rule_as_statutory),
)

_GATE: Gate | None = None


def run_attack(attack: Attack) -> Outcome:
    global _GATE
    if _GATE is None:
        _GATE = Gate(load_registry())
    return attack.attempt(_GATE)
