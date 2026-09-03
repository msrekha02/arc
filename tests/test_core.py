"""M1 acceptance gate: domain types, the FSM, money, time, and the clock scan.

The five named tests in the build doc are:

    test_illegal_fsm_transition_raises
    test_forborne_is_absorbing
    test_money_rejects_float
    test_half_open_window_boundary
    test_no_direct_datetime_now

Everything else here exists because these types are frozen after this
milestone, so a gap left now is inherited by every later one.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pytest
from arc.core.ids import claim_id, is_subject_token, subject_token
from arc.core.money import RUPEE_SIGN, NotMoney, Paise, format_inr, from_rupees, paise
from arc.core.time_authority import (
    FrozenTimeAuthority,
    NotUTC,
    TimeAuthority,
    TimezoneBasis,
    TzBasisKind,
    Window,
    ensure_utc,
    rolling_window,
)
from arc.core.types import (
    ABSORBING_STATES,
    LEGAL_TRANSITIONS,
    ActionType,
    Cause,
    CauseLabel,
    CauseLayer,
    Claim,
    ClaimState,
    ClaimType,
    CohortVerdict,
    DiagnosisPath,
    IllegalTransition,
    Rail,
    transition,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PEPPER = b"m1-acceptance-gate-pepper-000000"
TOKEN = subject_token("+919876543210", pepper=PEPPER)
KOLKATA = TimezoneBasis(kind=TzBasisKind.DECLARED, zone="Asia/Kolkata")

# A fixed instant. Nothing in this file reads a clock, which is also what makes
# the scan below able to cover the tests directory without exempting it.
T0 = datetime(2026, 3, 17, 6, 30, tzinfo=UTC)


def make_claim(**overrides: Any) -> Claim:
    fields: dict[str, Any] = {
        "claim_id": claim_id("razorpay", "evt_0001"),
        "subject_token": TOKEN,
        "amount_paise": paise(129900),
        "ltv_remaining_paise": paise(1500000),
        "claim_type": ClaimType.MANDATE_FAILURE,
        "rail": Rail.ENACH,
        "detected_at": T0,
        "evidence_hash": b"\x11" * 32,
    }
    fields.update(overrides)
    return Claim(**fields)


# ---------------------------------------------------------------------------
# Gate test 1
# ---------------------------------------------------------------------------
def test_illegal_fsm_transition_raises() -> None:
    detected = make_claim()

    # Skipping diagnosis to go straight to money movement.
    with pytest.raises(IllegalTransition):
        transition(detected, ClaimState.RECOVERED)

    # A state cannot re-enter itself; there is no self-edge in the table.
    with pytest.raises(IllegalTransition):
        transition(detected, ClaimState.DETECTED)

    # Escalation cannot skip the treatment path.
    with pytest.raises(IllegalTransition):
        transition(detected, ClaimState.ESCALATED)

    # A dispute cannot be written off through a path that does not exist.
    disputed = make_claim(state=ClaimState.DISPUTED)
    with pytest.raises(IllegalTransition):
        transition(disputed, ClaimState.RECOVERED)

    # The one legal edge out of DETECTED still works.
    assert transition(detected, ClaimState.DIAGNOSED).state is ClaimState.DIAGNOSED


# ---------------------------------------------------------------------------
# Gate test 2
# ---------------------------------------------------------------------------
def test_forborne_is_absorbing() -> None:
    """Every transition out of FORBORNE raises, including to WRITTEN_OFF."""
    forborne = make_claim(state=ClaimState.FORBORNE)

    assert LEGAL_TRANSITIONS[ClaimState.FORBORNE] == frozenset()

    for target in ClaimState:
        with pytest.raises(IllegalTransition):
            transition(forborne, target)

    # No expected-value argument reopens it: there is no bypass argument on
    # transition() to reach for.
    with pytest.raises(IllegalTransition):
        transition(forborne, ClaimState.WRITTEN_OFF)


def test_written_off_is_absorbing() -> None:
    written_off = make_claim(state=ClaimState.WRITTEN_OFF)

    assert LEGAL_TRANSITIONS[ClaimState.WRITTEN_OFF] == frozenset()
    for target in ClaimState:
        with pytest.raises(IllegalTransition):
            transition(written_off, target)


def test_absorbing_states_are_exactly_forborne_and_written_off() -> None:
    expected = frozenset({ClaimState.FORBORNE, ClaimState.WRITTEN_OFF})
    assert expected == ABSORBING_STATES


def test_transition_table_is_total_and_closed() -> None:
    """Every state has an entry, and no entry points outside the enum."""
    assert set(LEGAL_TRANSITIONS) == set(ClaimState)
    for frm, onward in LEGAL_TRANSITIONS.items():
        assert frm not in onward, f"{frm} has a self-edge"
        for to in onward:
            assert isinstance(to, ClaimState)


def test_transition_returns_a_new_claim_and_leaves_the_original_alone() -> None:
    detected = make_claim()
    diagnosed = transition(detected, ClaimState.DIAGNOSED)

    assert detected.state is ClaimState.DETECTED
    assert diagnosed.state is ClaimState.DIAGNOSED
    assert diagnosed.claim_id == detected.claim_id


# ---------------------------------------------------------------------------
# Gate test 3
# ---------------------------------------------------------------------------
def test_money_rejects_float() -> None:
    with pytest.raises(NotMoney):
        paise(1299.0)
    with pytest.raises(NotMoney):
        paise(12.5)
    with pytest.raises(NotMoney):
        from_rupees(1.5)

    # bool is a subclass of int and would otherwise become one paise.
    with pytest.raises(NotMoney):
        paise(True)

    # The claim is the real boundary: a float amount must not construct.
    with pytest.raises(TypeError):
        make_claim(amount_paise=1299.0)
    with pytest.raises(TypeError):
        make_claim(ltv_remaining_paise=15000.75)

    # Nor may a float hide inside evidence.
    with pytest.raises(TypeError):
        make_claim(evidence_structured={"amount_attempted": 1299.0})

    # And negative money is rejected rather than silently accepted.
    with pytest.raises(ValueError):
        make_claim(amount_paise=-1)

    assert paise(1299) == 1299
    assert from_rupees(1299) == 129900


def test_format_inr_is_display_only_and_groups_indian_style() -> None:
    assert format_inr(Paise(129900)) == f"{RUPEE_SIGN}1,299.00"
    assert format_inr(Paise(1234567800)) == f"{RUPEE_SIGN}1,23,45,678.00"
    assert format_inr(Paise(99)) == f"{RUPEE_SIGN}0.99"
    assert format_inr(Paise(-50000)) == f"-{RUPEE_SIGN}500.00"


# ---------------------------------------------------------------------------
# Gate test 4
# ---------------------------------------------------------------------------
def test_half_open_window_boundary() -> None:
    """[t - 7d, t) includes the lower bound and excludes the upper."""
    t = T0
    window = rolling_window(t, timedelta(days=7))

    assert window.start == t - timedelta(days=7)
    assert window.end == t

    assert window.contains(t - timedelta(days=7)) is True
    assert window.contains(t - timedelta(microseconds=1)) is True
    assert window.contains(t) is False
    assert window.contains(t + timedelta(microseconds=1)) is False
    assert window.contains(t - timedelta(days=7, microseconds=1)) is False

    # An event at exactly t belongs to the next window, not this one.
    later = rolling_window(t + timedelta(days=7), timedelta(days=7))
    assert later.contains(t) is True

    assert window.duration == timedelta(days=7)


def test_window_rejects_naive_and_non_utc_datetimes() -> None:
    naive = datetime(2026, 3, 17, 6, 30)
    with pytest.raises(NotUTC):
        rolling_window(naive, timedelta(days=7))

    with pytest.raises(NotUTC):
        Window(T0, naive)

    window = rolling_window(T0, timedelta(days=7))
    with pytest.raises(NotUTC):
        window.contains(naive)

    with pytest.raises(NotUTC):
        ensure_utc(T0.astimezone(ZoneInfo("Asia/Kolkata")))


def test_rolling_window_requires_positive_duration() -> None:
    with pytest.raises(ValueError):
        rolling_window(T0, timedelta(0))
    with pytest.raises(ValueError):
        rolling_window(T0, timedelta(days=-1))


# ---------------------------------------------------------------------------
# Gate test 5
# ---------------------------------------------------------------------------
# Resolved dotted names that read wall-clock time. A file may reach these only
# if it is the Time Authority itself.
FORBIDDEN_CLOCK_CALLS = frozenset(
    {
        "datetime.datetime.now",
        "datetime.datetime.utcnow",
        "datetime.datetime.today",
        "datetime.date.today",
        "time.time",
        "time.time_ns",
    }
)

CLOCK_OWNER = Path("arc/core/time_authority.py")

SKIP_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
}


def _python_files() -> list[Path]:
    return [
        path
        for path in sorted(REPO_ROOT.rglob("*.py"))
        if not SKIP_DIRS & set(path.relative_to(REPO_ROOT).parts)
    ]


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    """Map each locally bound name to the absolute module path it refers to.

    Resolving through the import statements is what makes the scan robust to
    aliasing: `from datetime import datetime as dt` then `dt.now()` is caught.
    """
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    bound[root] = root
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _resolve(name: str, bound: dict[str, str]) -> str:
    head, _, rest = name.partition(".")
    if head not in bound:
        return name
    return f"{bound[head]}.{rest}" if rest else bound[head]


def clock_reads(path: Path) -> list[tuple[int, str]]:
    """Every wall-clock read in one file, as (line, resolved name)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bound = _import_bindings(tree)

    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        name = _dotted(node)
        if name is None:
            continue
        resolved = _resolve(name, bound)
        if resolved in FORBIDDEN_CLOCK_CALLS:
            hits.append((node.lineno, resolved))
    return hits


def test_no_direct_datetime_now() -> None:
    """TimeAuthority is the only caller of a wall clock in the whole repo."""
    scanned = _python_files()
    assert len(scanned) >= 5, "the scan found almost no files; the walk is broken"

    offenders: list[str] = []
    for path in scanned:
        relative = path.relative_to(REPO_ROOT)
        if relative == CLOCK_OWNER:
            continue
        offenders += [
            f"{relative.as_posix()}:{line} calls {name}" for line, name in clock_reads(path)
        ]

    assert not offenders, "only TimeAuthority may read a clock:\n" + "\n".join(offenders)

    # The exemption must still be earning its keep. If the Time Authority stops
    # reading a clock, the allowlist is stale and this test says so.
    assert clock_reads(REPO_ROOT / CLOCK_OWNER), f"{CLOCK_OWNER} no longer reads a clock"


def test_clock_scan_catches_aliased_and_indirect_reads(tmp_path: Path) -> None:
    """The scan is only worth anything if it survives an evasion attempt."""
    evasions = [
        "import datetime\nx = datetime.datetime.now()\n",
        "from datetime import datetime\nx = datetime.now()\n",
        "from datetime import datetime as dt\nx = dt.now()\n",
        "import datetime as d\nx = d.datetime.utcnow()\n",
        "from datetime import date\nx = date.today()\n",
        "import time\nx = time.time()\n",
        "from time import time\nx = time()\n",
        # Squirrelled away and called later, never as a direct call node.
        "from datetime import datetime\nsneaky = datetime.now\nx = sneaky()\n",
    ]
    for index, source in enumerate(evasions):
        path = tmp_path / f"evasion_{index}.py"
        path.write_text(source, encoding="utf-8")
        assert clock_reads(path), f"scan missed:\n{source}"


def test_clock_scan_does_not_flag_injected_clocks(tmp_path: Path) -> None:
    """`clock.now()` is the correct pattern and must not be a false positive."""
    allowed = [
        "def f(clock):\n    return clock.now()\n",
        "def f(self):\n    return self.now()\n",
        "from datetime import datetime\ndef f(at: datetime) -> datetime:\n    return at\n",
        "from datetime import date\ndef f(d: date) -> date:\n    return d\n",
        "import time\ndef f():\n    return time.sleep(1)\n",
    ]
    for index, source in enumerate(allowed):
        path = tmp_path / f"allowed_{index}.py"
        path.write_text(source, encoding="utf-8")
        assert not clock_reads(path), f"false positive on:\n{source}"


# ---------------------------------------------------------------------------
# Time Authority behaviour
# ---------------------------------------------------------------------------
def test_frozen_clock_does_not_move() -> None:
    clock = FrozenTimeAuthority(T0)
    assert clock.now() == T0
    assert clock.now() == T0

    clock.advance(timedelta(hours=3))
    assert clock.now() == T0 + timedelta(hours=3)


def test_local_conversion_uses_the_recorded_tz_basis() -> None:
    clock = FrozenTimeAuthority(T0)
    local = clock.local(T0, KOLKATA)

    assert local.hour == 12  # 06:30 UTC is 12:00 IST
    assert local.minute == 0
    assert local.utcoffset() == timedelta(hours=5, minutes=30)


def test_next_legal_window_returns_now_when_already_inside_the_window() -> None:
    clock = FrozenTimeAuthority(T0)
    assert clock.next_legal_window(T0, KOLKATA) == T0  # 12:00 IST


def test_next_legal_window_defers_out_of_hours() -> None:
    clock = FrozenTimeAuthority(T0)

    # 20:30 IST, past the 19:00 close -> 08:00 IST the following morning.
    late = datetime(2026, 3, 17, 15, 0, tzinfo=UTC)
    opens = clock.next_legal_window(late, KOLKATA)
    assert clock.local(opens, KOLKATA).hour == 8
    assert clock.local(opens, KOLKATA).date() == clock.local(late, KOLKATA).date() + timedelta(
        days=1
    )

    # 05:30 IST, before the 08:00 open -> 08:00 IST the same morning.
    early = datetime(2026, 3, 17, 0, 0, tzinfo=UTC)
    opens = clock.next_legal_window(early, KOLKATA)
    assert clock.local(opens, KOLKATA).hour == 8
    assert clock.local(opens, KOLKATA).date() == clock.local(early, KOLKATA).date()


def test_contact_window_boundaries_are_half_open() -> None:
    clock = FrozenTimeAuthority(T0)

    # 19:00:00 IST exactly is outside the window and defers to the next day.
    at_close = datetime(2026, 3, 17, 13, 30, tzinfo=UTC)
    assert clock.local(at_close, KOLKATA).hour == 19
    assert clock.next_legal_window(at_close, KOLKATA) != at_close

    # 08:00:00 IST exactly is inside it.
    at_open = datetime(2026, 3, 17, 2, 30, tzinfo=UTC)
    assert clock.local(at_open, KOLKATA).hour == 8
    assert clock.next_legal_window(at_open, KOLKATA) == at_open


def test_bank_holidays_are_injected_not_fetched() -> None:
    holiday = datetime(2026, 3, 25, tzinfo=UTC).date()
    clock = TimeAuthority(holidays={"IN": frozenset({holiday})})

    assert clock.is_bank_holiday(holiday, "IN") is True
    assert clock.is_bank_holiday(holiday, "SG") is False
    assert clock.is_bank_holiday(holiday + timedelta(days=1), "IN") is False

    assert TimeAuthority().is_bank_holiday(holiday, "IN") is False


def test_unknown_timezone_fails_closed() -> None:
    with pytest.raises(ZoneInfoNotFoundError):
        TimezoneBasis(kind=TzBasisKind.DECLARED, zone="Mars/Olympus_Mons")


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------
def test_subject_token_is_deterministic_and_one_way() -> None:
    again = subject_token("+919876543210", pepper=PEPPER)
    assert again == TOKEN
    assert is_subject_token(TOKEN)

    # The raw identifier does not appear in the token.
    assert "9876543210" not in TOKEN

    # A different pepper gives a different token for the same subject.
    other = subject_token("+919876543210", pepper=b"a-completely-different-pepper!!")
    assert other != TOKEN

    assert subject_token("  +919876543210  ", pepper=PEPPER) == TOKEN


def test_claim_ids_are_derived_from_the_source_event() -> None:
    assert claim_id("razorpay", "evt_1") == claim_id("razorpay", "evt_1")
    assert claim_id("razorpay", "evt_1") != claim_id("razorpay", "evt_2")
    # Delimited join: ('a','bc') and ('ab','c') must not collide.
    assert claim_id("a", "bc") != claim_id("ab", "c")


def test_raw_identifier_is_rejected_as_a_subject_token() -> None:
    for raw in ["+919876543210", "priya@example.com", "sub_short", "", "Priya Sharma"]:
        assert not is_subject_token(raw)
        with pytest.raises(ValueError):
            make_claim(subject_token=raw)


# ---------------------------------------------------------------------------
# Claim and Cause construction
# ---------------------------------------------------------------------------
def test_claim_requires_timezone_aware_utc() -> None:
    with pytest.raises(NotUTC):
        make_claim(detected_at=datetime(2026, 3, 17, 6, 30))


def test_evidence_structured_keeps_free_text_out() -> None:
    narration = "ACH DR RETURN PRIYA SHARMA A/C 001234567890 INSUFF FUNDS 17MAR"

    # Too long for a structured field: it belongs behind evidence_ref.
    with pytest.raises(ValueError):
        make_claim(evidence_structured={"narration": narration * 3})

    # Nested containers are not a closed vocabulary.
    with pytest.raises(TypeError):
        make_claim(evidence_structured={"nested": {"a": 1}})

    # Keys must be vocabulary names, not free-form strings.
    with pytest.raises(ValueError):
        make_claim(evidence_structured={"bank narration": "x"})

    ok = make_claim(
        evidence_structured={
            "decline_code": "MAC03",
            "mandate_cap_paise": 500000,
            "is_first_charge_of_cycle": True,
            "prior_codes": ["MAC02", "MAC03"],
        }
    )
    assert ok.evidence_structured["decline_code"] == "MAC03"


def test_claim_evidence_is_immutable_after_construction() -> None:
    claim = make_claim(evidence_structured={"decline_code": "MAC03"})
    with pytest.raises(TypeError):
        claim.evidence_structured["decline_code"] = "MAC02"  # type: ignore[index]


def test_claim_requires_a_sha256_evidence_hash() -> None:
    with pytest.raises(ValueError):
        make_claim(evidence_hash=b"too-short")
    with pytest.raises(ValueError):
        make_claim(evidence_hash=b"\x00" * 31)


def test_cause_confidence_is_a_probability() -> None:
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            Cause(
                label=CauseLabel.INSUFFICIENT_FUNDS,
                layer=CauseLayer.CUSTOMER,
                confidence=bad,
                derived_from=DiagnosisPath.CODE_MAP,
                cohort_power=CohortVerdict.NORMAL,
            )


def test_llm_derived_cause_confidence_is_capped() -> None:
    with pytest.raises(ValueError):
        Cause(
            label=CauseLabel.INSUFFICIENT_FUNDS,
            layer=CauseLayer.CUSTOMER,
            confidence=0.95,
            derived_from=DiagnosisPath.LLM,
            cohort_power=CohortVerdict.NORMAL,
        )

    capped = Cause(
        label=CauseLabel.INSUFFICIENT_FUNDS,
        layer=CauseLayer.CUSTOMER,
        confidence=0.70,
        derived_from=DiagnosisPath.LLM,
        cohort_power=CohortVerdict.NORMAL,
    )
    assert capped.confidence == 0.70


def test_unknown_cause_is_representable_with_zero_confidence() -> None:
    unknown = Cause(
        label=CauseLabel.UNKNOWN,
        layer=CauseLayer.UNKNOWN,
        confidence=0.0,
        derived_from=DiagnosisPath.CODE_MAP,
        cohort_power=CohortVerdict.INSUFFICIENT_POWER,
    )
    assert unknown.layer is CauseLayer.UNKNOWN


# ---------------------------------------------------------------------------
# The closed enums
# ---------------------------------------------------------------------------
def test_action_type_is_closed_at_thirteen() -> None:
    assert len(ActionType) == 13
    assert [a.value for a in ActionType] == [
        "do_nothing",
        "retry",
        "card_updater",
        "mandate_re_register",
        "rail_fallback",
        "whatsapp_utility",
        "sms",
        "email",
        "payment_link",
        "voice_call",
        "instalment_offer",
        "human_handoff",
        "statutory_notice",
    ]


def test_claim_state_matches_the_frozen_contract() -> None:
    assert len(ClaimState) == 13
    assert [s.value for s in ClaimState] == [
        "detected",
        "diagnosed",
        "suppressed",
        "self_healing",
        "planned",
        "in_treatment",
        "promised",
        "escalated",
        "disputed",
        "recovered",
        "reversed",
        "written_off",
        "forborne",
    ]


def test_insufficient_power_is_distinct_from_normal() -> None:
    assert CohortVerdict.INSUFFICIENT_POWER is not CohortVerdict.NORMAL
    assert CohortVerdict.INSUFFICIENT_POWER != CohortVerdict.NORMAL
