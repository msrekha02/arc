"""M14 acceptance gate - the four console screens.

    all four render from real data
    the replay trace reads as prose, not JSON
    the headline cannot be shown without its guardrails
    prevention stays a separate line from recovery
    no rule is rendered as carrying more force than its instrument has

RENDERED FROM A REAL RUN, NOT A FIXTURE. The module-scoped `data` fixture is a
genuine `run_all` over the frozen world plus a genuine Sentinel pass over the
batch. A console rendered from a fixture is a screenshot, and a screenshot
cannot go wrong in the same way the system does - which is the only interesting
thing a console test can check.

THE SUITE IS NOT VACUOUSLY GREEN. `test_a_draft_rule_rendered_as_statutory_is_
caught` builds a rule whose basis says statutory and whose status says draft,
renders it the way a careless call site would, and asserts the audit catches it
on the GI-9 assertion rather than on some incidental difference.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest
from arc.console.badges import (
    ForceOverstated,
    Tone,
    assert_every_rule_shown,
    assert_no_overstated_force,
    badge_for,
    honest_mix,
    not_in_force,
)
from arc.console.build import build
from arc.console.replay import narrate
from arc.console.screens import BatchView
from arc.core.money import paise
from arc.gate.registry import RuleBasis, RuleStatus, load_registry
from arc.proving_ground.arms import Arm
from arc.proving_ground.metrics import GuardrailsMissing, assert_guardrails_present

# Small enough that the suite pays a few seconds for a real run, large enough
# that the injected outages are present and the diagnosis split is non-trivial.
POPULATION = 600
CYCLES = 2


@pytest.fixture(scope="module")
def registry():
    return load_registry()


@pytest.fixture(scope="module")
def data():
    return build(seed=1, size=POPULATION, cycles=CYCLES)


@pytest.fixture(scope="module")
def screens(data):
    return data.screens()


def _visible_text(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


# ---------------------------------------------------------------------------
# Gate 1 - all four render from real data
# ---------------------------------------------------------------------------
def test_all_four_screens_render_from_real_data(screens, data) -> None:
    """Four documents, each with content, from one genuine run."""
    assert set(screens) == {
        "batch.html",
        "firewall.html",
        "scoreboard.html",
        "replay.html",
        "index.html",
    }
    for name, html in screens.items():
        assert html.startswith("<!doctype html>"), f"{name} is not a document"
        assert "<title>" in html and "</html>" in html, f"{name} is malformed"
        assert len(_visible_text(html)) > 200, f"{name} rendered almost nothing"

    # From a real run, not a fixture: the numbers agree with the harness.
    assert data.result.subjects > 100
    assert data.batch.claims == sum(len(c.claims) for c in data.result.cases)


def test_batch_view_shows_the_outage_suppression(data) -> None:
    """The number this screen exists for.

    Claims a detected issuer outage took off the contact path entirely. The
    frozen world injects two outages, so a run that found none would mean the
    Sentinel is not being asked at the moment the failure happened.
    """
    batch = data.batch
    assert batch.issuer > 0, (
        "no claim was diagnosed issuer-layer on a batch containing two injected "
        "outages; the Sentinel is being asked at the wrong moment"
    )
    assert batch.suppressed_by_outage > 0
    assert batch.suppressed_by_outage <= batch.issuer

    text = _visible_text(batch.render())
    assert "suppressed by outage" in text
    assert "received no contact of any kind" in text
    assert "naive fixed-schedule arm messaged" in text, (
        "the contrast with the naive arm is the point of the number and is missing"
    )
    # The three self-monitoring diagnostics reach the screen too.
    assert "without cohort power" in text


def test_batch_view_refuses_a_split_that_does_not_cover_the_batch() -> None:
    """A claim with no layer has not been diagnosed and cannot be dropped."""
    with pytest.raises(ValueError, match="diagnosis split covers"):
        BatchView(
            seed=1,
            claims=100,
            subjects=80,
            at_risk_paise=paise(1),
            issuer=1,
            merchant=1,
            customer=1,
            unknown=1,
            suppressed_by_outage=0,
            self_healing=0,
            naive_contacted_same_claims=0,
        )

    with pytest.raises(ValueError, match="cannot exceed"):
        BatchView(
            seed=1,
            claims=4,
            subjects=4,
            at_risk_paise=paise(1),
            issuer=1,
            merchant=1,
            customer=1,
            unknown=1,
            suppressed_by_outage=3,
            self_healing=0,
            naive_contacted_same_claims=0,
        )


# ---------------------------------------------------------------------------
# Gate 2 - the compliance firewall is honest
# ---------------------------------------------------------------------------
def test_firewall_shows_the_honest_mix(screens, registry) -> None:
    """The counts, stated, including the ones that are not flattering."""
    text = _visible_text(screens["firewall.html"])
    mix = honest_mix(registry)

    assert mix["total"] == 33
    assert mix["statutory"] == 8
    assert mix["network_rule"] == 4
    assert mix["policy_choice"] == 21
    assert mix["stricter_than_binding_minimum"] == 25

    assert "33 rules" in text
    assert "8 statutory" in text
    assert "4 network" in text
    assert "21 our own policy choice" in text
    assert "stricter than the binding minimum in 25 places" in text.lower()


def test_firewall_names_the_three_rules_that_are_not_in_force(screens, registry) -> None:
    """A summary that lists only the flattering rules stops being believed."""
    pending = not_in_force(registry)
    assert {b.rule_id for b in pending} == {"ABS-EMPLOYER", "NET-RETRY-30D", "TIME-DAY"}

    text = _visible_text(screens["firewall.html"])
    assert "Not in force, and applied anyway" in text
    for badge in pending:
        assert badge.rule_id in text, f"{badge.rule_id} is not named on the panel"
        assert badge.text in text, f"{badge.rule_id} is named without saying why"
        assert badge.tone is Tone.PROVISIONAL


def test_firewall_shows_the_funnel_and_every_rule(screens, registry) -> None:
    text = _visible_text(screens["firewall.html"])
    for word in ("proposed", "blocked", "deferred", "executed"):
        assert word in text
    assert_every_rule_shown(screens["firewall.html"], registry)


def test_no_rule_is_rendered_as_carrying_more_force_than_it_has(screens, registry) -> None:
    """GI-9, checked on every screen rather than on the panel alone."""
    for html in screens.values():
        assert_no_overstated_force(html, registry)


def test_force_text_comes_from_the_registry_not_from_the_basis(registry) -> None:
    """Every badge's words are `Rule.force_label()` verbatim.

    Not "consistent with" it: identical to it. A renderer that paraphrased
    would be a second place the honesty rule lives, and the second place is the
    one that drifts.
    """
    for rule in registry:
        badge = badge_for(rule)
        assert badge.text == rule.force_label()
        if rule.is_binding_law():
            assert badge.tone is Tone.LAW
        else:
            assert badge.tone is not Tone.LAW, (
                f"{rule.id} is not binding law but was given the law tone; colour is a "
                f"claim about force too"
            )


def test_a_draft_rule_rendered_as_statutory_is_caught(registry) -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    A rule whose basis says statutory and whose status says draft is the exact
    trap GI-9 exists for: the honest description is "draft, not in force", and
    a renderer reading `basis` prints "statutory". Both halves are checked -
    that the chokepoint gets it right, and that the audit catches a call site
    that bypassed the chokepoint.
    """
    real = registry["TIME-WINDOW"]
    planted = replace(real, id="PLANT-DRAFT", status=RuleStatus.DRAFT)
    assert planted.basis is RuleBasis.STATUTORY
    assert not planted.is_binding_law()

    # 1. The chokepoint refuses to call it statutory.
    badge = badge_for(planted)
    assert "statutory" not in badge.text.lower(), (
        "a draft rule with a statutory basis was described as statutory"
    )
    assert badge.tone is Tone.PROVISIONAL

    # 2. A call site that bypassed the chokepoint is caught by the audit.
    careless = f'<span data-rule="{planted.id}">{planted.basis.value}, in force</span>'

    class _OneRule:
        version = "plant"

        def __iter__(self):
            return iter([planted])

        def __len__(self):
            return 1

    with pytest.raises(ForceOverstated) as caught:
        assert_no_overstated_force(careless, _OneRule())  # type: ignore[arg-type]

    message = str(caught.value)
    assert "GI-9" in message, (
        f"the assertion fired for the wrong reason: {message!r}. It must identify "
        f"this as an overstatement of regulatory force"
    )
    assert planted.id in message and "status=draft" in message

    # 3. And the honest rendering of the same rule passes the same audit.
    honest = f'<span data-rule="{planted.id}">{badge.text}</span>'
    assert_no_overstated_force(honest, _OneRule())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Gate 3 - the scoreboard
# ---------------------------------------------------------------------------
def test_headline_cannot_be_shown_without_guardrails(data, screens) -> None:
    """Structural, not conventional.

    The screen renders from `Scoreboard.to_dict()`, which refuses to serialise
    a recovery figure without the full guardrail block. There is no path to the
    number that does not pass the refusal.
    """
    payload = data.scoreboard.payload()
    for arm in payload["arms"]:
        assert_guardrails_present(arm)

    text = _visible_text(screens["scoreboard.html"])
    for column in ("recovered", "compl/1k", "optout/1k", "cancel", "ptp kept"):
        assert column in text, f"{column} is missing from the scoreboard"

    # The refusal is live, not decorative.
    with pytest.raises(GuardrailsMissing):
        assert_guardrails_present({"recovered_paise": 1_000_000})


def test_prevention_is_a_separate_line_from_recovery(data, screens) -> None:
    """Money that never failed was never recovered."""
    payload = data.scoreboard.payload()
    for arm in payload["arms"]:
        assert "prevented_paise" in arm
        assert arm["incremental_paise"] == (
            arm["recovered_paise"] - arm["comparator_recovered_paise"]
        ), "something other than recovery has been folded into the headline"

    text = _visible_text(screens["scoreboard.html"])
    assert "prevented" in text
    assert "NEVER added into recovery" in text


def test_scoreboard_shows_both_estimator_errors(data, screens) -> None:
    """Both seeds, and the judged one is the worse one.

    Reporting only the develop figure would be choosing the seed after seeing
    the result, which is what the three-seed discipline exists to stop.
    """
    view = data.scoreboard
    assert view.dr_error_judged > view.dr_error_develop, (
        "the judged error is not the worse one, so showing both proves nothing"
    )
    text = _visible_text(screens["scoreboard.html"])
    assert f"{view.dr_error_develop * 100:.2f}%" in text
    assert f"{view.dr_error_judged * 100:.2f}%" in text
    assert "judged seed 3" in text
    assert "worse" in text


def test_scoreboard_shows_the_greedy_decay_curve(data, screens) -> None:
    """The visual that is the argument for the constraints."""
    decay = data.scoreboard.decay
    assert Arm.GREEDY_UNCONSTRAINED in decay and Arm.ARC in decay
    greedy = decay[Arm.GREEDY_UNCONSTRAINED]
    assert len(greedy) == CYCLES
    assert greedy[0] > greedy[-1], (
        f"greedy recovery did not decay across cycles ({greedy}); the curve is the "
        f"argument and a flat one makes it silently"
    )

    text = _visible_text(screens["scoreboard.html"])
    assert "Recovery per cycle" in text
    assert "annoyance term" in text


# ---------------------------------------------------------------------------
# Gate 4 - the replay trace is prose
# ---------------------------------------------------------------------------
def test_replay_trace_is_prose_not_json(data) -> None:
    """A JSON dump is the raw material for a trace, not a trace."""
    trace = data.replay
    body = trace.text()

    assert len(trace.paragraphs) >= 5, "the trace is too short to explain a decision"
    with pytest.raises(json.JSONDecodeError):
        json.loads(body)

    assert "{" not in body and "}" not in body, "the trace contains a serialised object"
    assert '":' not in body, "the trace contains field-name syntax"
    # Real sentences, not labels with colons.
    sentences = [s for s in body.split(". ") if s.strip()]
    assert len(sentences) >= 10
    assert sum(len(p.split()) for p in trace.paragraphs) > 200


def test_replay_trace_covers_every_question_a_reviewer_asks(data) -> None:
    """Diagnosis, options, prices, rules with basis, propensity, outcome."""
    body = data.replay.text().lower()

    assert "sentinel attributed" in body and "confidence" in body
    assert "layer is" in body
    assert "were scored" in body or "actions survived" in body
    assert "shadow price" in body or "priced at" in body
    assert "the gate evaluated all" in body
    assert "probability" in body, "the propensity is not stated in words"
    assert "the outcome was" in body

    # The propensity sentence names an actual number.
    assert re.search(r"probability 0\.\d+", body), (
        "the trace mentions probability without giving one"
    )


def test_a_refusing_verdict_always_names_who_refused(data, registry) -> None:
    """A refusal that names nobody reads as a clean pass, which is worse than noise.

    THE BUG THIS WAS WRITTEN AFTER. `RuleRegistry` has no `__contains__`, so
    `rule_id in registry` fell back to iteration and compared a `str` against
    `Rule` objects - False for every id, including real ones. The replay
    screen filtered its firings through exactly that test, so every firing was
    discarded, and the renderer's `else` branch then printed "Nothing
    objected" underneath a verdict of block. Two lines later the same screen
    said the Gate had refused the sampled action. Both sentences were
    generated from one trace and they contradicted each other.

    WHAT THIS ASSERTS. For every rule in the registry that refuses, a trace
    carrying that refusal renders the rule id AND the force wording M3 allows
    for it - which is basis and status together, through `force_label`, the
    single function every renderer goes through. And no refusing verdict may
    ever render the words that started this.
    """
    from datetime import UTC, datetime

    from arc.console.replay import ConsideredAction, RuleFiring, Trace
    from arc.core.money import Paise
    from arc.core.types import ActionType, CauseLayer
    from arc.gate.lattice import Verdict

    refusing = [r for r in registry if r.on_violation is not Verdict.ALLOW]
    assert refusing, "the registry refuses nothing; this gate has nothing to watch"

    for rule in refusing:
        trace = Trace(
            claim_id="0" * 32,
            subject_token="sub_test",
            at=datetime(2026, 1, 1, 9, 0, tzinfo=UTC),
            amount_paise=Paise(120_000),
            ltv_paise=Paise(900_000),
            cause_label="insufficient_funds",
            cause_layer=CauseLayer.CUSTOMER,
            confidence=0.75,
            answered_by="code_map",
            cohort_power="sufficient",
            confidence_capped=False,
            considered=[
                ConsideredAction(
                    action=ActionType.SMS,
                    uplift=0.05,
                    adjusted_value=1_000.0,
                    propensity=0.4,
                )
            ],
            shadow_prices={"contact": 10.0, "rupee": 0.0},
            firings=[RuleFiring(rule_id=rule.id, verdict=rule.on_violation)],
            verdict=rule.on_violation,
            sampled_action=ActionType.SMS,
            sampled_propensity=0.4,
            realized_action=ActionType.DO_NOTHING,
            realized_propensity=0.6,
            veto_occurred=True,
            outcome="no response",
            recovered_paise=Paise(0),
            registry=registry,
        )
        view = narrate(trace)
        for surface in (view.text(), view.render()):
            assert rule.id in surface, (
                f"{rule.id} refused this decision and the trace does not name it"
            )
            assert rule.force_label() in surface, (
                f"{rule.id} is named without its force. Basis and status travel "
                f"together through force_label so nothing overstates its legal "
                f"weight, and a reader cannot tell law from our own policy without it"
            )
            assert "nothing objected" not in surface.lower(), (
                f"the trace returned {rule.on_violation.value} and also claimed nothing objected"
            )
        assert_no_overstated_force(view.render(), registry)

    # THE ORIGINAL SHAPE OF THE BUG: a refusing verdict whose firings were all
    # discarded on the way in. The renderer used to take its else branch here
    # and report that nothing objected, which is the one thing it must never
    # say under a refusal.
    stripped = replace(trace, firings=[], verdict=Verdict.BLOCK)
    view = narrate(stripped)
    for surface in (view.text(), view.render()):
        assert "nothing objected" not in surface.lower(), (
            "a block with no rule attached rendered as a clean pass; that is the "
            "contradiction this gate exists to catch"
        )


def test_the_built_replay_screen_does_not_contradict_itself(data) -> None:
    """The same check against the trace the console actually picks.

    The parametrised gate above proves the renderer can name a refuser. This
    one proves the screen a judge opens does, on real run data, where the
    refusal happens to come from the allocator's admission step rather than
    from a Gate rule - the case the original filter silently swallowed.
    """
    text = data.replay.text().lower()
    rendered = data.replay.render().lower()

    refused = data.replay.stages and any(
        value.split(" - ")[0] not in ("allow", "")
        for stage in data.replay.stages
        if stage.label == "Gate verdict"
        for key, value in stage.rows
        if key == "verdict"
    )
    if not refused:
        return

    assert "nothing objected" not in text, "the trace refuses and says nothing objected"
    assert "nothing objected" not in rendered, "the screen refuses and says nothing objected"

    named = [
        value
        for stage in data.replay.stages
        if stage.label == "Gate verdict"
        for key, value in stage.rows
        if key not in ("rules evaluated", "verdict")
    ]
    assert named, "the screen reports a refusal and names nobody who refused"


def test_replay_states_rule_force_through_the_registry(data, registry) -> None:
    """Rules named in the trace carry M3's wording, and nothing overstates."""
    assert_no_overstated_force(data.replay.render(), registry)
    assert_no_overstated_force(data.replay.text(), registry)


def test_replay_narration_is_deterministic(data) -> None:
    """The same trace narrated twice reads the same. A judge may re-open it."""
    from arc.console.build import _pick_trace

    trace = _pick_trace(data.result, load_registry(), data.result.at0)
    assert narrate(trace).text() == narrate(trace).text()
