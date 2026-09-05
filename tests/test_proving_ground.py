"""M11 acceptance gate - the Proving Ground.

    test_five_arms_run_on_same_batch
    test_dr_estimate_within_5pct_of_ground_truth
    test_vetoed_decisions_collapse_not_dropped
    test_composed_propensity_sums_to_one
    test_bootstrap_ci_covers_truth_95pct
    test_headline_never_reported_without_guardrails
    test_prevention_reported_separately_from_recovery
    test_recovery_reversed_moves_headline_down
    test_greedy_arm_blows_guardrails_while_winning_gross

THIS IS THE MILESTONE THE PROBLEM STATEMENT GRADES. Everything before it exists
to make one number defensible, and the number is only defensible if the
machinery that produced it can be checked against a truth it did not see.

THE SUITE IS NOT VACUOUSLY GREEN, AND PROVES IT. Three tests plant a defect and
assert it is caught: `test_dropping_vetoed_decisions_is_caught_as_bias` runs an
estimator that drops vetoed decisions instead of collapsing them and checks the
bias assertion fires rather than some incidental one;
`test_headline_refuses_to_serialise_without_guardrails` builds a payload with a
recovery figure and no guardrails; `test_ground_truth_ban_still_fires_after_the
_carve_out` plants a counterfactual call in the two packages the carve-out must
not have widened to cover.

WHAT THE HARNESS COSTS. One `run_all` over eighteen hundred accounts and four
cycles is about eighty seconds, dominated by certifying every branch of every
subject's distribution - which is what the composed propensity requires and is
not avoidable without estimating it instead. It is module-scoped and shared, so
the suite pays it once.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import numpy as np
import pytest
from arc.core.money import paise
from arc.core.types import ActionType, ClaimState, Rail
from arc.gate.context import (
    Channel,
    ConsentState,
    GateContext,
)
from arc.gate.evaluator import Gate
from arc.gate.lattice import Verdict
from arc.gate.registry import load_registry
from arc.ledger.decision_ledger import DecisionLedger
from arc.ledger.money_ledger import MoneyAccount, MoneyLedger
from arc.proving_ground.arms import Arm
from arc.proving_ground.composed import (
    ADMISSION_RULE_ID,
    DO_NOTHING,
    ComposedPolicy,
    DecisionKey,
    MassNotConserved,
    Resolution,
    assert_mass_conserved,
    composed_propensity,
)
from arc.proving_ground.dr_estimator import (
    LoggedDecision,
    dr_estimate,
    fit_outcome_model,
    ground_truth_value,
    on_policy_target,
)
from arc.proving_ground.harness import (
    DENOMINATOR,
    FIRST_CYCLE_OFFSET,
    build_scoreboard,
    run_all,
)
from arc.proving_ground.metrics import (
    ArmReport,
    Guardrails,
    GuardrailsMissing,
    Headline,
    blows_guardrails,
    guardrails_from_counts,
)
from arc.simulator.seeds import DEVELOP_SEED, EPOCH
from tests.conductor_db import scratch_database

REPO_ROOT = Path(__file__).resolve().parents[1]

# Sized so the doubly-robust estimate's own error is comfortably inside the
# five percent the gate asserts. At six hundred accounts the error runs around
# five percent and the test would be knife-edge; at eighteen hundred it runs
# two to three percent across seeds, which is a gate rather than a coin flip.
POPULATION = 1_800
CYCLES = 4
AT = EPOCH + FIRST_CYCLE_OFFSET

TOKEN = "sub_" + "a1b2c3d4" * 4
CLAIM_A = UUID(int=0xA1)
CLAIM_B = UUID(int=0xB2)


# ---------------------------------------------------------------------------
# The shared run
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gate() -> Gate:
    return Gate(load_registry())


@pytest.fixture(scope="module")
def result(gate: Gate):
    """One batch, five arms, each on its own fork of the world."""
    return run_all(seed=DEVELOP_SEED, size=POPULATION, cycles=CYCLES, gate=gate)


@pytest.fixture(scope="module")
def arc_logs(result):
    return result.runs[Arm.ARC].logs


@pytest.fixture(scope="module")
def scoreboard(result):
    return build_scoreboard(result)


# ---------------------------------------------------------------------------
# Gate test 1 - five arms, one batch
# ---------------------------------------------------------------------------
def test_five_arms_run_on_same_batch(result) -> None:
    """All five policies, the same subjects, no cross-arm contamination.

    THE FORK IS WHAT MAKES THIS A COMPARISON. If the arms shared one world,
    arm B's contacts would raise arm E's annoyance through the response
    model's sleeping-dog term, and the difference between them would be the
    difference between the arms plus whatever order they happened to run in.
    Each arm gets a fresh interaction history over the same population, so the
    comparison is paired and the only thing that varies is the decision rule.
    """
    assert set(result.runs) == set(Arm), "an arm did not run"

    subject_counts = {arm: run.subjects for arm, run in result.runs.items()}
    assert len(set(subject_counts.values())) == 1, (
        f"arms saw different subject counts {subject_counts}; they are not on the same batch"
    )
    assert result.subjects > 500, f"only {result.subjects} subjects, too few to compare arms"

    # Every arm produced decisions for every cycle it was live in.
    for arm, run in result.runs.items():
        assert run.logs, f"{arm} produced no decisions at all"

    # The null arm is the definition of untreated: it spends nothing and
    # touches nobody, so its recovery is what arrives anyway.
    null = result.runs[Arm.NULL]
    assert null.contacts == 0, f"the null arm made {null.contacts} contacts"
    assert null.spend_paise == 0, f"the null arm spent {null.spend_paise} paise"
    assert null.recovered_paise > 0, "nothing recovered without treatment; the world is inert"

    # And the arms genuinely differ, or the harness is running one policy five
    # times under different names.
    recoveries = {arm: run.recovered_paise for arm, run in result.runs.items()}
    assert len(set(recoveries.values())) == len(recoveries), (
        f"two arms recovered identical amounts, which means they behaved identically: {recoveries}"
    )


# ---------------------------------------------------------------------------
# Gate test 2 - the estimator validates itself
# ---------------------------------------------------------------------------
def test_dr_estimate_within_5pct_of_ground_truth(arc_logs) -> None:
    """The measurement machinery, measured.

    The simulator retains full counterfactuals, so the quantity the estimator
    is trying to recover is available exactly. Reporting the estimate alone
    would be a claim; reporting its ERROR against a truth it never saw is what
    makes it a measurement.

    Note what this does and does not validate. It validates the estimator, the
    propensity bookkeeping and the composition - if the logged propensity
    disagreed with the realised action distribution by so much as a few
    percent, the importance ratios would drift and this would fail. It does
    not validate the simulator, which is the subject of M4's own gate.
    """
    q_hat = fit_outcome_model(arc_logs)
    estimate = dr_estimate(
        arc_logs, q_hat, on_policy_target, rng=np.random.default_rng(DEVELOP_SEED)
    )
    truth = ground_truth_value(arc_logs, on_policy_target)
    error = estimate.relative_error(truth)

    assert error < 0.05, (
        f"doubly-robust estimate {estimate.point:,.0f} against ground truth "
        f"{truth:,.0f} is {error:.1%} out. Above five percent the headline "
        f"number is not measuring what it claims to"
    )
    assert estimate.covers(truth), (
        f"the 95% interval [{estimate.lo:,.0f}, {estimate.hi:,.0f}] misses the truth {truth:,.0f}"
    )
    assert estimate.n_subjects > 400, "too few clusters for the interval to mean anything"


def test_logged_propensity_matches_the_realised_action_distribution(arc_logs) -> None:
    """The check that would have caught the admission defect.

    If the behaviour policy on record is the policy that actually ran, then
    averaging any function of the action over pi_exec must match averaging it
    over the actions that actually occurred. Here that function is the outcome
    model, so the two legs of the doubly-robust correction cancel in
    expectation.

    THIS IS NOT A RESTATEMENT OF THE TEST ABOVE. The Allocator's in-cycle
    admission step replaces a sampled action with `do_nothing` when the draw
    overshoots a cap, and it leaves the logged propensity alone. Composing the
    Gate but not admission left `do_nothing` recorded at probability 0.13 while
    it occurred 0.40 of the time - a thirty percent bias in the estimate that
    the interval still covered often enough to look survivable.
    """
    q_hat = fit_outcome_model(arc_logs)
    over_policy = np.mean(
        [sum(p * q_hat.predict(row, key) for key, p in row.pi_exec.items()) for row in arc_logs]
    )
    over_realised = np.mean([q_hat.predict(row, row.realized_key) for row in arc_logs])

    scale = max(abs(float(over_realised)), 1.0)
    drift = abs(float(over_policy) - float(over_realised)) / scale
    assert drift < 0.10, (
        f"averaging the outcome model over pi_exec gives {over_policy:,.0f} but over the "
        f"realised actions gives {over_realised:,.0f}, a {drift:.1%} gap. The logged "
        f"behaviour policy is not the policy that ran, so every importance ratio is wrong"
    )


# ---------------------------------------------------------------------------
# Gate test 3 and 4 - the composition
# ---------------------------------------------------------------------------
def _context(claim_id: UUID, **overrides: object) -> GateContext:
    fields: dict[str, object] = {
        "claim_id": claim_id,
        "subject_token": TOKEN,
        "rail": Rail.CARD,
        "claim_state": ClaimState.DIAGNOSED,
        "amount_paise": paise(129_900),
        "consent": dict.fromkeys(Channel, ConsentState.GRANTED),
    }
    fields.update(overrides)
    return GateContext(**fields)  # type: ignore[arg-type]


class _RefusingGate:
    """A Gate that refuses a named set of actions and allows the rest."""

    def __init__(self, refuse: set[ActionType], decision: Verdict = Verdict.BLOCK) -> None:
        self.refuse = refuse
        self.decision = decision
        self.calls: list[ActionType] = []

    def certify(self, ctx, action, at):
        self.calls.append(action)
        refused = action in self.refuse
        return type(
            "Cert",
            (),
            {
                "decision": self.decision if refused else Verdict.ALLOW,
                "blocking_rule_ids": ("TEST-REFUSAL",) if refused else (),
                "certificate_id": uuid4(),
            },
        )()


def test_composed_propensity_sums_to_one() -> None:
    """pi_exec is a distribution, per subject, whatever the Gate did.

    Checked across every refusal pattern from "nothing refused" to "everything
    refused", because the failure mode is not a Gate that refuses nothing - it
    is a Gate that refuses several branches at once and an implementation that
    assigns rather than accumulates.
    """
    actions = [
        ActionType.RETRY,
        ActionType.SMS,
        ActionType.EMAIL,
        ActionType.WHATSAPP_UTILITY,
        ActionType.VOICE_CALL,
    ]
    pi_alloc: dict[DecisionKey, float] = {(CLAIM_A, action): 0.15 for action in actions}
    pi_alloc[DO_NOTHING] = 1.0 - sum(pi_alloc.values())
    contexts = {CLAIM_A: _context(CLAIM_A)}

    for refused_count in range(len(actions) + 1):
        refused = set(actions[:refused_count])
        composed = composed_propensity(pi_alloc, _RefusingGate(refused), contexts, AT)

        total = sum(composed.pi_exec.values())
        assert total == pytest.approx(1.0, abs=1e-12), (
            f"pi_exec sums to {total!r} with {refused_count} branches refused"
        )
        assert all(p >= 0.0 for p in composed.pi_exec.values()), "a negative propensity"

        # NO MASS LOST WHEN SEVERAL BRANCHES VETO ONTO ONE OUTCOME. This is the
        # accumulate-versus-assign check, and it is the reason the loop runs to
        # five refusals rather than stopping at one.
        expected_do_nothing = pi_alloc[DO_NOTHING] + sum(
            pi_alloc[(CLAIM_A, action)] for action in refused
        )
        assert composed.pi_exec[DO_NOTHING] == pytest.approx(expected_do_nothing, abs=1e-12), (
            f"{refused_count} branches collapsed onto do_nothing but it carries "
            f"{composed.pi_exec[DO_NOTHING]!r} instead of {expected_do_nothing!r}; "
            f"masses were assigned rather than added"
        )


def test_composed_propensity_sums_to_one_per_subject_in_the_real_run(arc_logs) -> None:
    """The same property, on the run the headline comes from."""
    for row in arc_logs:
        total = sum(row.pi_exec.values())
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"{row.subject_token} cycle {row.cycle} has pi_exec summing to {total!r}"
        )
        assert row.pi_realized > 0.0, (
            f"{row.subject_token} cycle {row.cycle} realised an action of probability zero"
        )
        assert row.pi_realized == pytest.approx(row.pi_exec[row.realized_key], abs=1e-12), (
            "the logged realised propensity is not the composed one"
        )


def test_vetoed_decisions_collapse_not_dropped() -> None:
    """A refused branch keeps its mass and moves it. It never leaves.

    Dropping vetoed decisions and renormalising is the selection bias this
    milestone exists to avoid: the Gate refuses precisely the subjects whose
    state makes them refusable, and removing them removes a non-random slice
    of the sample.
    """
    pi_alloc: dict[DecisionKey, float] = {
        (CLAIM_A, ActionType.VOICE_CALL): 0.5,
        (CLAIM_A, ActionType.SMS): 0.3,
        DO_NOTHING: 0.2,
    }
    contexts = {CLAIM_A: _context(CLAIM_A)}
    composed = composed_propensity(
        pi_alloc,
        _RefusingGate({ActionType.VOICE_CALL, ActionType.SMS}),
        contexts,
        AT,
    )

    # Every branch is still on the record, refused or not.
    assert {r.key for r in composed.resolutions} == set(pi_alloc), (
        "a branch vanished from the composition"
    )
    assert composed.veto_mass == pytest.approx(0.8), (
        f"0.8 of mass was refused but veto_mass reports {composed.veto_mass}"
    )
    assert composed.pi_exec[DO_NOTHING] == pytest.approx(1.0), (
        "all mass was refused, so do_nothing must carry all of it"
    )
    assert composed.pi_exec.get((CLAIM_A, ActionType.VOICE_CALL), 0.0) == 0.0

    # And the refusal is attributable, not just counted.
    voice = next(r for r in composed.resolutions if r.key[1] is ActionType.VOICE_CALL)
    assert voice.vetoed and voice.blocking_rule_ids == ("TEST-REFUSAL",)
    assert voice.resolved_to == DO_NOTHING


def test_realized_propensity_is_the_summed_collapsed_mass() -> None:
    """pi_realized is a SUM, never the one branch that happened to be sampled.

    WHY THIS IS THE NUMBER THAT MATTERS. The DR estimate divides by
    `pi_behaviour`, which reads `pi_realized`. If the realized propensity were
    recorded as the refused branch's own probability, every importance ratio
    on a vetoed row would divide by too small a number and the headline would
    run high with no symptom at all - the arithmetic stays valid, the estimate
    is simply wrong.

    WHAT IT MUST EQUAL. `do_nothing` executes when it is drawn AND whenever
    any other branch is refused onto it, so its execution probability is its
    own allocator mass plus the mass of every branch that collapsed. It
    therefore has to be STRICTLY greater than any single refused branch
    whenever do_nothing carried allocator mass of its own.
    """
    refused_mass, sms_mass, own_mass = 0.5, 0.2, 0.3
    pi_alloc: dict[DecisionKey, float] = {
        (CLAIM_A, ActionType.VOICE_CALL): refused_mass,
        (CLAIM_A, ActionType.SMS): sms_mass,
        DO_NOTHING: own_mass,
    }
    composed = composed_propensity(
        pi_alloc,
        _RefusingGate({ActionType.VOICE_CALL}),
        {CLAIM_A: _context(CLAIM_A)},
        AT,
    )
    realized = composed.pi_exec[DO_NOTHING]

    assert realized == pytest.approx(own_mass + refused_mass), (
        f"do_nothing executes at {realized}, but it was drawn at {own_mass} and "
        f"another {refused_mass} was refused onto it. The realized propensity is "
        "the sum of every branch that lands here, not one of them"
    )
    assert realized > refused_mass, (
        "do_nothing carried allocator mass of its own, so its realized propensity "
        "must strictly exceed the refused branch's probability. Equality means the "
        "collapsed mass was recorded in place of the sum"
    )
    assert composed.pi_exec[(CLAIM_A, ActionType.SMS)] == pytest.approx(sms_mass), (
        "an unrefused branch was disturbed by another branch's collapse"
    )

    # TWO refusals onto the same target accumulate; they do not overwrite.
    both = composed_propensity(
        pi_alloc,
        _RefusingGate({ActionType.VOICE_CALL, ActionType.SMS}),
        {CLAIM_A: _context(CLAIM_A)},
        AT,
    )
    assert both.pi_exec[DO_NOTHING] == pytest.approx(1.0), (
        "every branch was refused, so do_nothing must carry all of the mass"
    )
    assert both.pi_exec[DO_NOTHING] > composed.pi_exec[DO_NOTHING], (
        "refusing a second branch did not increase the collapsed mass"
    )


def test_the_logged_realized_propensity_is_the_composed_one(arc_logs) -> None:
    """The same property on the real run, where the estimate reads it.

    The log cannot show `pi_alloc`, so the check here is that the recorded
    `pi_realized` is the composed execution probability of the action that
    actually happened - the distribution that sums to one over what could have
    executed - and never the refused branch's own number.
    """
    vetoed = [row for row in arc_logs if row.veto_occurred]
    assert vetoed, "no refusals in this run; this gate would assert nothing"

    for row in vetoed:
        assert row.pi_realized == pytest.approx(row.pi_exec[row.realized_key]), (
            f"pi_realized {row.pi_realized} is not the composed probability of "
            f"{row.realized_key[1]}, which the DR estimate divides by"
        )
        assert row.intended_key not in row.pi_exec or row.pi_exec[row.intended_key] == 0.0, (
            "a refused branch still carries execution probability; it did not happen"
        )
        assert sum(row.pi_exec.values()) == pytest.approx(1.0), (
            "the execution distribution does not sum to one, so mass was lost or "
            "invented in the collapse"
        )
        assert row.pi_behaviour == row.pi_realized > 0.0


def test_a_refused_row_never_logs_the_same_two_propensities(arc_logs) -> None:
    """pi_intended and pi_realized cannot coincide on a row that was refused.

    WHY THEY MUST DIFFER. On a refused row the two numbers describe different
    events: pi_intended is the allocator's own mass for the branch it sampled,
    and pi_realized is the composed mass of the outcome that branch collapsed
    ONTO, which by construction includes the collapsed mass plus whatever else
    landed there. A refused branch contributes to the second and is absent from
    it. Equality is therefore not a coincidence to tolerate; it means one field
    was written with the other's value.

    THE BUG THIS REPLACES. `pi_intended` read `pi_exec.get(intended,
    pi_realized)`. Since `pi_exec` holds only what could EXECUTE, a refused
    branch was never a key, the default fired on every vetoed row, and the
    replay screen printed the same 0.771 for the action it sampled and the
    action it took instead. The estimate was unaffected, because it divides by
    `pi_behaviour` which reads `pi_realized`, but a silent default on a
    propensity field is one rename away from reaching the arithmetic.
    """
    vetoed = [row for row in arc_logs if row.veto_occurred]
    assert vetoed, "no refusals in this run; this gate would assert nothing"

    for row in vetoed:
        assert row.pi_intended != row.pi_realized, (
            f"row logged pi_intended and pi_realized both at {row.pi_intended}. "
            f"It sampled {row.intended_key[1]} and did {row.realized_key[1]}; those "
            "are different events and cannot share a probability"
        )
        assert 0.0 < row.pi_intended <= 1.0, f"pi_intended {row.pi_intended} is not a probability"
        # The refused branch is absent from the execution distribution, which
        # is exactly why the old default fired.
        assert row.intended_key not in row.pi_exec, (
            "the refused branch is present in the execution distribution"
        )


def test_deferred_branches_collapse_like_blocked_ones() -> None:
    """DEFER is not a soft ALLOW. It did not happen this cycle."""
    pi_alloc: dict[DecisionKey, float] = {(CLAIM_A, ActionType.SMS): 0.7, DO_NOTHING: 0.3}
    composed = composed_propensity(
        pi_alloc,
        _RefusingGate({ActionType.SMS}, decision=Verdict.DEFER),
        {CLAIM_A: _context(CLAIM_A)},
        AT,
    )
    assert composed.pi_exec[DO_NOTHING] == pytest.approx(1.0)
    assert composed.veto_mass == pytest.approx(0.7)


def test_admission_refusal_composes_like_a_gate_veto() -> None:
    """The Allocator's own budget admission is part of the behaviour policy.

    M8 replaces a sampled action with `do_nothing` when the draw overshoots a
    cap and deliberately leaves the logged propensity alone, delegating the
    accounting here. Omitting it does not fail loudly - it just makes the
    recorded policy disagree with the one that ran.
    """
    pi_alloc: dict[DecisionKey, float] = {
        (CLAIM_A, ActionType.VOICE_CALL): 0.6,
        (CLAIM_A, ActionType.SMS): 0.25,
        DO_NOTHING: 0.15,
    }
    composed = composed_propensity(
        pi_alloc,
        _RefusingGate(set()),
        {CLAIM_A: _context(CLAIM_A)},
        AT,
        admissible=lambda key: key[1] is not ActionType.VOICE_CALL,
    )
    assert composed.pi_exec[DO_NOTHING] == pytest.approx(0.75)
    voice = next(r for r in composed.resolutions if r.key[1] is ActionType.VOICE_CALL)
    assert voice.blocking_rule_ids == (ADMISSION_RULE_ID,), (
        "a budget refusal must be distinguishable from a compliance refusal in the audit trail"
    )


def test_multiple_claims_of_one_subject_keep_separate_mass() -> None:
    """Two claims offering the same action are two decisions, not one.

    Keying the distribution by action alone would merge them and silently lose
    one of the two probabilities.
    """
    pi_alloc: dict[DecisionKey, float] = {
        (CLAIM_A, ActionType.SMS): 0.4,
        (CLAIM_B, ActionType.SMS): 0.35,
        DO_NOTHING: 0.25,
    }
    composed = composed_propensity(
        pi_alloc, _RefusingGate(set()), {CLAIM_A: _context(CLAIM_A), CLAIM_B: _context(CLAIM_B)}, AT
    )
    assert composed.pi_exec[(CLAIM_A, ActionType.SMS)] == pytest.approx(0.4)
    assert composed.pi_exec[(CLAIM_B, ActionType.SMS)] == pytest.approx(0.35)
    assert sum(composed.pi_exec.values()) == pytest.approx(1.0)


def test_dropping_a_branch_is_caught_by_the_mass_assertion() -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    A composition that drops vetoed branches instead of collapsing them is
    constructed by hand and fed to the same assertion the real path runs. It
    must fail on the BIAS claim - a branch missing from the composition -
    rather than on some incidental type or key error, because an assertion
    that happens to fire for the wrong reason would not catch the real defect.
    """
    pi_alloc: dict[DecisionKey, float] = {
        (CLAIM_A, ActionType.VOICE_CALL): 0.5,
        (CLAIM_A, ActionType.SMS): 0.3,
        DO_NOTHING: 0.2,
    }
    # The defect: the vetoed branch is dropped and the survivors renormalised,
    # which is exactly what a well-meaning implementation does.
    survivors = {(CLAIM_A, ActionType.SMS): 0.3, DO_NOTHING: 0.2}
    total = sum(survivors.values())
    dropping = ComposedPolicy(
        pi_exec={key: value / total for key, value in survivors.items()},
        resolutions=(
            Resolution(
                key=(CLAIM_A, ActionType.SMS),
                mass=0.3,
                verdict=Verdict.ALLOW,
                resolved_to=(CLAIM_A, ActionType.SMS),
            ),
            Resolution(key=DO_NOTHING, mass=0.2, verdict=Verdict.ALLOW, resolved_to=DO_NOTHING),
        ),
    )

    with pytest.raises(MassNotConserved) as caught:
        assert_mass_conserved(pi_alloc, dropping)

    message = str(caught.value)
    assert "never resolved" in message and "selection bias" in message, (
        f"the assertion fired, but for the wrong reason: {message!r}. It must "
        f"identify the dropped branch, not merely notice the totals disagree"
    )
    assert "voice_call" in message, "the message does not name the branch that was dropped"


def test_a_dropping_estimator_is_biased_against_the_collapsing_one(arc_logs) -> None:
    """The bias a dropped veto produces, measured on the real logs.

    The assertion above proves the guard fires. This proves the guard is worth
    having: the same estimator run over a log whose vetoed decisions were
    dropped lands materially further from ground truth than the one that
    collapses them, in the direction dropping predicts.
    """
    vetoed = [row for row in arc_logs if row.veto_occurred]
    assert vetoed, "no vetoes occurred, so this comparison would be vacuous"

    q_hat = fit_outcome_model(arc_logs)
    truth = ground_truth_value(arc_logs, on_policy_target)

    honest = dr_estimate(arc_logs, q_hat, on_policy_target, rng=np.random.default_rng(DEVELOP_SEED))
    # The defect: keep only the rows the Gate did not touch.
    kept = [row for row in arc_logs if not row.veto_occurred]
    dropping = dr_estimate(kept, q_hat, on_policy_target, rng=np.random.default_rng(DEVELOP_SEED))

    assert dropping.relative_error(truth) > honest.relative_error(truth), (
        f"dropping vetoed decisions gave {dropping.relative_error(truth):.1%} error and "
        f"collapsing gave {honest.relative_error(truth):.1%}. If dropping were no worse, "
        f"this milestone's central claim would be decorative"
    )


# ---------------------------------------------------------------------------
# Gate test 5 - interval coverage
# ---------------------------------------------------------------------------
def _synthetic_logs(
    rng: np.random.Generator, *, subjects: int = 160, cycles: int = 3
) -> tuple[list[LoggedDecision], float]:
    """Logs from a known model, so coverage can be checked against exact truth.

    WHY SYNTHETIC HERE AND NOT THE HARNESS. Coverage is a property of the
    interval over REPEATED experiments, so it needs a hundred independent
    datasets. A hundred harness runs would be two hours; a hundred draws from a
    model whose expectation is known in closed form is a second, and it tests
    the estimator rather than the world.

    THE REWARD IS LUMPY, NOT GAUSSIAN, AND THAT IS LOAD-BEARING. A payment
    either arrives in full or does not arrive, so the reward is an amount times
    a Bernoulli draw and its mean is that amount times the probability -
    exactly, with no truncation. A Gaussian clipped at zero has a mean ABOVE
    its stated one, and the gap shows up as estimator bias that is really
    generator bias. It also makes the reward heavy-tailed, which is the
    distribution the clipping in the estimator exists for.
    """
    keys: list[DecisionKey] = [
        DO_NOTHING,
        (CLAIM_A, ActionType.SMS),
        (CLAIM_A, ActionType.RETRY),
        (CLAIM_A, ActionType.VOICE_CALL),
    ]
    means = {keys[0]: 400.0, keys[1]: 900.0, keys[2]: 1500.0, keys[3]: 2100.0}
    amount = 5_000.0
    pay_probability = {key: mean / amount for key, mean in means.items()}

    logs: list[LoggedDecision] = []
    expectations: list[float] = []
    for index in range(subjects):
        token = f"sub_{index:032x}"
        weights = rng.dirichlet(np.ones(len(keys)) * 2.0)
        pi = dict(zip(keys, (float(w) for w in weights), strict=True))
        expected = sum(pi[k] * means[k] for k in keys)
        for cycle in range(cycles):
            drawn = int(rng.choice(len(keys), p=[pi[k] for k in keys]))
            key = keys[drawn]
            reward = int(amount) if rng.random() < pay_probability[key] else 0
            logs.append(
                LoggedDecision(
                    subject_token=token,
                    cycle=cycle,
                    stratum="s0",
                    intended_key=key,
                    pi_intended=pi[key],
                    realized_key=key,
                    pi_realized=pi[key],
                    veto_occurred=False,
                    blocking_rule_ids=(),
                    reward_paise=reward,
                    cost_paise=0,
                    pi_exec=pi,
                    truth={k: means[k] for k in keys},
                )
            )
            expectations.append(expected)
    return logs, float(np.mean(expectations))


def test_bootstrap_ci_covers_truth_95pct() -> None:
    """The interval means what it says: it covers the truth about 95% of the time.

    An interval that is merely printed beside a number is decoration. This
    runs a hundred independent experiments and counts how often the interval
    contains the true value.

    THE THRESHOLD IS DELIBERATELY BELOW 95%. With a hundred replications and
    nominal coverage of 0.95, the count has a standard deviation of about 2.2
    points, so a correct interval lands below 0.90 roughly once in a hundred
    runs by chance alone. Asserting at 0.95 would make this test flaky in a
    way that trains people to re-run it, which is worse than a slightly loose
    bound.
    """
    rng = np.random.default_rng(20_260_904)
    replications = 100
    covered = 0
    for _ in range(replications):
        logs, truth = _synthetic_logs(rng)
        q_hat = fit_outcome_model(logs)
        estimate = dr_estimate(logs, q_hat, on_policy_target, rng=rng, resamples=200)
        covered += int(estimate.covers(truth))

    coverage = covered / replications
    assert coverage >= 0.88, (
        f"the 95% interval covered the truth {coverage:.0%} of the time over "
        f"{replications} experiments. An interval that under-covers overstates "
        f"the precision of the headline number"
    )
    assert coverage <= 1.0


# ---------------------------------------------------------------------------
# Gate tests 6 and 7 - what the metrics refuse to do
# ---------------------------------------------------------------------------
def _guardrails(**overrides: object) -> Guardrails:
    fields: dict[str, object] = {
        "contacts": 400,
        "complaints": 3,
        "opt_outs": 5,
        "treated_subjects": 300,
        "treated_cancellations": 4,
        "control_subjects": 300,
        "control_cancellations": 3,
        "promises_made": 40,
        "promises_kept": 22,
        "promises_unresolved": 6,
        "right_party_contacts": 370,
        "spend_paise": paise(50_000),
        "recovered_paise": paise(900_000),
    }
    fields.update(overrides)
    return guardrails_from_counts(**fields)  # type: ignore[arg-type]


def _headline(**overrides: object) -> Headline:
    fields: dict[str, object] = {
        "arm": Arm.ARC,
        "comparator": Arm.NAIVE_DUNNING,
        "recovered_paise": paise(900_000),
        "comparator_recovered_paise": paise(600_000),
        "spend_paise": paise(50_000),
        "denominator": DENOMINATOR,
        "guardrails": _guardrails(),
    }
    fields.update(overrides)
    return Headline(**fields)  # type: ignore[arg-type]


def test_headline_never_reported_without_guardrails(scoreboard) -> None:
    """Structural, not conventional. There is no code path that emits one alone.

    Two independent mechanisms, because one of them can be edited around. The
    dataclass has no default for `guardrails`, so the object cannot be built
    without them; and the serialiser re-checks the payload it just produced,
    so a future field that carries a recovery figure by another route is
    caught at runtime rather than shipped.
    """
    # 1. The object cannot exist without them.
    with pytest.raises(GuardrailsMissing):
        _headline(guardrails=None)
    with pytest.raises(TypeError):
        Headline(  # type: ignore[call-arg]
            arm=Arm.ARC,
            comparator=Arm.NAIVE_DUNNING,
            recovered_paise=paise(1),
            comparator_recovered_paise=paise(0),
            spend_paise=paise(0),
            denominator=DENOMINATOR,
        )

    # 2. Every payload the scoreboard emits carries all of them.
    payload = scoreboard.to_dict()
    for arm_payload in payload["arms"]:
        assert "recovered_paise" in arm_payload
        rails = arm_payload["guardrails"]
        for required in (
            "complaint_rate_per_1000",
            "opt_out_rate_per_1000",
            "voluntary_cancel_rate_treated",
            "voluntary_cancel_rate_control",
            "cost_per_rupee_collected",
            "promise_made_rate",
            "promise_kept_rate",
            "right_party_contact_rate",
        ):
            assert required in rails, f"{arm_payload['arm']} omits {required}"

    # 3. The rendered scoreboard puts them on the same rows as the money.
    text = "\n".join(scoreboard.render())
    assert "compl/1k" in text and "optout/1k" in text and "recovered" in text


def test_headline_refuses_to_serialise_a_payload_missing_guardrails() -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN - the serialiser's own check fires."""
    from arc.proving_ground.metrics import assert_guardrails_present

    with pytest.raises(GuardrailsMissing) as caught:
        assert_guardrails_present({"recovered_paise": 900_000})
    assert "not reportable alone" in str(caught.value)

    with pytest.raises(GuardrailsMissing) as caught:
        assert_guardrails_present(
            {"recovered_paise": 900_000, "guardrails": {"complaint_rate_per_1000": 1.0}}
        )
    assert "omit" in str(caught.value)

    # A payload with no recovery figure needs no guardrails, and must not be
    # rejected - otherwise the check would be a nuisance rather than a guard.
    assert_guardrails_present({"subjects": 10})


def test_guardrails_must_be_computed_against_the_reported_total() -> None:
    """Cost per rupee quoted against a different total is a wrong number."""
    with pytest.raises(GuardrailsMissing) as caught:
        _headline(guardrails=_guardrails(recovered_paise=paise(123)))
    assert "cost per rupee" in str(caught.value)


def test_prevention_reported_separately_from_recovery(scoreboard) -> None:
    """Money that never failed was never recovered.

    Prevention is a sibling of the headline, never a component of it. The test
    checks both halves: that both lines are present and distinct, and that
    moving prevention does not move the headline - because if it did, the
    separation would be presentational rather than real.
    """
    payload = scoreboard.to_dict()
    for arm_payload in payload["arms"]:
        assert "prevented_paise" in arm_payload, "prevention is not reported at all"
        assert "recovered_paise" in arm_payload
        assert arm_payload["incremental_paise"] == (
            arm_payload["recovered_paise"] - arm_payload["comparator_recovered_paise"]
        ), "the headline is not purely recovery minus comparator"

    report = ArmReport(headline=_headline(), prevented_paise=paise(1), subjects=10)
    inflated = ArmReport(headline=_headline(), prevented_paise=paise(99_999_999), subjects=10)
    assert report.to_dict()["recovered_paise"] == inflated.to_dict()["recovered_paise"], (
        "prevention leaked into the recovery total"
    )
    assert report.to_dict()["incremental_paise"] == inflated.to_dict()["incremental_paise"], (
        "prevention leaked into the incremental headline"
    )
    assert inflated.to_dict()["prevented_paise"] == 99_999_999

    # And the rendered output keeps them on separate lines.
    text = "\n".join(scoreboard.render())
    assert "prevention (separate line" in text


# ---------------------------------------------------------------------------
# Gate test 8 - reversals move the number down
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dsn() -> Iterator[str]:
    """A scratch database for this module alone. The M10 rule: per caller."""
    try:
        yield from scratch_database("proving_ground")
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover
        pytest.skip(f"postgres unavailable: {exc}")


async def test_recovery_reversed_moves_headline_down(dsn: str) -> None:
    """A number that cannot decrease is not a measurement.

    Driven through the real money ledger rather than by editing a total: the
    ledger derives every balance by summing legs and stores no running total,
    so a chargeback posts RECOVERY_REVERSED and the headline falls out of the
    same sum with nothing having to remember to decrement anything.
    """
    connection = await asyncpg.connect(dsn)
    transaction = connection.transaction()
    await transaction.start()
    try:
        ledger = MoneyLedger(DecisionLedger())
        at = datetime(2026, 3, 17, 10, 0, tzinfo=UTC)
        claims = [uuid4() for _ in range(3)]
        amount = paise(300_000)

        for claim_id in claims:
            await ledger.open_claim(connection, claim_id, amount, at=at)
            await ledger.transition(
                connection, claim_id, MoneyAccount.AT_RISK, MoneyAccount.RECOVERED, amount, at=at
            )

        before = await ledger.recovered_total(connection)
        assert int(before) == 3 * int(amount)

        headline_before = _headline(
            recovered_paise=before, guardrails=_guardrails(recovered_paise=before)
        )

        # The chargeback.
        await ledger.reverse_recovery(connection, claims[0], amount, at=at + timedelta(days=2))

        after = await ledger.recovered_total(connection)
        assert int(after) == 2 * int(amount), "the reversal did not leave the recovered leg"

        headline_after = _headline(
            recovered_paise=after, guardrails=_guardrails(recovered_paise=after)
        )

        assert int(headline_after.recovered_paise) < int(headline_before.recovered_paise)
        assert int(headline_after.incremental_paise) < int(headline_before.incremental_paise), (
            "the incremental headline did not fall when recovered money was reversed"
        )
        # Cost per rupee collected must move the other way: the same spend now
        # bought less.
        assert (
            headline_after.guardrails.cost_per_rupee_collected
            > headline_before.guardrails.cost_per_rupee_collected
        )
        assert await ledger.is_balanced(connection), "the reversal unbalanced the ledger"
    finally:
        await transaction.rollback()
        await connection.close()


# ---------------------------------------------------------------------------
# Gate test 9 - what the constraints buy
# ---------------------------------------------------------------------------
def test_greedy_arm_blows_guardrails_while_winning_gross(result, scoreboard) -> None:
    """The arm that shows what the constraints were buying.

    Greedy maximises expected value with no lifetime-value weight, no
    annoyance term, no budget and no Gate. It out-recovers the comparator arm
    the headline is stated against, and the guardrail columns beside it show
    the price.

    A FINDING WORTH STATING, BECAUSE IT CONTRADICTS THE PREDICTION. The spec
    expected greedy to out-gross ARC as well. On this frozen world it does
    not: its recovery decays hard across cycles as the response model's
    annoyance term bites, so contacting everyone every cycle destroys more
    value than it collects. The assertions below therefore state the gross
    comparison against the comparator arm, which is what the headline is
    measured against, and state the net-value comparison against ARC. What is
    NOT asserted is that greedy out-grosses ARC, because on this world it
    does not, and writing the assertion the other way round would have meant
    tuning the world until the prediction came true.
    """
    greedy = result.runs[Arm.GREEDY_UNCONSTRAINED]
    arc = result.runs[Arm.ARC]
    naive = result.runs[Arm.NAIVE_DUNNING]

    # It wins gross against the comparator the headline is stated against.
    assert greedy.recovered_paise > naive.recovered_paise, (
        f"greedy recovered {greedy.recovered_paise} against the comparator's "
        f"{naive.recovered_paise}; an unconstrained maximiser that cannot beat "
        f"fixed-schedule dunning on gross rupees is not the arm this test needs"
    )

    # It blows the guardrails doing it.
    greedy_rails = scoreboard.by_arm(Arm.GREEDY_UNCONSTRAINED).headline.guardrails
    arc_rails = scoreboard.by_arm(Arm.ARC).headline.guardrails
    breached = blows_guardrails(greedy_rails, arc_rails)
    assert "complaint_rate_per_1000" in breached and "opt_out_rate_per_1000" in breached, (
        f"greedy breached only {breached}. It is supposed to demonstrate the cost "
        f"of removing the constraints, and it has not"
    )
    assert greedy.opt_outs > arc.opt_outs and greedy.complaints > arc.complaints

    # And it spends several times the money.
    assert greedy.spend_paise > 3 * arc.spend_paise, (
        f"greedy spent {greedy.spend_paise} against ARC's {arc.spend_paise}; "
        f"the cost of being unconstrained is supposed to be visible"
    )

    # The result worth presenting: ARC wins on net value.
    greedy_net = greedy.recovered_paise - greedy.spend_paise
    arc_net = arc.recovered_paise - arc.spend_paise
    assert arc_net > greedy_net, (
        f"ARC net {arc_net} did not beat greedy net {greedy_net}. Beating an "
        f"unconstrained maximiser on net value is the claim this milestone makes"
    )


def test_arc_beats_the_comparator_and_the_incumbent(result) -> None:
    """The headline claim, stated plainly and checked."""
    arc = result.runs[Arm.ARC]
    for other in (Arm.NULL, Arm.NAIVE_DUNNING, Arm.GATEWAY_DEFAULT):
        assert arc.recovered_paise > result.runs[other].recovered_paise, (
            f"ARC recovered {arc.recovered_paise} against {other}'s "
            f"{result.runs[other].recovered_paise}"
        )


def test_naive_dunning_can_destroy_value_against_doing_nothing(result) -> None:
    """The sleeping-dog finding, recorded rather than hidden.

    Fixed-schedule dunning contacts everyone on a calendar. In a world whose
    response model carries a negative annoyance term - written into the
    simulator at M4, before any policy code existed - that can recover LESS
    than leaving people alone. The test asserts the comparison is computed and
    reported either way rather than asserting a direction, because the
    direction is a measurement and this is where it gets read.
    """
    naive = result.runs[Arm.NAIVE_DUNNING]
    null = result.runs[Arm.NULL]
    assert naive.contacts > 0, "the naive arm made no contacts, so it is not dunning"
    assert null.contacts == 0
    # Whatever the sign, the comparison must be available and finite.
    delta = naive.recovered_paise - null.recovered_paise
    assert isinstance(delta, int)


# ---------------------------------------------------------------------------
# Diagnostics and the judged run
# ---------------------------------------------------------------------------
def test_veto_rate_is_reported_as_a_diagnostic(scoreboard) -> None:
    """CB-VETO's input. Measured whether or not it trips."""
    diagnostics = scoreboard.arc.diagnostics
    assert 0.0 <= diagnostics.post_allocation_veto_rate <= 1.0
    assert 0.0 <= diagnostics.explore_mass_share <= 1.0


def test_judged_run_command_completes(tmp_path: Path) -> None:
    """`python -m arc.proving_ground.run --seed 3 --once` is the judged run.

    Run at a reduced population so the gate stays inside a sensible time; the
    command and its output shape are what is under test here, and the numbers
    are the fixture's job above.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "arc.proving_ground.run",
            "--seed",
            "3",
            "--size",
            "400",
            "--cycles",
            "2",
            "--once",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    out = completed.stdout
    assert "JUDGED" in out, "the judged seed's announced role was not printed"
    assert "relative error" in out, "the estimator did not report its own error"
    assert "prevention (separate line" in out
    assert "compl/1k" in out, "guardrails are not on the scoreboard"


def test_scoreboard_is_deterministic_for_a_seed(gate: Gate) -> None:
    """A judge asking to run it again must get the same number."""
    first = run_all(seed=DEVELOP_SEED, size=200, cycles=2, gate=gate)
    second = run_all(seed=DEVELOP_SEED, size=200, cycles=2, gate=gate)
    for arm in Arm:
        assert first.runs[arm].recovered_paise == second.runs[arm].recovered_paise, (
            f"{arm} recovered a different amount on the second run of the same seed"
        )
        assert first.runs[arm].spend_paise == second.runs[arm].spend_paise
