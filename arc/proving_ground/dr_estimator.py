"""Doubly-robust off-policy estimation, and the error it reports about itself.

THE ESTIMATOR

    V_DR(pi) = (1/n) sum_i [ sum_a pi(a|s_i) qhat(s_i, a)
                             + (pi(a_i|s_i) / pi_b(a_i|s_i)) * (r_i - qhat(s_i, a_i)) ]

The first term is a direct model-based estimate. The second corrects it using
only the logged action, weighted by how much more likely the target policy was
to take it than the behaviour policy was.

WHY DOUBLY ROBUST. The estimate stays consistent if EITHER the outcome model
`qhat` or the propensity `pi_b` is correct. It does not need both. Standard
practice has to estimate the propensity and inherits the mis-specification as
bias, so the usual position is "hopefully one of two fitted models is right."

    OUR POSITION IS STRONGER, AND IT IS STRUCTURAL. `pi_b` here is not
    estimated. The Allocator drew with it and logged it, the epsilon floor
    guarantees it is bounded away from zero, and `composed_propensity` folds
    the Gate into it in closed form. One of the two legs is therefore correct
    BY CONSTRUCTION rather than by hope, which is why the crude outcome model
    below is a deliberate choice and not a shortcut - a better `qhat` reduces
    variance here, it does not buy consistency, because consistency is already
    paid for.

WHY THE OUTCOME MODEL IS SHRUNK CELL MEANS. It is fitted on the same logs it
corrects, so an expressive model would interpolate the noise it is meant to
average out, and its residuals would shrink toward zero exactly where the
importance weights are largest. Shrinking cell means toward the global mean
keeps it honest at thin cells and keeps the residual real.

VARIANCE CONTROL. Monetary outcomes are heavy-tailed - one large invoice
recovering dominates a thousand small ones - so an unclipped importance ratio
lets a single row move the estimate. Ratios are clipped, and the clipped share
is REPORTED rather than absorbed, because clipping trades bias for variance
and a reader is entitled to know how much was traded.

THE BOOTSTRAP RESAMPLES SUBJECTS, NOT ROWS. The subject is the unit of
randomisation (GI-8) and therefore the unit of independence. Resampling rows
would treat a subject's four cycles as four independent observations, and the
interval would come out roughly half as wide as the truth deserves.

VALIDATION AGAINST GROUND TRUTH. The simulator retains full counterfactuals,
so the quantity this estimator is trying to recover is available exactly. That
makes it possible to report the estimator's OWN error rather than only its
output, which is the difference between a measurement and a claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from arc.core.types import ActionType
from arc.proving_ground.composed import DO_NOTHING, DecisionKey

# Importance ratios above this are clipped. Twenty means a row may count for
# at most twenty ordinary rows; with an epsilon floor of 0.05 over roughly a
# dozen actions the unclipped ceiling is an order of magnitude higher, so this
# binds occasionally and is reported when it does.
DEFAULT_CLIP = 20.0

# Bootstrap resamples for the interval. Enough that the percentile endpoints
# are stable to well under a percent of the estimate.
DEFAULT_RESAMPLES = 400

# Shrinkage constant for the outcome model. A cell with `k` observations is
# weighted half toward its own mean and half toward its parent.
SHRINKAGE_K = 25.0


# A target policy is a function from a logged row to a distribution over that
# row's decision keys. It is a callable rather than an object because every
# policy this evaluates is already expressed as a distribution somewhere else,
# and wrapping them in classes would only move the code.
TargetPolicyFn = Callable[["LoggedDecision"], Mapping[DecisionKey, float]]


@dataclass(frozen=True)
class LoggedDecision:
    """One subject-cycle, as the estimator needs it.

    THE THREE-WAY LOG IS HERE. `intended_key` with `pi_intended` is what the
    Allocator sampled; `realized_key` with `pi_realized` is what survived the
    Gate; `veto_occurred` says whether they differ. An estimator that
    conditioned on `pi_intended` would be conditioning on an action that did
    not happen, so `pi_behaviour` below deliberately reads the realized one.
    """

    subject_token: str
    cycle: int
    stratum: str

    intended_key: DecisionKey
    pi_intended: float
    realized_key: DecisionKey
    pi_realized: float
    veto_occurred: bool
    blocking_rule_ids: tuple[str, ...]

    reward_paise: int
    cost_paise: int

    # pi_exec over every decision this subject could have received. The
    # support of the target policy must lie inside this, or the target is
    # asking about actions the logs cannot speak to.
    pi_exec: Mapping[DecisionKey, float] = field(default_factory=dict)

    # GROUND TRUTH, harness only. Expected reward in paise under each action,
    # read from the simulator before the world advanced. Never an input to any
    # estimate - only to the error the estimate reports about itself.
    truth: Mapping[DecisionKey, float] = field(default_factory=dict)

    @property
    def pi_behaviour(self) -> float:
        """The propensity of what ACTUALLY happened. Never `pi_intended`."""
        return self.pi_realized


@dataclass(frozen=True)
class Estimate:
    """A point estimate with its interval and the diagnostics that qualify it."""

    point: float
    lo: float
    hi: float
    n_rows: int
    n_subjects: int
    clipped_share: float
    effective_sample_size: float
    resamples: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def covers(self, value: float) -> bool:
        return self.lo <= value <= self.hi

    def relative_error(self, truth: float) -> float:
        """|estimate - truth| / |truth|. The number that validates the machinery."""
        if truth == 0.0:
            return abs(self.point)
        return abs(self.point - truth) / abs(truth)

    def to_dict(self) -> dict[str, object]:
        return {
            "point": self.point,
            "ci_95": [self.lo, self.hi],
            "n_rows": self.n_rows,
            "n_subjects": self.n_subjects,
            "clipped_share": self.clipped_share,
            "effective_sample_size": self.effective_sample_size,
        }


class OutcomeModel:
    """qhat(s, a): shrunk mean reward per (stratum, ACTION), in paise.

    IT KEYS ON THE ACTION, NOT ON THE DECISION KEY. A decision key carries the
    claim id, which is unique to an account, so keying cells on it would give
    every cell a single observation, make the model a lookup of each row's own
    reward, and drive its residual to zero exactly where the correction term
    needs one. The estimate would then be the direct model estimate wearing a
    doubly-robust label. Two claims of the same subject are different
    decisions and the same conditional expectation; the stratum is what
    carries the state.

    Deliberately weak beyond that. See the module docstring: the propensity
    leg is exact, so this one buys variance reduction rather than consistency,
    and an expressive model fitted on the same logs would eat its own
    residuals.
    """

    def __init__(
        self,
        cells: Mapping[tuple[str, ActionType], float],
        by_action: Mapping[ActionType, float],
        grand_mean: float,
    ):
        self._cells = dict(cells)
        self._by_action = dict(by_action)
        self._grand_mean = grand_mean

    @property
    def grand_mean(self) -> float:
        return self._grand_mean

    def predict(self, row: LoggedDecision, key: DecisionKey) -> float:
        """Cell, then the action marginal, then the grand mean.

        The fallback chain is ordered by how much it conditions on, so an
        unseen (stratum, action) pair still gets an action-specific answer
        rather than the average of everything.
        """
        action = key[1]
        cell = self._cells.get((row.stratum, action))
        if cell is not None:
            return cell
        return self._by_action.get(action, self._grand_mean)


def fit_outcome_model(
    logs: Sequence[LoggedDecision], *, shrinkage: float = SHRINKAGE_K
) -> OutcomeModel:
    """Cell means shrunk toward the action mean, and that toward the grand mean.

        qhat_cell = w * mean_cell + (1 - w) * mean_action,   w = n / (n + k)

    The same partial-pooling form the Sentinel uses at thin cohorts, for the
    same reason: a cell with three observations should not be trusted as
    though it had three hundred.
    """
    if not logs:
        raise ValueError("an outcome model needs logged rewards to fit to")

    grand = float(np.mean([row.reward_paise for row in logs]))

    action_rewards: dict[ActionType, list[float]] = {}
    for row in logs:
        action_rewards.setdefault(row.realized_key[1], []).append(float(row.reward_paise))
    by_action = {action: float(np.mean(values)) for action, values in action_rewards.items()}

    cell_rewards: dict[tuple[str, ActionType], list[float]] = {}
    for row in logs:
        cell_rewards.setdefault((row.stratum, row.realized_key[1]), []).append(
            float(row.reward_paise)
        )

    cells: dict[tuple[str, ActionType], float] = {}
    for (stratum, action), values in cell_rewards.items():
        count = len(values)
        weight = count / (count + shrinkage)
        parent = by_action.get(action, grand)
        cells[(stratum, action)] = weight * float(np.mean(values)) + (1.0 - weight) * parent

    return OutcomeModel(cells, by_action, grand)


def _row_value(
    row: LoggedDecision,
    q_hat: OutcomeModel,
    pi_target: Mapping[DecisionKey, float],
    clip: float,
) -> tuple[float, bool, float]:
    """One row's doubly-robust contribution, plus whether it was clipped."""
    direct = sum(probability * q_hat.predict(row, key) for key, probability in pi_target.items())

    behaviour = row.pi_behaviour
    if behaviour <= 0.0:
        raise ValueError(
            f"{row.subject_token} cycle {row.cycle} logged a realized propensity of "
            f"{behaviour!r}. An action the policy could not have taken cannot be "
            "corrected for, and the epsilon floor exists so this cannot happen"
        )

    raw = pi_target.get(row.realized_key, 0.0) / behaviour
    weight = min(raw, clip)
    correction = weight * (float(row.reward_paise) - q_hat.predict(row, row.realized_key))
    return direct + correction, raw > clip, weight


def dr_estimate(
    logs: Sequence[LoggedDecision],
    q_hat: OutcomeModel,
    pi_target: TargetPolicyFn,
    *,
    clip: float = DEFAULT_CLIP,
    resamples: int = DEFAULT_RESAMPLES,
    rng: np.random.Generator | None = None,
) -> Estimate:
    """V_DR per decision, with a subject-clustered bootstrap 95% interval.

    `pi_target` maps a logged row to the distribution the policy under
    evaluation would have used in that state. For an on-policy check it is
    `row.pi_exec` itself, and the estimate then reproduces the sample mean.

    THE POINT ESTIMATE IS A ROW MEAN AND THE BOOTSTRAP IS CLUSTERED. Averaging
    within a subject before averaging across subjects looks like the careful
    thing to do and is a bias here, because a subject's ROW COUNT DEPENDS ON
    ITS OUTCOME: paying closes the claim, so a subject who pays in cycle one
    contributes a single row carrying a full payment, while a subject who
    never pays contributes four rows averaging to nearly nothing. Weighting
    subjects equally therefore over-weights exactly the rows that paid, and
    the estimate runs high - by around sixty percent on this batch, which is
    not a rounding error and has no symptom.

    So the estimand is per DECISION and the point estimate is the pooled row
    mean, while the BOOTSTRAP resamples subjects as clusters. That keeps
    GI-8's unit of independence where it belongs - in the interval, which is
    what dependence between a subject's cycles actually affects - without
    letting outcome-dependent cluster sizes into the point estimate.
    """
    if not logs:
        raise ValueError("no logged decisions to estimate from")
    if rng is None:
        raise ValueError("an injected generator is required; there is no global rng in this repo")

    by_subject: dict[str, list[float]] = {}
    clipped = 0
    weights: list[float] = []

    for row in logs:
        target = pi_target(row)
        total = float(sum(target.values()))
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"target policy for {row.subject_token} sums to {total!r}, not one")
        value, was_clipped, weight = _row_value(row, q_hat, target, clip)
        by_subject.setdefault(row.subject_token, []).append(value)
        clipped += int(was_clipped)
        weights.append(weight)

    subjects = sorted(by_subject)
    sums = np.array([float(np.sum(by_subject[s])) for s in subjects])
    counts = np.array([len(by_subject[s]) for s in subjects], dtype=float)
    point = float(sums.sum() / counts.sum())

    # The cluster bootstrap: resample SUBJECTS, pool their rows, take the row
    # mean of the pooled draw. Summing each cluster's total and count makes
    # that exact without materialising the pooled rows.
    index = np.arange(len(subjects))
    draws = np.empty(resamples, dtype=float)
    for draw in range(resamples):
        picked = rng.choice(index, size=len(index), replace=True)
        drawn_count = counts[picked].sum()
        draws[draw] = float(sums[picked].sum() / drawn_count) if drawn_count else 0.0
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))

    weight_array = np.asarray(weights, dtype=float)
    ess = (
        float(weight_array.sum() ** 2 / np.square(weight_array).sum())
        if weight_array.size and float(np.square(weight_array).sum()) > 0.0
        else 0.0
    )

    return Estimate(
        point=point,
        lo=lo,
        hi=hi,
        n_rows=len(logs),
        n_subjects=len(subjects),
        clipped_share=clipped / len(logs),
        effective_sample_size=ess,
        resamples=resamples,
    )


def ips_estimate(
    logs: Sequence[LoggedDecision],
    pi_target: TargetPolicyFn,
    *,
    clip: float = DEFAULT_CLIP,
) -> float:
    """Plain importance sampling, no outcome model. The comparison point.

    Kept because reporting DR beside IPS shows what the outcome model bought:
    the same expectation, materially less variance. Row-averaged, for the same
    reason `dr_estimate` is.
    """
    values = []
    for row in logs:
        target = pi_target(row)
        weight = min(target.get(row.realized_key, 0.0) / row.pi_behaviour, clip)
        values.append(weight * float(row.reward_paise))
    return float(np.mean(values))


def ground_truth_value(logs: Sequence[LoggedDecision], pi_target: TargetPolicyFn) -> float:
    """The quantity V_DR is trying to recover, read from the simulator.

        V_true(pi) = (1/n) sum_i sum_a pi(a|s_i) * E[reward | s_i, a]

    EVALUATION HARNESS ONLY. `row.truth` was read from the world before the
    world advanced, so it is the expectation at exactly the state the row was
    logged in. Nothing in the estimate above touches it - it exists so the
    estimator can report its own error instead of only its output.

    Row-averaged, matching `dr_estimate` exactly. Comparing a row-averaged
    estimate against a subject-averaged truth would report the aggregation
    difference as estimator error.
    """
    expectations: list[float] = []
    for row in logs:
        if not row.truth:
            raise ValueError(
                f"{row.subject_token} cycle {row.cycle} carries no ground truth; "
                "the harness must record counterfactuals before the world advances"
            )
        target = pi_target(row)
        expectations.append(
            sum(probability * row.truth.get(key, 0.0) for key, probability in target.items())
        )
    return float(np.mean(expectations))


def on_policy_target(row: LoggedDecision) -> Mapping[DecisionKey, float]:
    """The behaviour policy itself, as a target. The self-consistency check."""
    return row.pi_exec


def always_do_nothing(row: LoggedDecision) -> Mapping[DecisionKey, float]:
    """The null policy, as a target. Its value is what arrives anyway."""
    return {DO_NOTHING: 1.0}


def tilted_target(
    exponent: float,
) -> TargetPolicyFn:
    """A policy that leans harder on whatever the behaviour policy preferred.

    Used to evaluate a target that genuinely DIFFERS from the behaviour policy
    while keeping its support, which is the case off-policy evaluation exists
    for. An estimator only tested on-policy has not been tested.
    """

    def target(row: LoggedDecision) -> Mapping[DecisionKey, float]:
        keys = list(row.pi_exec)
        raised = np.array([row.pi_exec[key] ** exponent for key in keys], dtype=float)
        total = float(raised.sum())
        if total <= 0.0 or not math.isfinite(total):
            return {key: 1.0 / len(keys) for key in keys}
        return {key: float(value / total) for key, value in zip(keys, raised, strict=True)}

    return target
