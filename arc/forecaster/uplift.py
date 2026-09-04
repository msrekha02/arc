"""Model B - uplift, as an X-learner.

THE HARD ONE, AND THE ONE THAT MATTERS. It estimates the conditional average
treatment effect

    tau(x, a) = E[Y | X = x, A = a] - E[Y | X = x, A = do_nothing]

for every action in the closed set. The label is never observed per unit: an
account either was contacted or was not, and the difference between the two
worlds is exactly what is missing from the data.

WHY AN X-LEARNER AND NOT THE OBVIOUS ALTERNATIVES:

  S-learner - one model with the treatment as a feature. It systematically
  under-detects. Nothing forces the tree to split on the treatment column, and
  when ability-to-pay and issuer health dominate the loss, it simply will not,
  so the estimated effect collapses toward zero and the sleeping dogs vanish.

  T-learner - one model per arm. Unbiased, but its variance is set by the
  smaller arm, and the arms here are wildly imbalanced by design: do_nothing
  and retry dominate the log while voice is rare, because voice is expensive
  and the Allocator prices it accordingly. A T-learner on the voice arm is
  fitting a handful of units.

  X-learner - built for that imbalance. It imputes the effect on both sides
  using the model fitted on the OTHER side, so the rare arm borrows the
  abundant arm's outcome surface instead of standing alone, then blends the
  two estimates by propensity so the better-supported side carries more weight
  where it is better supported.

THE PROPENSITY IS KNOWN, NOT ESTIMATED. Step 4 needs g(x). Most industrial
uplift work fits a model for it and inherits the mis-specification as bias.
The Allocator logged the exact sampling probability, so the weight is a
recorded fact. `dataset.require_propensities` refuses to run without it, and
there is no estimation fallback - see the note in that module.

SIGNED OUTPUT IS THE PRODUCT, NOT A SIDE EFFECT. tau < 0 means contacting this
account reduces recovery. The sleeping-dog rule falls out of the model rather
than being written by hand, which is the difference between a policy that has
found a segment and a policy that has been told one exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from arc.core.types import ActionType
from arc.forecaster.dataset import (
    InsufficientTreatedUnits,
    LoggedDecision,
    feature_matrix,
    require_propensities,
    reward_vector,
    rows_for,
)
from arc.forecaster.estimates import (
    COLD_START_EXPLORATION,
    COLD_START_MIN_OBSERVATIONS,
    DEGRADED_EXPLORATION,
    MIN_DEGRADED_HALF_WIDTH,
    EstimateBasis,
    UpliftEstimate,
)
from arc.forecaster.features import FeatureContext, ObservableLike, extract

# Base learners are regressors even for the binary outcome: the imputed
# pseudo-effects in step 2 are signed and continuous, so a classifier has
# nothing to classify.
#
# DELIBERATELY LOW CAPACITY, and this is the single most consequential setting
# in the module. The pseudo-effect D1 = Y - mu0(x) is a Bernoulli outcome minus
# a smooth prediction: at a payment rate in the low tens of percent it is
# almost entirely noise, and the systematic part - the account's responsiveness
# and its annoyance - is a small fraction of the variance. A learner with
# enough capacity to fit that noise will, and the estimated effect surface then
# ranks accounts by which way the noise fell.
#
# Measured against the frozen simulator, over three seeds: relaxing these to
# ordinary outcome-model settings (24 leaves, min_child_samples 30, lambda 2)
# drops correlation with ground-truth tau from 0.41 to 0.28 and sleeping-dog
# enrichment in the bottom decile from 1.86x to 1.36x. Constraining the model
# is what finds the segment.
#
# The same settings are used for the step-1 outcome models rather than giving
# those more room, which was also measured: an over-fitted mu0 injects its own
# error into every pseudo-effect computed from it, so smoothing it helps the
# imputation twice over.
BASE_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 120,
    "learning_rate": 0.04,
    "num_leaves": 8,
    "min_child_samples": 150,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 25.0,
    "verbose": -1,
}

# Below this many units in either arm the pseudo-effect regression is fitting
# noise. The action is still scored, from the arm-level mean, and marked cold
# rather than silently returned as a confident zero.
MIN_ARM_UNITS = 60

CONTROL_ACTION = ActionType.DO_NOTHING


@dataclass(frozen=True)
class ActionUplift:
    """One action's fitted X-learner, and how much support it had."""

    action: ActionType
    treated_units: int
    control_units: int
    mean_effect: float
    cold: bool


class XLearner:
    """One X-learner per treated action, sharing a single control arm.

    The control arm is `do_nothing`, which is always in the candidate set at
    M8 precisely so this model has one.
    """

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self._params = dict(BASE_PARAMS if params is None else params)
        self._mu0: lgb.LGBMRegressor | None = None
        self._tau0: dict[ActionType, lgb.LGBMRegressor] = {}
        self._tau1: dict[ActionType, lgb.LGBMRegressor] = {}
        self._support: dict[ActionType, ActionUplift] = {}
        self._control_units = 0

    @property
    def actions(self) -> tuple[ActionType, ...]:
        return tuple(sorted(self._support))

    def support(self, action: ActionType) -> ActionUplift | None:
        return self._support.get(action)

    # -- fitting -----------------------------------------------------------
    def fit(
        self, rows: Sequence[LoggedDecision], *, seed: int
    ) -> Mapping[ActionType, ActionUplift]:
        """The four steps, in order, exactly as specified.

        1. mu0(x) = E[Y | X, do_nothing];  mu1(x) = E[Y | X, a]
        2. D1_i = Y_i - mu0(x_i)  for treated;  D0_i = mu1(x_i) - Y_i  for control
        3. tau1 = fit(D1);  tau0 = fit(D0)
        4. tau(x) = g(x)*tau0(x) + (1 - g(x))*tau1(x),  g logged not estimated
        """
        # Refuse before fitting anything, so a log without propensities fails
        # loudly at the start rather than producing a plausible model.
        require_propensities(rows)

        control = rows_for(rows, CONTROL_ACTION)
        if len(control) < MIN_ARM_UNITS:
            raise InsufficientTreatedUnits(
                f"the control arm has {len(control)} units, under the {MIN_ARM_UNITS} "
                "floor; without a control arm there is no counterfactual to difference against"
            )

        x_control, y_control = feature_matrix(control), reward_vector(control)

        # Step 1a: the control outcome surface, shared by every action.
        # Bare ndarrays throughout: naming the columns here would make every
        # later predict on an unnamed array warn, and the column order is
        # already the contract (`features.FEATURE_NAMES`).
        self._mu0 = self._regressor(seed).fit(x_control, y_control)
        self._control_units = len(control)

        treated_actions = sorted({row.action for row in rows if row.action is not CONTROL_ACTION})
        for offset, action in enumerate(treated_actions):
            self._fit_action(action, rows, x_control, y_control, seed + offset + 1)

        return dict(self._support)

    def _fit_action(
        self,
        action: ActionType,
        rows: Sequence[LoggedDecision],
        x_control: np.ndarray,
        y_control: np.ndarray,
        seed: int,
    ) -> None:
        treated = rows_for(rows, action)
        x_treated, y_treated = feature_matrix(treated), reward_vector(treated)

        control_mean = float(y_control.mean())
        treated_mean = float(y_treated.mean()) if len(treated) else control_mean
        mean_effect = treated_mean - control_mean

        if len(treated) < MIN_ARM_UNITS:
            # Not enough to fit a surface. The arm-level difference is still a
            # real number and is reported as one, flagged cold so the estimate
            # can never present itself as a confident point.
            self._support[action] = ActionUplift(
                action=action,
                treated_units=len(treated),
                control_units=len(y_control),
                mean_effect=mean_effect,
                cold=True,
            )
            return

        # Step 1b: the treated outcome surface for this action.
        mu1 = self._regressor(seed).fit(x_treated, y_treated)
        assert self._mu0 is not None

        # Step 2: cross-impute. Each side's pseudo-effect is computed with the
        # OTHER side's model, which is what lets the rare arm borrow strength.
        d1 = y_treated - self._mu0.predict(x_treated)
        d0 = mu1.predict(x_control) - y_control

        # Step 3: regress the pseudo-effects.
        self._tau1[action] = self._regressor(seed + 101).fit(x_treated, d1)
        self._tau0[action] = self._regressor(seed + 202).fit(x_control, d0)

        self._support[action] = ActionUplift(
            action=action,
            treated_units=len(treated),
            control_units=len(y_control),
            mean_effect=mean_effect,
            cold=False,
        )

    def _regressor(self, seed: int) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(random_state=seed, **self._params)

    # -- serving -----------------------------------------------------------
    def tau(self, features: np.ndarray, action: ActionType, propensity: np.ndarray) -> np.ndarray:
        """Step 4. `propensity` is g(x): the LOGGED probability of treatment.

        Passed in rather than looked up, because the value that belongs here is
        the one the policy actually drew with at decision time.
        """
        if action is CONTROL_ACTION:
            return np.zeros(len(features), dtype=float)

        support = self._support.get(action)
        if support is None:
            raise KeyError(f"{action} was never fitted; it did not appear in the log")

        features = np.atleast_2d(features)
        if support.cold:
            return np.full(len(features), support.mean_effect, dtype=float)

        g = np.clip(np.asarray(propensity, dtype=float), 0.0, 1.0)
        tau0 = self._tau0[action].predict(features)
        tau1 = self._tau1[action].predict(features)
        return g * tau0 + (1.0 - g) * tau1

    def uplift(
        self,
        obs: ObservableLike,
        action: ActionType,
        ctx: FeatureContext,
        *,
        propensity: float,
    ) -> UpliftEstimate:
        """The contract method. Signed, intervalled, and honest about staleness."""
        stale = ctx.stale_families()
        support = self._support.get(action)
        features = np.array([extract(obs, ctx)], dtype=float)

        if support is None or support.cold or support.treated_units < COLD_START_MIN_OBSERVATIONS:
            value = support.mean_effect if support else 0.0
            return _prior_estimate(action, value, cold_start=True)

        if stale:
            return _prior_estimate(action, support.mean_effect, degraded=True, stale=stale)

        value = float(self.tau(features, action, np.array([propensity]))[0])
        half_width = _effect_half_width(support)
        return UpliftEstimate(
            action=action,
            value=value,
            lower=value - half_width,
            upper=value + half_width,
            basis=EstimateBasis.MODEL,
        )


def _prior_estimate(
    action: ActionType,
    value: float,
    *,
    cold_start: bool = False,
    degraded: bool = False,
    stale: tuple = (),
) -> UpliftEstimate:
    width = MIN_DEGRADED_HALF_WIDTH
    return UpliftEstimate(
        action=action,
        value=value,
        lower=value - width,
        upper=value + width,
        basis=EstimateBasis.SEGMENT_PRIOR,
        degraded=degraded,
        cold_start=cold_start,
        stale_families=stale,
        exploration_boost=COLD_START_EXPLORATION if cold_start else DEGRADED_EXPLORATION,
    )


def _effect_half_width(support: ActionUplift) -> float:
    """Shrinks with the smaller arm, because that is what the estimate rests on."""
    units = max(min(support.treated_units, support.control_units), 1)
    return max(1.0 / float(np.sqrt(units)), 0.01)


# ---------------------------------------------------------------------------
# Evaluation - Qini and decile uplift
#
# Per-unit error is undefined in production: the individual effect is never
# observed. What can be measured is whether the RANKING is real, which is what
# both of these do.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QiniCurve:
    """The curve, the raw area, and a per-unit reading of that area.

    `coefficient` is the area between the model curve and the random line
    divided by the sample size and scaled to a thousand units, which reads as
    "incremental payments gained per 1000 accounts by ranking with this model
    rather than at random".

    IT IS DELIBERATELY NOT NORMALISED BY THE OVERALL GAIN, which is the usual
    textbook denominator. That denominator is the total incremental response of
    treating everyone, and for an action whose true average effect is near zero
    it is near zero too - so the ratio explodes and reports a huge coefficient,
    positive or negative, off pure noise. Measured here, email is exactly that
    action: a true mean effect around one percentage point with the sign split
    almost evenly across the population. Dividing by its overall gain produced
    swings between -0.80 and +0.12 across seeds on identical data.
    """

    fractions: tuple[float, ...]
    gains: tuple[float, ...]
    area: float
    coefficient: float
    overall_gain: float


def qini_curve(
    scores: np.ndarray,
    treated: np.ndarray,
    outcomes: np.ndarray,
    points: int = 20,
) -> QiniCurve:
    """Uplift's analogue of an ROC curve.

    Sort by predicted effect, and at each prefix compare the treated responses
    against the control responses scaled to the same treated-to-control ratio.
    A model that ranks by effect accumulates gain faster than random
    assignment, and the area between the two curves is how much faster.
    """
    scores = np.asarray(scores, dtype=float)
    treated = np.asarray(treated, dtype=bool)
    outcomes = np.asarray(outcomes, dtype=float)
    total = scores.size
    if total == 0:
        raise ValueError("cannot build a Qini curve on an empty sample")
    if treated.all() or not treated.any():
        raise ValueError("a Qini curve needs both treated and control units")

    order = np.argsort(-scores, kind="stable")
    treated_sorted = treated[order]
    outcome_sorted = outcomes[order]

    treated_responses = np.cumsum(outcome_sorted * treated_sorted)
    control_responses = np.cumsum(outcome_sorted * ~treated_sorted)
    treated_count = np.cumsum(treated_sorted)
    control_count = np.cumsum(~treated_sorted)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(control_count > 0, treated_count / np.maximum(control_count, 1), 0.0)
    gain = treated_responses - control_responses * ratio

    overall = float(gain[-1])
    fractions = np.linspace(0.0, 1.0, points + 1)
    indices = np.clip((fractions * total).astype(int) - 1, 0, total - 1)
    curve = np.concatenate([[0.0], gain[indices][1:]])

    # Area between the model curve and the diagonal a random ranking traces.
    random_curve = overall * fractions
    area = float(np.trapezoid(curve - random_curve, fractions))

    return QiniCurve(
        fractions=tuple(float(f) for f in fractions),
        gains=tuple(float(g) for g in curve),
        area=area,
        coefficient=area / total * 1000.0,
        overall_gain=overall,
    )


def decile_uplift(
    scores: np.ndarray, truth: np.ndarray, deciles: int = 10
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean predicted and mean true effect per decile of predicted effect.

    The monotonicity check. Correlating a predicted CATE against a per-unit
    truth punishes a model for variance it cannot see; comparing decile means
    asks the question the model is actually answering, which is whether it can
    order accounts by how much contact helps them.
    """
    scores = np.asarray(scores, dtype=float)
    truth = np.asarray(truth, dtype=float)
    order = np.argsort(scores, kind="stable")
    buckets = np.array_split(order, deciles)
    predicted = tuple(float(scores[bucket].mean()) for bucket in buckets if bucket.size)
    observed = tuple(float(truth[bucket].mean()) for bucket in buckets if bucket.size)
    return predicted, observed
