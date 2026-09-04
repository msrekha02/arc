"""Model C - P(promise kept).

TWO STATISTICAL PROBLEMS AT ONCE, AND USING ONE TECHNIQUE FOR BOTH IS THE
COMMON ERROR.

CENSORING. A promise dated the 20th is neither kept nor broken on the 18th. It
is unresolved, and unresolved is a third answer rather than a soft version of
broken. Coding it broken biases the model pessimistic in a specific and nasty
way: the promises still in flight at any analysis moment are disproportionately
the RECENT ones, and recent promises are disproportionately the ones about to
be kept. So the bias is not noise, it is a systematic pull toward "nobody
pays", concentrated on exactly the population being scored.

The discrete-time hazard model is what handles it properly. Each promise is
expanded into one row per day it was actually observed. A kept promise
contributes zeros up to its keeping day and a one on that day; a broken promise
contributes zeros for its whole horizon; a censored promise contributes zeros
for the days observed and then simply stops. It never gets a zero for the days
nobody watched. That is the entire mechanism, and it is why the expansion
exists rather than a single logistic regression over promises.

SELECTION. Kept-versus-broken is only observed for promises that were MADE, and
promises are made by people who were contacted, who answered, and who engaged.
That population is not the population the model is asked about. The correction
is inverse-probability weighting on P(promise made), so the promises that came
from rarely-promising segments carry the weight of the segment they represent.

The hazard model itself is a small gradient-boosted tree rather than a plain
logistic regression: the strongest single feature is the gap between the
promise date and the inferred salary day, and its effect is sharply non-linear
around zero - a promise dated three days BEFORE the credit is a materially
different object from one dated the day after. A linear term through that
would average the two into nothing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from arc.forecaster.dataset import PromiseRecord, feature_matrix, split_by_account
from arc.forecaster.estimates import COLD_START_MIN_OBSERVATIONS, Calibrated
from arc.forecaster.features import FeatureContext, ObservableLike, extract

HAZARD_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 160,
    "learning_rate": 0.06,
    "num_leaves": 16,
    "min_child_samples": 40,
    "reg_lambda": 2.0,
    "verbose": -1,
}

# Importance weights are clipped. A promise from a segment that promises once
# in five hundred times would otherwise carry five hundred units of weight and
# the fit would be that one promise.
MAX_IPW = 20.0

# Longest promise horizon modelled. Beyond three weeks a promise is a wish.
MAX_HORIZON_DAYS = 21


@dataclass(frozen=True)
class PromiseReport:
    promises: int
    kept: int
    broken: int
    censored: int
    person_periods: int
    mean_weight: float

    @property
    def naive_broken_rate(self) -> float:
        """What the pessimistic coding would have reported.

        Kept over everything, with censored promises counted as failures. Held
        next to `observed_broken_rate` it shows the size of the bias the
        censoring treatment avoids.
        """
        total = self.promises
        return (total - self.kept) / total if total else 0.0

    @property
    def observed_broken_rate(self) -> float:
        """Broken over resolved. The honest denominator."""
        resolved = self.kept + self.broken
        return self.broken / resolved if resolved else 0.0


class PromiseModel:
    """Discrete-time hazard over person-periods, IPW-corrected for selection."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self._params = dict(HAZARD_PARAMS if params is None else params)
        self._hazard: lgb.LGBMClassifier | None = None
        self._prior = 0.0
        self._observations = 0

    # -- fitting -----------------------------------------------------------
    def fit(self, promises: Sequence[PromiseRecord], *, seed: int) -> PromiseReport:
        rows = list(promises)
        if not rows:
            raise ValueError("no promises to fit")

        design, labels, weights = self._expand(rows)
        if design.size == 0:
            raise ValueError("expansion produced no person-periods")
        if len(np.unique(labels)) < 2:
            raise ValueError("no variation in the hazard label; nothing to learn")

        model = lgb.LGBMClassifier(random_state=seed, **self._params)
        model.fit(design, labels, sample_weight=weights)
        self._hazard = model

        resolved = [row for row in rows if not row.censored]
        self._prior = (
            float(np.mean([1.0 if row.kept else 0.0 for row in resolved])) if resolved else 0.0
        )
        self._observations = len(resolved)

        return PromiseReport(
            promises=len(rows),
            kept=sum(1 for row in rows if row.kept),
            broken=sum(1 for row in rows if row.broken),
            censored=sum(1 for row in rows if row.censored),
            person_periods=int(design.shape[0]),
            mean_weight=float(weights.mean()),
        )

    def _expand(
        self, promises: Sequence[PromiseRecord]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One row per observed day, per promise.

        THE CENSORING LIVES HERE. `observed_days` stops at the analysis moment
        for an unresolved promise, so those days after it contribute nothing at
        all - not a zero, nothing. A zero would be a claim that the promise was
        watched and not kept on that day, and it was not watched.
        """
        design: list[list[float]] = []
        labels: list[float] = []
        weights: list[float] = []

        for promise in promises:
            observed = min(promise.observed_days, MAX_HORIZON_DAYS)
            if observed <= 0:
                continue
            weight = _ipw(promise.selection_propensity)
            kept_day = observed if promise.kept else None

            for period in range(1, observed + 1):
                design.append([*promise.features, float(period), float(promise.horizon_days)])
                labels.append(1.0 if kept_day is not None and period == kept_day else 0.0)
                weights.append(weight)

        return (
            np.array(design, dtype=float),
            np.array(labels, dtype=float),
            np.array(weights, dtype=float),
        )

    # -- serving -----------------------------------------------------------
    def _period_hazards(self, features: Sequence[float], horizon: int) -> np.ndarray:
        if self._hazard is None:
            raise RuntimeError("promise model has not been fitted")
        horizon = max(min(horizon, MAX_HORIZON_DAYS), 1)
        design = np.array(
            [[*features, float(period), float(horizon)] for period in range(1, horizon + 1)],
            dtype=float,
        )
        return np.asarray(self._hazard.predict_proba(design))[:, 1]

    def survival(self, features: Sequence[float], horizon: int) -> float:
        """P(kept by the due date) = 1 - product over periods of (1 - hazard)."""
        hazards = np.clip(self._period_hazards(features, horizon), 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - hazards))

    def p_ptp_kept(
        self, promise: PromiseRecord, obs: ObservableLike, ctx: FeatureContext
    ) -> Calibrated[float]:
        stale = ctx.stale_families()
        if self._observations < COLD_START_MIN_OBSERVATIONS:
            return Calibrated.from_prior(
                self._prior, cold_start=True, observations=self._observations
            )
        if stale:
            return Calibrated.from_prior(
                self._prior,
                degraded=True,
                stale_families=stale,
                observations=self._observations,
            )

        features = extract(obs, ctx)
        value = self.survival(features, promise.horizon_days)
        half_width = max(
            2.0 * float(np.sqrt(max(value * (1.0 - value), 1e-6) / max(self._observations, 1))),
            0.01,
        )
        return Calibrated.from_model(
            min(max(value, 0.0), 1.0),
            half_width=min(half_width, min(value, 1.0 - value) if 0 < value < 1 else half_width),
            observations=self._observations,
        )


def _ipw(selection_propensity: float | None) -> float:
    """Weight by the inverse of the probability this promise was ever asked for.

    An absent propensity weights at one rather than raising: unlike the
    Allocator's action propensity, which is a fact the policy recorded, the
    selection probability is a modelled quantity and its absence means the
    promise was observed outside a randomised offer, not that a log is broken.
    """
    if selection_propensity is None or not np.isfinite(selection_propensity):
        return 1.0
    if selection_propensity <= 0.0:
        return MAX_IPW
    return float(min(1.0 / selection_propensity, MAX_IPW))


def selection_propensities(
    promises: Sequence[PromiseRecord],
    contacted: Sequence[tuple[str, tuple[float, ...]]],
    *,
    seed: int,
) -> dict[str, float]:
    """P(promise made | contacted), fitted on everyone who was asked.

    The denominator is the contacted population, not the whole portfolio: a
    subject nobody called had no opportunity to promise, and including them
    would be modelling the Allocator's choices rather than the customer's.
    """
    if not contacted:
        return {}
    promised = {promise.account_id for promise in promises}
    design = np.array([features for _, features in contacted], dtype=float)
    labels = np.array(
        [1.0 if account_id in promised else 0.0 for account_id, _ in contacted], dtype=float
    )
    if len(np.unique(labels)) < 2:
        rate = float(labels.mean()) or 1.0
        return {account_id: rate for account_id, _ in contacted}

    model = lgb.LGBMClassifier(
        random_state=seed,
        objective="binary",
        n_estimators=120,
        learning_rate=0.06,
        num_leaves=12,
        min_child_samples=40,
        verbose=-1,
    )
    model.fit(design, labels)
    scores = np.asarray(model.predict_proba(design))[:, 1]
    # Floored so the inverse cannot explode before the clip even sees it.
    scores = np.clip(scores, 1.0 / MAX_IPW, 1.0)
    return {
        account_id: float(score) for (account_id, _), score in zip(contacted, scores, strict=True)
    }


def split_promises(
    promises: Sequence[PromiseRecord], holdout: float, seed: int
) -> tuple[list[PromiseRecord], list[PromiseRecord]]:
    return split_by_account(promises, holdout, seed)


def promise_matrix(promises: Sequence[PromiseRecord]) -> np.ndarray:
    return feature_matrix(promises)
