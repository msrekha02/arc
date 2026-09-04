"""Model A - P(bounce) at T-24h.

THE ONE MODEL WITH NO INTERVENTION IN IT. It asks whether a debit scheduled
for tomorrow will fail, given nothing is done. That makes it an ordinary
binary classification problem, and it is the only one of the three that is.

WHY IT EXISTS. It powers the prevention layer. A bounce predicted twenty-four
hours ahead can be intervened on inside the mandatory pre-debit notification
window, which turns a future failure into a non-event. Prevention is strictly
cheaper than recovery, and at M11 it is reported on its own line, because
money that never failed was never recovered and merging the two inflates the
headline indefensibly.

WHY LIGHTGBM. The data is tabular, mixed-type, with non-linear thresholds and
real missing values - an account with no payment history is a different thing
from an account whose last payment was today, and a GBDT branches on that
natively. It trains in seconds on one CPU core. A neural network would be
slower and worse on this shape.

WHY THE FIT IS A THREE-WAY SPLIT. Train, calibrate, evaluate - each on
disjoint ACCOUNTS. Calibrating on the training split fits the isotonic map to
scores the trees have already memorised, which produces a beautiful reliability
curve and a model that is still wrong in production. Evaluating on the
calibration split does the same thing one layer up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import lightgbm as lgb
import numpy as np

from arc.forecaster.calibration import (
    IsotonicCalibrator,
    ReliabilityCurve,
    UncalibratedScore,
    baseline_pr_auc,
    brier_score,
    pr_auc,
    reliability_curve,
)
from arc.forecaster.dataset import feature_matrix, split_by_account
from arc.forecaster.estimates import (
    COLD_START_MIN_OBSERVATIONS,
    Calibrated,
)
from arc.forecaster.features import FeatureContext, ObservableLike, extract

# Deliberately small. The dataset is thousands of rows, not millions, and a
# deep forest on this shape memorises accounts instead of learning the
# thresholds that generalise.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 220,
    "learning_rate": 0.05,
    "num_leaves": 24,
    "min_child_samples": 40,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
}


@dataclass(frozen=True)
class BounceReport:
    """What the fit produced, on the held-out evaluation split."""

    pr_auc: float
    baseline_pr_auc: float
    brier: float
    reliability: ReliabilityCurve
    train_rows: int
    calibration_rows: int
    evaluation_rows: int
    prevalence: float

    @property
    def expected_calibration_error(self) -> float:
        return self.reliability.expected_calibration_error

    @property
    def pr_auc_lift(self) -> float:
        """How much better than a constant scorer, in ratio terms."""
        return self.pr_auc / self.baseline_pr_auc if self.baseline_pr_auc else float("nan")


@dataclass(frozen=True)
class BounceObservation:
    """One scheduled debit and whether it bounced.

    Separate from `LoggedDecision` because this model is not about an action.
    The label is what the world did to a presentation nobody intervened on.
    """

    account_id: str
    at: datetime
    features: tuple[float, ...]
    bounced: bool


class BounceModel:
    """LightGBM, then isotonic. There is no path that skips the second half."""

    def __init__(self, params: dict[str, object] | None = None) -> None:
        self._params = dict(DEFAULT_PARAMS if params is None else params)
        self._booster: lgb.LGBMClassifier | None = None
        self._calibrator: IsotonicCalibrator | None = None
        self._prior = 0.0
        self._observations = 0

    # -- fitting -----------------------------------------------------------
    def fit(
        self,
        observations: Sequence[BounceObservation],
        *,
        seed: int,
        calibration_holdout: float = 0.25,
        evaluation_holdout: float = 0.25,
    ) -> BounceReport:
        """Train, calibrate, evaluate - on three disjoint account splits.

        Returns the report rather than storing it, so a caller cannot fit and
        then quietly not look at what came out.
        """
        rows = list(observations)
        if len(rows) < 200:
            raise ValueError(
                f"{len(rows)} observations is not enough to fit and honestly "
                "calibrate a bounce model"
            )

        working, evaluation = split_by_account(rows, evaluation_holdout, seed)
        train, calibration = split_by_account(
            working, calibration_holdout / (1.0 - evaluation_holdout), seed + 1
        )

        x_train, y_train = feature_matrix(train), _labels(train)
        if len(np.unique(y_train)) < 2:
            raise ValueError("training split contains a single class; nothing to learn")

        booster = lgb.LGBMClassifier(random_state=seed, **self._params)
        # Fitted on a bare ndarray, deliberately: registering feature names here
        # makes every later predict on an unnamed array warn, and the column
        # ORDER is already the contract (`features.FEATURE_NAMES`).
        booster.fit(x_train, y_train)
        self._booster = booster

        # MANDATORY. Fitted on a split the trees have never seen.
        raw_calibration = self._raw(feature_matrix(calibration))
        calibrator = IsotonicCalibrator().fit(raw_calibration, _labels(calibration))
        self._calibrator = calibrator

        self._prior = float(_labels(train).mean())
        self._observations = len(train)

        y_eval = _labels(evaluation)
        probabilities = self.predict(feature_matrix(evaluation))
        return BounceReport(
            pr_auc=pr_auc(probabilities, y_eval),
            baseline_pr_auc=baseline_pr_auc(y_eval),
            brier=brier_score(probabilities, y_eval),
            reliability=reliability_curve(probabilities, y_eval),
            train_rows=len(train),
            calibration_rows=len(calibration),
            evaluation_rows=len(evaluation),
            prevalence=float(y_eval.mean()),
        )

    # -- serving -----------------------------------------------------------
    def _raw(self, features: np.ndarray) -> np.ndarray:
        if self._booster is None:
            raise UncalibratedScore("bounce model has not been fitted")
        return np.asarray(self._booster.predict_proba(features))[:, 1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Calibrated probabilities.

        Raises rather than returning the raw score when calibration is absent.
        That is the whole enforcement: there is no flag, no keyword, and no
        degraded path that hands back an uncalibrated number, because the
        caller would not be able to tell.
        """
        if self._calibrator is None:
            raise UncalibratedScore(
                "bounce scores are not probabilities until the isotonic "
                "calibrator is fitted; refusing to feed a raw GBDT score into "
                "an expected-value product"
            )
        return self._calibrator(self._raw(np.atleast_2d(features)))

    def p_bounce(self, obs: ObservableLike, ctx: FeatureContext) -> Calibrated[float]:
        """The contract method: one account, one moment, one honest number."""
        stale = ctx.stale_families()
        features = np.array([extract(obs, ctx)], dtype=float)

        if self._observations < COLD_START_MIN_OBSERVATIONS:
            return Calibrated.from_prior(
                self._prior,
                cold_start=True,
                observations=self._observations,
            )
        if stale:
            # Do not extrapolate across a stale family. Fall back to the
            # segment prior, widen, and say which family went cold.
            return Calibrated.from_prior(
                self._prior,
                degraded=True,
                stale_families=stale,
                observations=self._observations,
            )

        value = float(self.predict(features)[0])
        return Calibrated.from_model(
            value,
            half_width=_binomial_half_width(value, self._observations),
            observations=self._observations,
        )


def _binomial_half_width(probability: float, observations: int) -> float:
    """A two-sigma binomial interval, floored so it is never nothing."""
    if observations <= 0:
        return 0.5
    spread = 2.0 * float(np.sqrt(max(probability * (1.0 - probability), 1e-6) / observations))
    return max(spread, 0.01)


def _labels(rows: Sequence[BounceObservation]) -> np.ndarray:
    return np.array([1.0 if row.bounced else 0.0 for row in rows], dtype=float)
