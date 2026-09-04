"""Isotonic calibration, and the metrics that decide whether it worked.

WHY THIS MODULE EXISTS AT ALL. A gradient-boosted tree trained on binary
logloss emits a score, not a probability. The score ranks well and is
systematically wrong in level - typically over-confident at both tails. That
is harmless if the number is only ever compared to other numbers, and it is
not harmless here: at M8 the bounce probability enters an expected-value
product, multiplied by a rupee amount and traded against a budget. An
uncalibrated 0.8 that is truly 0.55 does not announce itself; it quietly
misprices every allocation downstream, and the headline number inherits the
error with no symptom.

So calibration is not a post-processing step that can be skipped under time
pressure. `BounceModel` cannot emit a probability without it - the predict
path raises rather than falling back to the raw score.

ISOTONIC RATHER THAN PLATT. Platt scaling assumes the distortion is a
sigmoid; a GBDT's distortion is not, and a monotone step function fits it
without that assumption. The price is a hunger for data and a tendency to
overfit small held-out sets, which is why the calibration split is separate
from both the training and the evaluation split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression


class UncalibratedScore(RuntimeError):
    """A raw model score was asked to behave like a probability."""


@dataclass(frozen=True)
class ReliabilityCurve:
    """Predicted probability against observed frequency, per bin."""

    edges: tuple[float, ...]
    predicted: tuple[float, ...]
    observed: tuple[float, ...]
    counts: tuple[int, ...]

    @property
    def expected_calibration_error(self) -> float:
        """Count-weighted mean gap between claim and reality."""
        total = sum(self.counts)
        if total == 0:
            return 0.0
        return (
            sum(
                count * abs(predicted - observed)
                for count, predicted, observed in zip(
                    self.counts, self.predicted, self.observed, strict=True
                )
            )
            / total
        )

    @property
    def max_calibration_error(self) -> float:
        gaps = [
            abs(predicted - observed)
            for count, predicted, observed in zip(
                self.counts, self.predicted, self.observed, strict=True
            )
            if count
        ]
        return max(gaps) if gaps else 0.0


class IsotonicCalibrator:
    """Monotone map from raw score to probability, fitted on held-out data."""

    def __init__(self) -> None:
        self._model: IsotonicRegression | None = None
        self._observations = 0

    @property
    def observations(self) -> int:
        return self._observations

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if scores.shape != labels.shape:
            raise ValueError(f"score/label shape mismatch: {scores.shape} vs {labels.shape}")
        if scores.size == 0:
            raise ValueError("cannot calibrate on an empty held-out split")
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(scores, labels)
        self._model = model
        self._observations = int(scores.size)
        return self

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise UncalibratedScore(
                "calibrator was never fitted; a raw GBDT score is not a probability"
            )
        return np.clip(self._model.predict(np.asarray(scores, dtype=float)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def reliability_curve(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> ReliabilityCurve:
    """Equal-width bins over [0, 1]. Empty bins are reported with count 0
    rather than dropped, so a model that never predicts above 0.4 shows it."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # The last bin is closed at 1.0; every other one is half-open, like every
    # window in ARC.
    index = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)

    predicted: list[float] = []
    observed: list[float] = []
    counts: list[int] = []
    for bucket in range(bins):
        mask = index == bucket
        count = int(mask.sum())
        counts.append(count)
        predicted.append(float(probabilities[mask].mean()) if count else 0.0)
        observed.append(float(labels[mask].mean()) if count else 0.0)

    return ReliabilityCurve(
        edges=tuple(float(edge) for edge in edges),
        predicted=tuple(predicted),
        observed=tuple(observed),
        counts=tuple(counts),
    )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    return reliability_curve(probabilities, labels, bins).expected_calibration_error


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error on probabilities - discrimination and calibration
    together, which is why it is reported next to PR-AUC rather than instead."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision.

    PR-AUC AND NOT ROC-AUC. Bounces are the minority class, and ROC-AUC's
    denominator is dominated by the true negatives that are easy to get right.
    It flatters a model that has learned the majority class and little else.
    Precision-recall keeps the positives in the denominator where they belong.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    positives = float(labels.sum())
    if positives == 0 or positives == labels.size:
        raise ValueError("average precision is undefined when one class is absent")

    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, ranked.size + 1)
    # Sum precision at each positive, divided by the positive count: average
    # precision as the area under the step-wise PR curve.
    return float((precision * ranked).sum() / positives)


def baseline_pr_auc(labels: np.ndarray) -> float:
    """What a constant scorer achieves: the prevalence. The bar PR-AUC beats."""
    labels = np.asarray(labels, dtype=float)
    return float(labels.mean())
