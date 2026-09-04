"""L3 - the Forecaster. Three statistical problems, three techniques.

    Model A  p_bounce      LightGBM + isotonic calibration
    Model B  uplift        X-learner over LightGBM base learners
    Model C  p_ptp_kept    discrete-time hazard, censored, IPW-corrected

Using one technique for all three is the common error. They differ in what is
observable: A has no intervention in it, B has a label that is never observed
per unit, and C has both selection and censoring in its labels.

This package may not import `arc.simulator`, and may not touch the ground-truth
surface by name. Both bans are enforced in CI. A forecaster that reads the
answer key produces a headline number that measures nothing.
"""

from arc.forecaster.bounce import BounceModel, BounceObservation, BounceReport
from arc.forecaster.calibration import (
    IsotonicCalibrator,
    ReliabilityCurve,
    UncalibratedScore,
    expected_calibration_error,
    pr_auc,
    reliability_curve,
)
from arc.forecaster.dataset import (
    LoggedDecision,
    LoggedPropensityMissing,
    PromiseRecord,
    require_propensities,
)
from arc.forecaster.estimates import Calibrated, EstimateBasis, UpliftEstimate
from arc.forecaster.features import (
    TTL,
    EngagementHistory,
    FeatureContext,
    FeatureFamily,
    FeatureFreshness,
    IssuerSignal,
    ObservableLike,
    extract,
)
from arc.forecaster.ptp import PromiseModel, PromiseReport
from arc.forecaster.service import Forecaster, ModelVersions
from arc.forecaster.uplift import XLearner, decile_uplift, qini_curve

__all__ = [
    "TTL",
    "BounceModel",
    "BounceObservation",
    "BounceReport",
    "Calibrated",
    "EngagementHistory",
    "EstimateBasis",
    "FeatureContext",
    "FeatureFamily",
    "FeatureFreshness",
    "Forecaster",
    "IsotonicCalibrator",
    "IssuerSignal",
    "LoggedDecision",
    "LoggedPropensityMissing",
    "ModelVersions",
    "ObservableLike",
    "PromiseModel",
    "PromiseRecord",
    "PromiseReport",
    "ReliabilityCurve",
    "UncalibratedScore",
    "UpliftEstimate",
    "XLearner",
    "decile_uplift",
    "expected_calibration_error",
    "extract",
    "pr_auc",
    "qini_curve",
    "reliability_curve",
    "require_propensities",
]
