"""The Forecaster facade - the four methods M8 will call.

It holds the three models and one rule they all obey: an estimate never
presents itself as more confident than its inputs allow. Staleness and cold
start are checked HERE as well as inside each model, because the Allocator
consumes all three through this object and a single place to ask "is this
degraded" is what lets it restrict itself to conservative actions without
knowing which model went cold.

Nothing here reads a clock. `at` arrives inside the `FeatureContext`, for the
same reason the Gate takes it as a parameter: a forecast that depends on when
it was asked cannot be replayed, and a decision that cannot be replayed cannot
be audited.
"""

from __future__ import annotations

from dataclasses import dataclass

from arc.core.types import ActionType
from arc.forecaster.bounce import BounceModel
from arc.forecaster.dataset import PromiseRecord
from arc.forecaster.estimates import Calibrated, UpliftEstimate
from arc.forecaster.features import FeatureContext, FeatureFamily, ObservableLike
from arc.forecaster.ptp import PromiseModel
from arc.forecaster.uplift import XLearner


@dataclass(frozen=True)
class ModelVersions:
    """Pinned into every decision record, so a replay knows what decided."""

    bounce: str
    uplift: str
    ptp: str
    feature_contract: str


class Forecaster:
    """One object, three models, one staleness rule."""

    def __init__(
        self,
        *,
        bounce: BounceModel,
        uplift: XLearner,
        ptp: PromiseModel,
        versions: ModelVersions | None = None,
    ) -> None:
        self._bounce = bounce
        self._uplift = uplift
        self._ptp = ptp
        self._versions = versions or ModelVersions("m7", "m7", "m7", "m7")

    def versions(self) -> ModelVersions:
        return self._versions

    def is_degraded(self, obs: ObservableLike, ctx: FeatureContext) -> bool:
        """True when any feature family this account depends on is past its TTL.

        The Allocator must respect this by restricting to conservative actions.
        Takes `obs` for signature symmetry with the other three and because a
        later refinement may make freshness account-specific.
        """
        del obs
        return bool(ctx.stale_families())

    def stale_families(self, ctx: FeatureContext) -> tuple[FeatureFamily, ...]:
        return ctx.stale_families()

    def p_bounce(self, obs: ObservableLike, ctx: FeatureContext) -> Calibrated[float]:
        return self._bounce.p_bounce(obs, ctx)

    def uplift(
        self,
        obs: ObservableLike,
        action: ActionType,
        ctx: FeatureContext,
        *,
        propensity: float,
    ) -> UpliftEstimate:
        return self._uplift.uplift(obs, action, ctx, propensity=propensity)

    def p_ptp_kept(
        self, promise: PromiseRecord, obs: ObservableLike, ctx: FeatureContext
    ) -> Calibrated[float]:
        return self._ptp.p_ptp_kept(promise, obs, ctx)

    def sleeping_dog(
        self,
        obs: ObservableLike,
        ctx: FeatureContext,
        actions: tuple[ActionType, ...],
        *,
        propensity: float,
    ) -> bool:
        """Every contact action scores negative.

        A MODEL OUTPUT, NOT A HEURISTIC. Nothing here knows why an account is
        better left alone; it knows that the estimated effect of reaching them
        is below zero on every channel available, which is the same statement
        the Allocator needs to choose silence.
        """
        estimates = [self.uplift(obs, action, ctx, propensity=propensity) for action in actions]
        return bool(estimates) and all(estimate.value < 0.0 for estimate in estimates)
