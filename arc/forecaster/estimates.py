"""What a forecast is allowed to look like.

Every number this package returns is wrapped, and the wrapper is where the
discipline lives rather than in the call sites that consume it.

THREE RULES, ENFORCED IN `__post_init__` RATHER THAN DOCUMENTED:

1. A degraded estimate cannot claim to be a model output. Past a family's TTL
   the answer comes from a segment prior with a widened interval, and the
   basis says so. Silent extrapolation on stale features is how these systems
   produce confident nonsense.

2. A cold-start estimate cannot be a confident point estimate. Below the
   observation floor the answer is the segment prior plus elevated
   exploration. The agent's own choices generate its future training data, so
   an early confident error becomes self-confirming - the interval is what
   keeps the Allocator sampling instead of committing.

3. An interval always contains its point. A model that reports a value outside
   its own interval is not reporting uncertainty, it is reporting noise.

`Calibrated` is generic because the three models return different things and
the discipline is the same for all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from arc.core.types import ActionType
from arc.forecaster.features import FeatureFamily

T = TypeVar("T")

# Below this many observations in a segment there is no point estimate worth
# reporting, only a prior. The floor is deliberately generous: the cost of an
# unnecessary interval is a little exploration, and the cost of a confident
# wrong answer is a policy that trains on its own mistake.
COLD_START_MIN_OBSERVATIONS = 50

# How much exploration mass a cold or degraded estimate asks the Allocator to
# add on top of its epsilon floor. Reported as a `B_explore` line at M11 so it
# reads as deliberate rather than as waste.
COLD_START_EXPLORATION = 0.15
DEGRADED_EXPLORATION = 0.05

# Minimum half-width once an estimate is degraded or cold. An interval that
# widens by nothing has not acknowledged anything.
MIN_DEGRADED_HALF_WIDTH = 0.10


class EstimateBasis(StrEnum):
    """Where the number came from. Recorded on every estimate."""

    MODEL = "model"
    SEGMENT_PRIOR = "segment_prior"


class ConfidenceViolation(ValueError):
    """An estimate claimed more confidence than its provenance allows."""


@dataclass(frozen=True)
class Calibrated(Generic[T]):
    """A number, its interval, and an honest account of where it came from."""

    value: T
    lower: float
    upper: float
    basis: EstimateBasis = EstimateBasis.MODEL
    degraded: bool = False
    cold_start: bool = False
    stale_families: tuple[FeatureFamily, ...] = ()
    exploration_boost: float = 0.0
    observations: int | None = None

    def __post_init__(self) -> None:
        point = float(self.value)  # type: ignore[arg-type]
        if not self.lower <= point <= self.upper:
            raise ConfidenceViolation(
                f"point estimate {point} outside its own interval [{self.lower}, {self.upper}]"
            )
        if self.cold_start and self.basis is EstimateBasis.MODEL:
            raise ConfidenceViolation(
                "a cold-start estimate cannot be a model point estimate; "
                "below the observation floor the answer is a segment prior"
            )
        if self.degraded and self.basis is EstimateBasis.MODEL:
            raise ConfidenceViolation(
                "a degraded estimate cannot be a model point estimate; "
                f"stale families: {[str(f) for f in self.stale_families] or 'unrecorded'}"
            )
        if (self.degraded or self.cold_start) and self.half_width < MIN_DEGRADED_HALF_WIDTH:
            raise ConfidenceViolation(
                f"interval half-width {self.half_width:.4f} is under the "
                f"{MIN_DEGRADED_HALF_WIDTH} floor for a degraded or cold estimate"
            )
        if (self.degraded or self.cold_start) and self.exploration_boost <= 0.0:
            raise ConfidenceViolation(
                "a degraded or cold estimate must raise exploration, not merely flag itself"
            )

    @property
    def half_width(self) -> float:
        return (self.upper - self.lower) / 2.0

    @property
    def is_confident(self) -> bool:
        """True only for a model output on fresh features with enough data."""
        return self.basis is EstimateBasis.MODEL and not self.degraded and not self.cold_start

    @classmethod
    def from_model(
        cls, value: T, *, half_width: float, observations: int | None = None
    ) -> Calibrated[T]:
        point = float(value)  # type: ignore[arg-type]
        return cls(
            value=value,
            lower=point - half_width,
            upper=point + half_width,
            basis=EstimateBasis.MODEL,
            observations=observations,
        )

    @classmethod
    def from_prior(
        cls,
        value: T,
        *,
        half_width: float = MIN_DEGRADED_HALF_WIDTH,
        degraded: bool = False,
        cold_start: bool = False,
        stale_families: tuple[FeatureFamily, ...] = (),
        observations: int | None = None,
    ) -> Calibrated[T]:
        """The fallback. Widened, flagged, and asking for exploration."""
        point = float(value)  # type: ignore[arg-type]
        width = max(half_width, MIN_DEGRADED_HALF_WIDTH)
        boost = COLD_START_EXPLORATION if cold_start else DEGRADED_EXPLORATION
        return cls(
            value=value,
            lower=point - width,
            upper=point + width,
            basis=EstimateBasis.SEGMENT_PRIOR,
            degraded=degraded,
            cold_start=cold_start,
            stale_families=stale_families,
            exploration_boost=boost,
            observations=observations,
        )


@dataclass(frozen=True)
class UpliftEstimate:
    """A signed conditional average treatment effect for one action.

    SIGNED IS THE POINT. `value < 0` means contacting this account reduces
    recovery, and the sleeping-dog rule at M8 - silent actions only when every
    contact action scores negative - is then a model output rather than a
    hand-written heuristic.
    """

    action: ActionType
    value: float
    lower: float
    upper: float
    basis: EstimateBasis = EstimateBasis.MODEL
    degraded: bool = False
    cold_start: bool = False
    stale_families: tuple[FeatureFamily, ...] = ()
    exploration_boost: float = 0.0

    def __post_init__(self) -> None:
        # Reuse one implementation of the discipline rather than restating it.
        Calibrated(
            value=self.value,
            lower=self.lower,
            upper=self.upper,
            basis=self.basis,
            degraded=self.degraded,
            cold_start=self.cold_start,
            stale_families=self.stale_families,
            exploration_boost=self.exploration_boost,
        )

    @property
    def is_confident(self) -> bool:
        return self.basis is EstimateBasis.MODEL and not self.degraded and not self.cold_start

    @property
    def sleeping_dog(self) -> bool:
        """Confidently negative: the interval does not reach zero."""
        return self.upper < 0.0
