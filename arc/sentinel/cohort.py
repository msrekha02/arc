"""Is this systemic? The first question, and the expensive one.

An issuer outage produces hundreds of correlated declines that each look, on
their own, like an ordinary customer failure. Ask the cheap question first and
you dun four hundred people for an incident at their bank.

The detector is an EWMA z-score on the decline rate per cell, against a
seasonal baseline built from the same cell's own recent history. It fires only
when the cell has enough sample to mean anything, and when it does not, it says
so in a distinct verdict that is NEVER coerced to NORMAL.

WHY that distinction is the whole point: for most issuer-instrument
combinations most of the time there is no power. A detector that quietly
answered NORMAL there would restore code-map-first behaviour for the majority
of traffic without anybody noticing it had happened.

Three mechanisms handle the thin case, all of them from the spec:

  BACK-OFF     climb the ladder until a level has sample, and record which
               level answered, so the blind-spot metric is measurable
  SHRINKAGE    partial pooling toward the parent cell, so a cell of eleven
               transactions cannot fire on one unlucky bucket
  INDEPENDENT  the gateway's own downtime feed needs none of our sample, and
    SIGNAL     for a thin issuer it is the primary detector rather than a
               cross-check
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol

from arc.core.time_authority import ensure_utc
from arc.core.types import CohortVerdict, Rail

# EWMA smoothing. Low enough that one busy bucket does not redefine the
# baseline, high enough that a genuine shift is absorbed within a few hours.
ALPHA = 0.25

# z above which a cell is called DEGRADED. Three sigma on a rate that already
# has a floor under its variance.
TAU = 3.0

# Floor under the standard deviation. Without it, a cell whose decline rate has
# been flat produces a divide-by-almost-zero and fires on noise.
SIGMA_FLOOR = 0.02

# Minimum attempts in a cell before its rate means anything. Below this the
# cell has no power and the ladder climbs - it does NOT answer NORMAL.
N_MIN = 12

# Rails a cohort sweep covers. An invoice never passes through an issuer, so
# an issuer incident cannot explain one.
SWEEPABLE_RAILS: tuple[Rail, ...] = (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY)

# Partial-pooling constant. w = n/(n+k), so a cell of size k carries half its
# own weight and borrows half from the level that answered.
SHRINKAGE_K = 20

# A historical bucket needs at least this many attempts before its rate joins
# the baseline. One transaction in a quiet bucket is a rate of exactly 0 or 1,
# and a baseline built from those is mostly noise about how quiet the night was.
MIN_BASELINE_ATTEMPTS = 3


class CohortLevel(StrEnum):
    """The back-off ladder, in the order it is climbed.

    Which level answered is recorded on every result, because "we found
    nothing" and "we found nothing at a resolution coarse enough to hide it"
    are different answers and the dashboard has to tell them apart.
    """

    ISSUER_INSTRUMENT_15M = "issuer_instrument_15m"
    ISSUER_INSTRUMENT_2H = "issuer_instrument_2h"
    ISSUER_DAY = "issuer_day"
    INSTRUMENT_NETWORK_15M = "instrument_network_15m"
    # Not a cell. An external signal that needs none of our sample, which is
    # what makes it the primary detector for a thin issuer.
    DOWNTIME_FEED = "downtime_feed"


LADDER: tuple[CohortLevel, ...] = (
    CohortLevel.ISSUER_INSTRUMENT_15M,
    CohortLevel.ISSUER_INSTRUMENT_2H,
    CohortLevel.ISSUER_DAY,
    CohortLevel.INSTRUMENT_NETWORK_15M,
)

_BUCKET_SECONDS: Mapping[CohortLevel, int] = {
    CohortLevel.ISSUER_INSTRUMENT_15M: 900,
    CohortLevel.ISSUER_INSTRUMENT_2H: 7200,
    CohortLevel.ISSUER_DAY: 86400,
    CohortLevel.INSTRUMENT_NETWORK_15M: 900,
}

# How far back the baseline for each level looks. Long enough to see a normal
# week at coarse resolution, short enough that a fine cell is not compared
# against a different regime.
_LOOKBACK: Mapping[CohortLevel, int] = {
    CohortLevel.ISSUER_INSTRUMENT_15M: 96,
    CohortLevel.ISSUER_INSTRUMENT_2H: 84,
    CohortLevel.ISSUER_DAY: 14,
    CohortLevel.INSTRUMENT_NETWORK_15M: 96,
}

# NOTE ON SHRINKAGE, because the obvious implementation is wrong and the first
# version of this file shipped it.
#
# Pooling a cell toward a parent that CONTAINS it destroys the signal: a
# two-hour outage cell shrunk toward its own day is dragged back to the day
# average, which is mostly the twenty-two normal hours around the incident.
# That version found nothing on an outage that collapses authorisation to 11%.
#
# Pooling across instruments is wrong in the other direction: an eNACH return
# rate has no business setting the expectation for a card cell.
#
# So shrinkage blends the FINEST cell toward the level that ANSWERED, which
# preserves both issuer and instrument and only ever borrows strength from a
# wider window of the same thing. It stabilises the estimate reported for a
# thin cell. It does not decide the verdict.


@dataclass(frozen=True)
class Observation:
    attempts: int = 0
    declines: int = 0

    @property
    def rate(self) -> float:
        return self.declines / self.attempts if self.attempts else 0.0

    def plus(self, *, declined: bool) -> Observation:
        return Observation(self.attempts + 1, self.declines + (1 if declined else 0))


class PresentationLike(Protocol):
    """The minimum a cohort needs from an event, structurally.

    `RawEvent` from L0 satisfies this without importing anything, which keeps
    the Sentinel free of an ingest dependency it does not need.
    """

    issuer_ref: str | None
    rail: Rail
    event_timestamp: datetime
    succeeded: bool


class DowntimeFeed(Protocol):
    """The gateway's own status signal. Independent of our sample.

    For a thin issuer this is the primary detector, not a cross-check: no
    amount of back-off will find an incident in eleven transactions, and the
    honest answer is to use a signal that does not need them.
    """

    def is_degraded(self, issuer: str, at: datetime) -> bool: ...

    def resolves_at(self, issuer: str, at: datetime) -> datetime | None: ...


@dataclass
class StaticDowntimeFeed:
    """A feed backed by declared windows. Injected, never inferred."""

    windows: tuple[tuple[str, datetime, datetime], ...] = ()

    def is_degraded(self, issuer: str, at: datetime) -> bool:
        return any(name == issuer and start <= at < end for name, start, end in self.windows)

    def resolves_at(self, issuer: str, at: datetime) -> datetime | None:
        ends = [end for name, start, end in self.windows if name == issuer and start <= at < end]
        return max(ends) if ends else None


@dataclass
class CohortHistory:
    """Attempts and declines per cell, at every level of the ladder.

    Fed from the accepted event stream at L0, which sees captures as well as
    failures. The denominator is not optional: a burst of declines means
    nothing without the volume it was drawn from, and a system that only saw
    failures could not tell a busy hour from a broken issuer.
    """

    _cells: dict[tuple[CohortLevel, str, int], Observation] = field(default_factory=dict)

    def record(
        self, issuer: str | None, instrument: Rail, at: datetime, *, succeeded: bool
    ) -> None:
        ensure_utc(at)
        for level in LADDER:
            key = (level, _cell_key(level, issuer, instrument), _bucket(level, at))
            self._cells[key] = self._cells.get(key, Observation()).plus(declined=not succeeded)

    @classmethod
    def from_events(cls, events: Iterable[PresentationLike]) -> CohortHistory:
        history = cls()
        for event in events:
            history.record(
                event.issuer_ref, event.rail, event.event_timestamp, succeeded=event.succeeded
            )
        return history

    def cell(
        self, level: CohortLevel, issuer: str | None, instrument: Rail, at: datetime
    ) -> Observation:
        return self._cells.get(
            (level, _cell_key(level, issuer, instrument), _bucket(level, at)), Observation()
        )

    def rate(self, level: CohortLevel, issuer: str | None, instrument: Rail, at: datetime) -> float:
        """The cell's own decline rate, unpooled. The verdict is decided on it."""
        return self.cell(level, issuer, instrument, at).rate

    def shrunk_rate(
        self,
        issuer: str | None,
        instrument: Rail,
        at: datetime,
        answered_at: CohortLevel,
    ) -> float:
        """Partial-pooling estimate for the finest cell, per the build doc.

            r_hat = w * r_cell + (1 - w) * r_parent,   w = n / (n + k)

        The parent is the level that answered, so the pooling borrows strength
        from a wider window of the SAME issuer and instrument. A cell of eleven
        transactions then cannot swing the estimate on one unlucky bucket.
        """
        finest = self.cell(LADDER[0], issuer, instrument, at)
        parent = self.cell(answered_at, issuer, instrument, at)
        if finest.attempts == 0:
            return parent.rate
        if parent.attempts == 0 or answered_at is LADDER[0]:
            return finest.rate
        weight = finest.attempts / (finest.attempts + SHRINKAGE_K)
        return weight * finest.rate + (1.0 - weight) * parent.rate


def _bucket(level: CohortLevel, at: datetime) -> int:
    return int(at.timestamp()) // _BUCKET_SECONDS[level]


def _cell_key(level: CohortLevel, issuer: str | None, instrument: Rail) -> str:
    """What the level groups on. The ladder widens by dropping a dimension."""
    issuer_name = issuer or "unknown"
    if level is CohortLevel.ISSUER_DAY:
        return issuer_name  # instrument dropped
    if level is CohortLevel.INSTRUMENT_NETWORK_15M:
        return str(instrument)  # issuer dropped
    return f"{issuer_name}|{instrument}"


@dataclass(frozen=True)
class CohortResult:
    """What the ladder found, and where it found it.

    `level` is on the result rather than inferred later, because the
    blind-spot metric at M11 counts claims diagnosed without cohort power by
    issuer, and an unmeasured blind spot is a defect while a measured one is a
    known limitation.
    """

    verdict: CohortVerdict
    level: CohortLevel | None
    attempts: int
    rate: float
    baseline: float
    z: float
    # The partial-pooling estimate for the finest cell. Reported rather than
    # used for the verdict: it is what an operator wants when asking how bad
    # this specific issuer and instrument look right now.
    shrunk_rate: float = 0.0
    degraded_until: datetime | None = None
    # The instrument the degradation was actually found on. Different from the
    # one asked about when the incident was visible on a busier rail at the
    # same issuer.
    detected_on: Rail | None = None

    @property
    def has_power(self) -> bool:
        return self.verdict is not CohortVerdict.INSUFFICIENT_POWER


def ewma_baseline(rates: list[float]) -> tuple[float, float]:
    """Mean and variance of a rate series, per the build doc's recursion.

        mu_t   = a*r_t + (1-a)*mu_{t-1}
        sig2_t = a*(r_t - mu_{t-1})^2 + (1-a)*sig2_{t-1}

    Seeded from the first observation with zero variance, which the sigma floor
    then keeps from producing an infinite z on the second.
    """
    if not rates:
        return 0.0, 0.0
    mu = rates[0]
    sigma2 = 0.0
    for rate in rates[1:]:
        sigma2 = ALPHA * (rate - mu) ** 2 + (1.0 - ALPHA) * sigma2
        mu = ALPHA * rate + (1.0 - ALPHA) * mu
    return mu, sigma2


def _z_at(history: CohortHistory, level, issuer, instrument, at) -> tuple[float, float, float]:
    """The current bucket's z against its own recent history at this level."""
    seconds = _BUCKET_SECONDS[level]
    lookback = _LOOKBACK[level]

    series: list[float] = []
    for step in range(lookback, 0, -1):
        moment = at - timedelta(seconds=seconds * step)
        if history.cell(level, issuer, instrument, moment).attempts >= MIN_BASELINE_ATTEMPTS:
            series.append(history.rate(level, issuer, instrument, moment))

    mu_prev, sigma2_prev = ewma_baseline(series)
    rate = history.rate(level, issuer, instrument, at)

    # THE ONE DELIBERATE DEVIATION FROM THE SPECIFIED FORMULA, and why.
    #
    # The build doc gives the recursion as
    #
    #     mu_t   = a*r_t + (1-a)*mu_{t-1}
    #     sig2_t = a*(r_t - mu_{t-1})^2 + (1-a)*sig2_{t-1}
    #     z_t    = (r_t - mu_t) / max(sqrt(sig2_t), floor)
    #
    # Read literally, z is tested against a mean and a variance that have both
    # already absorbed r_t. Substituting, with d = r_t - mu_{t-1}:
    #
    #     r_t - mu_t = (1-a)*d
    #     sig2_t     >= a*d^2
    #     z_t        <= (1-a)*d / sqrt(a*d^2) = (1-a)/sqrt(a)
    #
    # which at a = 0.25 is exactly 1.5, for ANY anomaly however large. With
    # tau at 3.0 the detector could never fire, and an outage that collapses
    # authorisation from 13% to 89% would score 1.4. `test_the_specified_z_
    # formula_is_self_limiting` proves that bound rather than asserting it.
    #
    # So the current observation is tested against the PRIOR baseline, which is
    # what an EWMA control chart does and what makes z mean anything. The
    # doc's recursion is kept verbatim as the UPDATE rule below - it is a
    # perfectly good update, it is just not a test.
    z = (rate - mu_prev) / max(math.sqrt(sigma2_prev), SIGMA_FLOOR)

    # The update, exactly as specified, carrying the baseline forward.
    _sigma2_next = ALPHA * (rate - mu_prev) ** 2 + (1.0 - ALPHA) * sigma2_prev
    _mu_next = ALPHA * rate + (1.0 - ALPHA) * mu_prev

    return z, rate, mu_prev


def cohort_check(
    issuer: str | None,
    instrument: Rail,
    at: datetime,
    history: CohortHistory,
    *,
    downtime: DowntimeFeed | None = None,
) -> CohortResult:
    """DEGRADED, NORMAL, or INSUFFICIENT_POWER. Never a silent NORMAL.

    The independent feed is consulted first because it needs none of our
    sample. Then the ladder is climbed until a level has power. If no level
    does, the answer is INSUFFICIENT_POWER and the caller has to deal with it -
    there is no branch here that turns "we could not tell" into "nothing is
    wrong".
    """
    ensure_utc(at)

    if downtime is not None and downtime.is_degraded(str(issuer), at):
        return CohortResult(
            verdict=CohortVerdict.DEGRADED,
            level=CohortLevel.DOWNTIME_FEED,
            attempts=0,
            rate=1.0,
            baseline=0.0,
            z=math.inf,
            degraded_until=downtime.resolves_at(str(issuer), at),
            detected_on=instrument,
        )

    own = _evaluate(issuer, instrument, at, history)
    if own.verdict is CohortVerdict.DEGRADED:
        return own

    # An incident is an event at the ISSUER, not at one of its rails. It is
    # routinely visible only on the busiest instrument, because that is where
    # the sample is, and the thin rails of the same issuer are failing for
    # exactly the same reason. Carrying the finding across errs toward zero
    # customer contact, which is the only direction an issuer-layer cause
    # should ever err in.
    for sibling in SWEEPABLE_RAILS:
        if sibling is instrument:
            continue
        found = _evaluate(issuer, sibling, at, history)
        if found.verdict is CohortVerdict.DEGRADED:
            return CohortResult(
                verdict=CohortVerdict.DEGRADED,
                level=found.level,
                attempts=found.attempts,
                rate=found.rate,
                baseline=found.baseline,
                z=found.z,
                shrunk_rate=found.shrunk_rate,
                degraded_until=found.degraded_until,
                detected_on=sibling,
            )
    return own


def _evaluate(
    issuer: str | None, instrument: Rail, at: datetime, history: CohortHistory
) -> CohortResult:
    """Climb the ladder for one issuer and instrument. No sibling sweep."""
    for level in LADDER:
        observed = history.cell(level, issuer, instrument, at)
        if observed.attempts < N_MIN:
            continue

        z, rate, baseline = _z_at(history, level, issuer, instrument, at)
        verdict = CohortVerdict.DEGRADED if z > TAU and rate > baseline else CohortVerdict.NORMAL
        return CohortResult(
            verdict=verdict,
            level=level,
            attempts=observed.attempts,
            rate=rate,
            baseline=baseline,
            z=z,
            shrunk_rate=history.shrunk_rate(issuer, instrument, at, level),
            degraded_until=(
                at + timedelta(seconds=_BUCKET_SECONDS[level])
                if verdict is CohortVerdict.DEGRADED
                else None
            ),
            detected_on=instrument if verdict is CohortVerdict.DEGRADED else None,
        )

    # Every rung was too thin. This is the common case, and saying so is the
    # entire reason the verdict has three members instead of two.
    deepest = history.cell(LADDER[0], issuer, instrument, at)
    return CohortResult(
        verdict=CohortVerdict.INSUFFICIENT_POWER,
        level=None,
        attempts=deepest.attempts,
        rate=deepest.rate,
        baseline=0.0,
        z=0.0,
    )
