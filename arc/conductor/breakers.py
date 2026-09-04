"""The ten circuit breakers, and the three that watch the watcher.

SEVEN OF THESE MEASURE HARM TO PEOPLE. Complaints, opt-outs, cancellations,
volume, channel failures, sentiment - a recovery number bought with any of
these is not a win, it is a cost moved into a quarter where nobody is looking
for it.

THREE OF THEM MEASURE WHETHER THE SYSTEM'S OWN MACHINERY IS WORKING, and having
them at all is what separates an engineered system from a demo:

    CB-VETO          post-allocation Gate veto rate above 2%
                     `project` and `certify` share one registry and one
                     evaluator, so only RUNTIME-class rules can fire after
                     allocation. A veto rate above two percent does not mean
                     the Gate is strict; it means the eligibility projection is
                     broken and the allocator is optimising over a candidate
                     set it is not allowed to have.

    CB-DEGRADED      share of decisions taken on stale features above 20%
                     Silent extrapolation past a TTL is how these systems
                     produce confident nonsense. The flag exists at M7; this is
                     what makes ignoring it impossible.

    CB-COHORT-BLIND  share diagnosed without cohort power above 40%
                     THE HONEST ONE. M6's hierarchical back-off cannot always
                     find power - the forty-minute outage on a thin issuer is
                     the case it legitimately misses - and this is where that
                     miss SURFACES rather than passing as a clean NORMAL. An
                     unmeasured blind spot is a defect; a measured one is a
                     known limitation, and the difference between those two is
                     this breaker.

EVERY BREAKER TRIPS TO SHADOW, NOT TO OFF. Shadow keeps L0-L5 running and the
ledger filling, so the diagnosis of whatever tripped it is being recorded while
it is tripped. A system that goes dark when it detects a problem destroys the
evidence about the problem.

RATIOS AGAINST A BASELINE, NOT ABSOLUTE COUNTS. A complaint rate is meaningless
without the volume it came from and the level it is normally at. Every
threshold here is either a multiple of a trailing median or an absolute share
whose denominator is stated in the metric's own name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from arc.core.time_authority import ensure_utc


class BreakerId(StrEnum):
    """The ten. Closed: a breaker that is not in this enum does not exist."""

    COMPLAINT = "CB-COMPLAINT"
    OPTOUT = "CB-OPTOUT"
    CANCEL = "CB-CANCEL"
    VOLUME = "CB-VOLUME"
    VETO = "CB-VETO"
    DEGRADED = "CB-DEGRADED"
    COHORT_BLIND = "CB-COHORT-BLIND"
    CHANNEL_FAIL = "CB-CHANNEL-FAIL"
    RESUME_RAMP = "CB-RESUME-RAMP"
    SENTIMENT = "CB-SENTIMENT"


class Comparison(StrEnum):
    """How a breaker's threshold is expressed.

    RATIO_OF_BASELINE multiplies a trailing median; ABSOLUTE_SHARE is a
    fraction with a stated denominator. Keeping the distinction explicit stops
    a reader from assuming 1.5 and 0.02 mean the same kind of thing.
    """

    RATIO_OF_BASELINE = "ratio_of_baseline"
    ABSOLUTE_SHARE = "absolute_share"


@dataclass(frozen=True)
class BreakerSpec:
    breaker_id: BreakerId
    metric: str
    comparison: Comparison
    threshold: float
    rationale: str
    self_monitoring: bool = False


# The catalogue, from spec section 7.6. Thresholds live here beside their
# rationale rather than in a config file, because a breaker whose threshold can
# be edited without reading why it is what it is will be edited under pressure.
SPECS: Mapping[BreakerId, BreakerSpec] = {
    BreakerId.COMPLAINT: BreakerSpec(
        BreakerId.COMPLAINT,
        "complaints_per_1000_contacts",
        Comparison.RATIO_OF_BASELINE,
        1.5,
        "Recovery that generates complaints is not recovery, it is a cost deferred.",
    ),
    BreakerId.OPTOUT: BreakerSpec(
        BreakerId.OPTOUT,
        "opt_outs_per_1000_contacts",
        Comparison.RATIO_OF_BASELINE,
        1.5,
        "An opt-out destroys the channel for every future claim this person holds.",
    ),
    BreakerId.CANCEL: BreakerSpec(
        BreakerId.CANCEL,
        "voluntary_cancel_rate_treated_over_control",
        Comparison.RATIO_OF_BASELINE,
        1.5,
        "The sleeping-dog check. Treated cancelling above control is value destroyed.",
    ),
    BreakerId.VOLUME: BreakerSpec(
        BreakerId.VOLUME,
        "outbound_volume",
        Comparison.RATIO_OF_BASELINE,
        3.0,
        "A volume spike is either a bug upstream or a burst nobody authorised.",
    ),
    BreakerId.VETO: BreakerSpec(
        BreakerId.VETO,
        "post_allocation_veto_rate",
        Comparison.ABSOLUTE_SHARE,
        0.02,
        "project and certify share one evaluator, so only RUNTIME rules can fire "
        "after allocation. Above two percent the projection is broken.",
        self_monitoring=True,
    ),
    BreakerId.DEGRADED: BreakerSpec(
        BreakerId.DEGRADED,
        "degraded_decision_share",
        Comparison.ABSOLUTE_SHARE,
        0.20,
        "Silent extrapolation past a feature TTL produces confident nonsense.",
        self_monitoring=True,
    ),
    BreakerId.COHORT_BLIND: BreakerSpec(
        BreakerId.COHORT_BLIND,
        "cohort_blindspot_share",
        Comparison.ABSOLUTE_SHARE,
        0.40,
        "Where a cohort detector that could not find power surfaces as a known "
        "limitation instead of passing as a clean NORMAL.",
        self_monitoring=True,
    ),
    BreakerId.CHANNEL_FAIL: BreakerSpec(
        BreakerId.CHANNEL_FAIL,
        "channel_dispatch_failure_rate",
        Comparison.ABSOLUTE_SHARE,
        0.10,
        "A channel failing one dispatch in ten is a provider incident, and "
        "retrying into it makes the incident worse.",
    ),
    BreakerId.RESUME_RAMP: BreakerSpec(
        BreakerId.RESUME_RAMP,
        "post_resume_admission_over_cap",
        Comparison.ABSOLUTE_SHARE,
        1.0,
        "Admission above the ramp's cap is the thundering herd the ramp exists "
        "to prevent, arriving anyway.",
    ),
    BreakerId.SENTIMENT: BreakerSpec(
        BreakerId.SENTIMENT,
        "negative_sentiment_share_on_calls",
        Comparison.RATIO_OF_BASELINE,
        1.5,
        "Distress that has not yet become a complaint is still distress.",
    ),
}

SELF_MONITORING: frozenset[BreakerId] = frozenset(
    spec.breaker_id for spec in SPECS.values() if spec.self_monitoring
)


@dataclass(frozen=True)
class Reading:
    """One metric, with the baseline it is judged against.

    `baseline` is required for a RATIO breaker and ignored by an ABSOLUTE one.
    A ratio breaker with no baseline does NOT trip - see `evaluate` - because
    tripping on an unknown baseline would fire the whole catalogue on the first
    cycle of a fresh deployment.
    """

    breaker_id: BreakerId
    observed: float
    baseline: float | None = None
    sample: int = 0


@dataclass(frozen=True)
class Verdict:
    breaker_id: BreakerId
    tripped: bool
    observed: float
    threshold: float
    detail: str

    @property
    def self_monitoring(self) -> bool:
        return self.breaker_id in SELF_MONITORING


# Below this many observations a rate is noise. A single complaint out of three
# contacts is 333 per thousand and means nothing.
# source: the sample size at which a per-thousand rate stops swinging on one
# event, for the contact volumes a portfolio cycle produces.
MIN_SAMPLE = 30


def evaluate(reading: Reading, *, min_sample: int = MIN_SAMPLE) -> Verdict:
    """One reading against its spec. Pure - no clock, no I/O."""
    spec = SPECS[reading.breaker_id]

    if reading.sample and reading.sample < min_sample:
        return Verdict(
            breaker_id=reading.breaker_id,
            tripped=False,
            observed=reading.observed,
            threshold=float("nan"),
            detail=f"sample {reading.sample} below {min_sample}; rate not yet meaningful",
        )

    if spec.comparison is Comparison.ABSOLUTE_SHARE:
        threshold = spec.threshold
        tripped = reading.observed > threshold
        detail = f"{spec.metric} {reading.observed:.4f} vs cap {threshold:.4f}"
    else:
        if reading.baseline is None or reading.baseline <= 0.0:
            # No baseline is not a trip. A fresh deployment has no trailing
            # median and firing every ratio breaker on cycle one would make the
            # system unusable exactly when it is being watched most closely.
            return Verdict(
                breaker_id=reading.breaker_id,
                tripped=False,
                observed=reading.observed,
                threshold=float("nan"),
                detail=f"{spec.metric} has no baseline yet; not evaluated",
            )
        threshold = spec.threshold * reading.baseline
        tripped = reading.observed > threshold
        detail = (
            f"{spec.metric} {reading.observed:.4f} vs {spec.threshold:.2f}x "
            f"baseline {reading.baseline:.4f} = {threshold:.4f}"
        )

    return Verdict(
        breaker_id=reading.breaker_id,
        tripped=tripped,
        observed=reading.observed,
        threshold=threshold,
        detail=detail,
    )


def evaluate_all(readings: Sequence[Reading], *, min_sample: int = MIN_SAMPLE) -> list[Verdict]:
    """Every reading. NO SHORT-CIRCUIT ON THE FIRST TRIP.

    The same reasoning as the Gate evaluating all rules: an operator looking at
    a tripped system needs to know everything that is wrong, not the first
    thing the loop happened to reach.
    """
    return [evaluate(reading, min_sample=min_sample) for reading in readings]


async def record(conn: Any, verdicts: Sequence[Verdict], *, at: datetime) -> list[BreakerId]:
    """Persist every verdict; return the ones that are newly tripped.

    `tripped_at` is only set on the transition, so a breaker that has been open
    for six hours does not look like it just fired.
    """
    ensure_utc(at)
    newly: list[BreakerId] = []
    for verdict in verdicts:
        was = await conn.fetchval(
            "SELECT tripped FROM breaker_state WHERE breaker_id = $1", verdict.breaker_id.value
        )
        await conn.execute(
            """
            INSERT INTO breaker_state
                (breaker_id, tripped, observed, threshold, evaluated_at, tripped_at,
                 cleared_at, detail)
            VALUES ($1, $2::boolean, $3::double precision, $4::double precision,
                    $5::timestamptz,
                    -- Cast on every branch: a parameter used both as a column
                    -- value and as a CASE result is otherwise ambiguous.
                    CASE WHEN $2::boolean THEN $5::timestamptz END,
                    CASE WHEN $2::boolean THEN NULL::timestamptz ELSE $5::timestamptz END,
                    $6)
            ON CONFLICT (breaker_id) DO UPDATE SET
                tripped    = EXCLUDED.tripped,
                observed   = EXCLUDED.observed,
                threshold  = EXCLUDED.threshold,
                evaluated_at = EXCLUDED.evaluated_at,
                tripped_at = CASE
                    WHEN EXCLUDED.tripped AND NOT breaker_state.tripped THEN EXCLUDED.evaluated_at
                    WHEN EXCLUDED.tripped THEN breaker_state.tripped_at
                    ELSE NULL END,
                cleared_at = CASE
                    WHEN NOT EXCLUDED.tripped AND breaker_state.tripped
                        THEN EXCLUDED.evaluated_at
                    ELSE breaker_state.cleared_at END,
                detail     = EXCLUDED.detail
            """,
            verdict.breaker_id.value,
            verdict.tripped,
            verdict.observed,
            None if verdict.threshold != verdict.threshold else verdict.threshold,
            at,
            verdict.detail,
        )
        if verdict.tripped and not was:
            newly.append(verdict.breaker_id)
    return newly


async def tripped(conn: Any) -> list[BreakerId]:
    rows = await conn.fetch(
        "SELECT breaker_id FROM breaker_state WHERE tripped ORDER BY breaker_id"
    )
    return [BreakerId(r["breaker_id"]) for r in rows]


async def evaluate_and_apply(
    conn: Any,
    readings: Sequence[Reading],
    *,
    at: datetime,
    changed_by: str = "breakers",
    min_sample: int = MIN_SAMPLE,
) -> tuple[list[Verdict], list[BreakerId]]:
    """Evaluate, persist, and trip the system to SHADOW if anything fired.

    SHADOW rather than FREEZE: a breaker has detected a signal, not established
    a catastrophe. Shadow stops the harm while keeping the diagnosis running,
    and a human decides whether it needs to become a freeze.
    """
    from arc.conductor.kill_switch import Mode, current_mode, set_mode

    verdicts = evaluate_all(readings, min_sample=min_sample)
    newly = await record(conn, verdicts, at=at)

    if newly:
        state = await current_mode(conn)
        if state.mode is Mode.NORMAL:
            await set_mode(
                conn,
                Mode.SHADOW,
                at=at,
                changed_by=changed_by,
                reason="tripped: " + ", ".join(b.value for b in newly),
            )
    return verdicts, newly


def render(verdicts: Sequence[Verdict]) -> list[str]:
    """Readable output. Self-monitoring breakers are marked as such.

    The distinction matters to whoever reads it: CB-COMPLAINT firing is the
    system harming people, CB-VETO firing is the system failing to work.
    """
    lines = []
    for verdict in sorted(verdicts, key=lambda v: v.breaker_id.value):
        flag = "TRIPPED" if verdict.tripped else "ok"
        kind = "self" if verdict.self_monitoring else "harm"
        lines.append(f"  [{kind}] {verdict.breaker_id.value:<18} {flag:<8} {verdict.detail}")
    return lines
