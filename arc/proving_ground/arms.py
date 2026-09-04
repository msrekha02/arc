"""Experiment arms, assigned at the SUBJECT level (GI-8).

THE UNIT OF RANDOMISATION IS THE UNIT OF INTERFERENCE, and interference here is
at the subject. A person with one claim in control and another in treatment
breaks SUTVA twice over: the portfolio-level allocation can starve the control
claim of a budget the treated claim consumed, and a WhatsApp about invoice A
reminds them about invoice B. Neither effect is small and neither is
detectable after the fact - the estimate is simply wrong, with no symptom.

So every claim inherits its subject's arm, with no exceptions, and the
assignment is made once and persisted. `assign_arm` is a pure function of the
token, the experiment and the stratum; `ArmRegistry` is what makes the FIRST
answer the only answer, because a subject's claim count grows as claims arrive
and a stratum recomputed later would otherwise move them between arms.

Stratification is on claim-count bucket, value decile and rail. WHY those
three: subjects holding one claim and subjects holding five behave differently
and would otherwise imbalance the arms, widening every interval; value decile
because recovery rates and rupee weights both track it; rail because the
mechanics of recovery differ per rail more than anything else about a claim.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from arc.core.ids import is_subject_token
from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.core.types import Rail


class Arm(StrEnum):
    """The five arms. Closed, and named the same way M11 reports them.

    NULL measures what arrives anyway. NAIVE_DUNNING is the bar to beat and the
    headline's comparator. GATEWAY_DEFAULT is the incumbent. GREEDY is what
    happens without constraints, and beating it on net value rather than gross
    recovery is the result worth having. ARC is the policy under test.
    """

    NULL = "null"
    NAIVE_DUNNING = "naive_dunning"
    GATEWAY_DEFAULT = "gateway_default"
    GREEDY_UNCONSTRAINED = "greedy_unconstrained"
    ARC = "arc"


ARMS: tuple[Arm, ...] = tuple(Arm)

# Equal shares. Unequal ones buy power for the arm you already believe in,
# which is the wrong instinct when the point is to be falsifiable.
ARM_SHARES: tuple[float, ...] = tuple(1.0 / len(ARMS) for _ in ARMS)

# Arms that receive no treatment at all. They are excluded from the allocation
# pool entirely at M8 rather than merely left untreated: a control subject that
# still contends for a shared budget has been treated, just invisibly.
CONTROL_ARMS: frozenset[Arm] = frozenset({Arm.NULL})

CLAIM_COUNT_BUCKETS: tuple[str, ...] = ("1", "2_3", "4_plus")
VALUE_DECILES = 10


def claim_count_bucket(claim_count: int) -> str:
    """One, a few, or many. Three buckets because ten would leave most empty."""
    if claim_count < 1:
        raise ValueError(f"a subject with {claim_count} claims is not a subject")
    if claim_count == 1:
        return "1"
    if claim_count <= 3:
        return "2_3"
    return "4_plus"


def value_decile(total_paise: Paise, cutoffs: tuple[Paise, ...]) -> int:
    """Which decile of portfolio value this subject falls in, 0 through 9.

    Cutoffs are passed in rather than computed here, because they are a
    property of the batch and this function must stay pure - the same subject
    with the same cutoffs must land in the same decile on every replay.
    """
    if len(cutoffs) != VALUE_DECILES - 1:
        raise ValueError(f"expected {VALUE_DECILES - 1} cutoffs, got {len(cutoffs)}")
    decile = 0
    for cutoff in cutoffs:
        if total_paise < cutoff:
            break
        decile += 1
    return min(decile, VALUE_DECILES - 1)


def decile_cutoffs(values: list[Paise]) -> tuple[Paise, ...]:
    """The nine boundaries of the value deciles for one batch."""
    if not values:
        return tuple(Paise(0) for _ in range(VALUE_DECILES - 1))
    ordered = sorted(values)
    return tuple(
        ordered[min(len(ordered) - 1, (index * len(ordered)) // VALUE_DECILES)]
        for index in range(1, VALUE_DECILES)
    )


@dataclass(frozen=True)
class Strata:
    """The stratum a subject is randomised within."""

    claim_count_bucket: str
    value_decile: int
    rail: Rail

    def __post_init__(self) -> None:
        if self.claim_count_bucket not in CLAIM_COUNT_BUCKETS:
            raise ValueError(f"unknown claim-count bucket {self.claim_count_bucket!r}")
        if not 0 <= self.value_decile < VALUE_DECILES:
            raise ValueError(f"value decile {self.value_decile} outside [0, 10)")

    @property
    def key(self) -> str:
        return f"{self.claim_count_bucket}|{self.value_decile}|{self.rail}"


def assign_arm(subject_token: str, experiment_id: str, strata: Strata) -> Arm:
    """Deterministic, subject-level, stratified assignment. PURE.

    The stratum is inside the hash, so each stratum draws independently and the
    arms balance within it rather than only across the batch. That is what
    stratification buys: without it, a stratum holding forty subjects can land
    thirty of them in one arm by chance and widen every interval that uses it.

    Nothing about the claim enters here. There is no claim id in the signature
    and there is nowhere to put one.
    """
    if not is_subject_token(subject_token):
        raise ValueError(f"{subject_token!r} is not a derived subject token")
    if not experiment_id:
        raise ValueError("experiment_id is required; an unnamed experiment cannot be replayed")

    material = f"{experiment_id}\x1f{strata.key}\x1f{subject_token}".encode()
    draw = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64

    cumulative = 0.0
    for arm, share in zip(ARMS, ARM_SHARES, strict=True):
        cumulative += share
        if draw < cumulative:
            return arm
    return ARMS[-1]


# ---------------------------------------------------------------------------
# Persistence. Everything above this line is pure; everything below makes the
# FIRST assignment the ONLY assignment.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmAssignment:
    subject_token: str
    arm: Arm
    strata: Strata
    assigned_at: datetime
    newly_assigned: bool


class ArmRegistry:
    """First assignment wins, forever.

    WHY persistence is load-bearing rather than a cache: a subject's stratum
    moves as their claims arrive - one claim today, a fourth next week - and
    recomputing the assignment would move them between arms mid-experiment.
    A subject who was control in week one and treated in week two is in
    neither arm, and every claim they hold is contaminated.
    """

    def __init__(self, experiment_id: str) -> None:
        if not experiment_id:
            raise ValueError("an experiment needs a name to be replayable")
        self.experiment_id = experiment_id

    async def assign_once(
        self, conn: Any, subject_token: str, strata: Strata, at: datetime
    ) -> ArmAssignment:
        """Return the stored arm, or store and return a freshly assigned one.

        The insert is the claim on the subject. Two workers racing the same new
        subject both compute the same arm anyway, because assignment is a pure
        function - but only one row is written, and it is the one everything
        reads afterwards.
        """
        ensure_utc(at)
        arm = assign_arm(subject_token, self.experiment_id, strata)

        row = await conn.fetchrow(
            """
            INSERT INTO subject_arms
                (subject_token, experiment_id, arm, claim_count_bucket,
                 value_decile, rail, assigned_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (subject_token) DO NOTHING
            RETURNING arm, claim_count_bucket, value_decile, rail, assigned_at
            """,
            subject_token,
            self.experiment_id,
            arm.value,
            strata.claim_count_bucket,
            strata.value_decile,
            strata.rail.value,
            at,
        )
        if row is not None:
            return ArmAssignment(subject_token, arm, strata, at, newly_assigned=True)

        stored = await conn.fetchrow(
            """
            SELECT arm, claim_count_bucket, value_decile, rail, assigned_at
              FROM subject_arms WHERE subject_token = $1
            """,
            subject_token,
        )
        return ArmAssignment(
            subject_token=subject_token,
            arm=Arm(stored["arm"]),
            strata=Strata(
                claim_count_bucket=stored["claim_count_bucket"],
                value_decile=int(stored["value_decile"]),
                rail=Rail(stored["rail"]),
            ),
            assigned_at=stored["assigned_at"],
            newly_assigned=False,
        )

    async def arm_of(self, conn: Any, subject_token: str) -> Arm | None:
        stored = await conn.fetchval(
            "SELECT arm FROM subject_arms WHERE subject_token = $1", subject_token
        )
        return None if stored is None else Arm(stored)
