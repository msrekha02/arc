"""Training records, and the guard on the propensity.

THE LOGGED PROPENSITY IS NOT OPTIONAL AND IS NOT ESTIMATED. Most industrial
uplift work has to fit a propensity model, and mis-estimating it is the
dominant bias source in the whole literature. ARC does not have that problem:
the Allocator samples from a softmax with an epsilon floor and writes down the
exact probability it drew with. That number is a fact about the decision, not
an inference about it, and it is what makes one leg of the doubly-robust
estimator at M11 correct by construction.

So this module refuses to help anyone throw that away. `require_propensities`
raises on a row whose propensity was not recorded, and the X-learner calls it
before it fits anything. There is no fallback that estimates `g(x)` from the
data - a fallback would be used, and the moment it is used the guarantee is
gone with no visible symptom.

The epsilon floor also bounds the importance weights. With every eligible
action drawn at probability at least `eps / n`, the ratio `pi / pi_b` cannot
blow up, which is the other half of why the estimator behaves.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from arc.core.money import Paise, paise
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType


class LoggedPropensityMissing(ValueError):
    """A decision reached training without the probability it was drawn with."""


class InsufficientTreatedUnits(ValueError):
    """An arm has too few units to fit anything worth trusting."""


@dataclass(frozen=True)
class LoggedDecision:
    """One (state, action, outcome) triple as the ledger recorded it.

    `features` is the vector as it stood AT DECISION TIME, not as it can be
    reconstructed later. Recomputing features after the fact leaks the outcome
    back into the input - the contact that was made, the adverse event that
    followed - and the model then scores its own consequences as causes.
    """

    account_id: str
    at: datetime
    action: ActionType
    propensity: float
    features: tuple[float, ...]
    paid: bool
    amount_paise: Paise = paise(0)
    adverse: bool = False
    opted_out: bool = False
    complained: bool = False
    promised: bool = False
    subject_token: str | None = None

    def __post_init__(self) -> None:
        ensure_utc(self.at)
        if self.propensity is None or not np.isfinite(self.propensity):
            raise LoggedPropensityMissing(
                f"{self.account_id} at {self.at.isoformat()} carries no usable propensity"
            )
        if not 0.0 < self.propensity <= 1.0:
            raise LoggedPropensityMissing(
                f"propensity {self.propensity} is outside (0, 1]; a zero-probability "
                "action cannot have been sampled, so the log is wrong"
            )

    @property
    def reward(self) -> float:
        """Binary payment. The rupee weighting belongs to the Allocator's
        objective, not to the outcome model, which would otherwise learn plan
        value twice."""
        return 1.0 if self.paid else 0.0


def require_propensities(rows: Sequence[LoggedDecision]) -> np.ndarray:
    """Every row's recorded propensity, or a refusal to proceed.

    Deliberately has no `estimate=True` path. A guard with a bypass is not a
    guard, and this one protects the only leg of the DR estimator that is
    guaranteed rather than assumed.
    """
    if not rows:
        raise LoggedPropensityMissing("no logged decisions; nothing to train on")
    values = np.array([row.propensity for row in rows], dtype=float)
    bad = ~np.isfinite(values) | (values <= 0.0) | (values > 1.0)
    if bad.any():
        first = int(np.argmax(bad))
        raise LoggedPropensityMissing(
            f"{int(bad.sum())} of {len(rows)} rows lack a usable logged propensity "
            f"(first: {rows[first].account_id}); the X-learner will not estimate one"
        )
    return values


@runtime_checkable
class HasFeatures(Protocol):
    """Anything carrying an account id and a fixed-width feature vector.

    Structural so the bounce observations, the logged decisions and the
    promise records share one splitter and one matrix builder. Three copies of
    an account-level split is three chances to leak a row across it.
    """

    account_id: str
    features: tuple[float, ...]


def feature_matrix(rows: Sequence[HasFeatures]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    width = len(rows[0].features)
    for row in rows:
        if len(row.features) != width:
            raise ValueError(
                f"{row.account_id} has {len(row.features)} features, expected {width}; "
                "the column order is part of the contract"
            )
    return np.array([row.features for row in rows], dtype=float)


def reward_vector(rows: Sequence[LoggedDecision]) -> np.ndarray:
    return np.array([row.reward for row in rows], dtype=float)


def rows_for(rows: Iterable[LoggedDecision], action: ActionType) -> list[LoggedDecision]:
    return [row for row in rows if row.action is action]


def action_counts(rows: Iterable[LoggedDecision]) -> dict[ActionType, int]:
    counts: dict[ActionType, int] = {}
    for row in rows:
        counts[row.action] = counts.get(row.action, 0) + 1
    return counts


Row = TypeVar("Row", bound=HasFeatures)


def split_by_account(rows: Sequence[Row], holdout: float, seed: int) -> tuple[list[Row], list[Row]]:
    """Split on the ACCOUNT, never on the row.

    An account contributes several cycles. Splitting rows would put the same
    account's cycle 2 in training and its cycle 5 in evaluation, and the
    evaluation would then be measuring memorisation of an account rather than
    generalisation to a new one.
    """
    if not 0.0 < holdout < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    accounts = sorted({row.account_id for row in rows})
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(accounts))
    cut = int(round(len(accounts) * (1.0 - holdout)))
    train_accounts = {accounts[index] for index in order[:cut]}
    train = [row for row in rows if row.account_id in train_accounts]
    test = [row for row in rows if row.account_id not in train_accounts]
    if not train or not test:
        raise ValueError(
            f"an account-level {holdout:.0%} split of {len(accounts)} accounts left one "
            "side empty; too few distinct accounts to hold anything out"
        )
    return train, test


# ---------------------------------------------------------------------------
# Promises
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PromiseRecord:
    """One promise to pay, and how much of its life we actually observed.

    `observed_until` is what separates a censored promise from a broken one. A
    promise dated the 20th, looked at on the 18th, is neither kept nor broken -
    it is unresolved, and the record says so by ending early rather than by
    carrying a `False`.
    """

    account_id: str
    features: tuple[float, ...]
    made_at: datetime
    due_at: datetime
    observed_until: datetime
    kept_at: datetime | None = None
    grace_hours: int = 24
    selection_propensity: float | None = None

    def __post_init__(self) -> None:
        for moment in (self.made_at, self.due_at, self.observed_until):
            ensure_utc(moment)
        if self.kept_at is not None:
            ensure_utc(self.kept_at)
        if self.due_at < self.made_at:
            raise ValueError("a promise cannot fall due before it was made")

    @property
    def horizon_days(self) -> int:
        """Whole days from promise to due date, at least one."""
        return max(int(round((self.due_at - self.made_at).total_seconds() / 86400.0)), 1)

    @property
    def deadline(self) -> datetime:
        return self.due_at + timedelta(hours=self.grace_hours)

    @property
    def kept(self) -> bool:
        return self.kept_at is not None

    @property
    def censored(self) -> bool:
        """Unresolved: not kept, and the deadline has not been reached.

        THE ONE THING THIS MODULE EXISTS TO GET RIGHT. Coding these as broken
        biases the model pessimistic in exactly the population it is used on,
        because the promises still in flight are disproportionately the recent
        ones and the recent ones are disproportionately the ones about to be
        kept.
        """
        return not self.kept and self.observed_until < self.deadline

    @property
    def broken(self) -> bool:
        return not self.kept and not self.censored

    @property
    def observed_days(self) -> int:
        """Discrete periods actually observed, for the person-period expansion."""
        end = min(self.observed_until, self.deadline)
        elapsed = int((end - self.made_at).total_seconds() // 86400)
        return max(min(elapsed, self.horizon_days), 0 if self.censored else 1)
