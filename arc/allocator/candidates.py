"""Candidate construction: what the optimiser is allowed to choose from.

Three exclusions happen here, and each one is doing different work.

CONTROL SUBJECTS ARE REMOVED FROM THE POOL ENTIRELY, not merely left
untreated. A control subject that still contended for a shared contact budget
would be affecting the treated subjects' allocation - starving them, or being
starved by them - and that is itself a treatment effect. The arm would then be
measuring "untreated, in a portfolio that had to route around them", which is
not the counterfactual anyone wants. Removing them makes the control arm what
it claims to be.

ELIGIBILITY COMES FROM `gate.project()`, EVALUATED HERE AND PRUNED BEFORE THE
OPTIMISATION. Not filtered afterwards, because an action vetoed after the
allocator sampled it contaminates the logged propensity: the probability
written down would be the probability of an action that could never have run.
There is no compliance rule in this module - not a cooldown, not a consent
check, not a contact window. This file asks the Gate and believes the answer.
`tests/test_allocator.py` walks the AST of `arc/allocator/` and fails the build
if any rule evaluation appears here, which is the reciprocal of M3's assertion
that `project` and `certify` share one evaluator.

`do_nothing` IS ALWAYS PRESENT, once per subject, at value zero and cost zero.
Without it the knapsack has no decline-to-act option: every subject would
receive the least-bad available action, and a portfolio of least-bad actions
spends real budget on negative expected value. It sits at the subject level
rather than per claim because doing nothing is not a fact about a claim, and
three copies of it would triple its softmax mass for no reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import log1p
from typing import Protocol, runtime_checkable
from uuid import UUID

from arc.allocator.budgets import ACTION_COST, ZERO_COST, CostVector, cost_of
from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType, Claim
from arc.gate.context import GateContext
from arc.proving_ground.arms import CONTROL_ARMS, Arm

# The closed action space, in a fixed order so a projection mask and a
# candidate list can be compared without sorting surprises.
ALL_ACTIONS: tuple[ActionType, ...] = tuple(ActionType)

# ---------------------------------------------------------------------------
# Objective weights.
#
# source: subscription and lending economics, where the lifetime value of a
# retained payer is a large multiple of any single failed instalment. The
# weight is logarithmic rather than linear so that a customer worth twenty
# times the failed charge is prioritised over one worth two, without a single
# very high-value account consuming the entire cycle's budget.
# ---------------------------------------------------------------------------
LTV_SENSITIVITY = 0.35

# The marginal probability that ONE action of this kind, at ordinary contact
# pressure, ends the relationship. Silent actions reach nobody and risk
# nothing; a statutory notice is the most aggressive thing in the ladder.
#
# source: subscription churn reporting for involuntary-failure cohorts, where
# a failed charge followed by pressure is a leading voluntary-cancellation
# cause, and collections research on contact frequency and disengagement.
# These are the ALLOCATOR's own estimates of harm, deliberately independent of
# the simulator's response model, which this package cannot see.
#
# THE LEVELS ARE LOAD-BEARING, not decoration. The term is multiplied by
# REMAINING LIFETIME VALUE, which is large, so an intrusion figure that is too
# high prices every contact action out of the portfolio entirely and the
# system silently becomes retry-only. Measured at these values a utility
# message on a fifteen-thousand-rupee relationship costs about thirty-seven
# rupees of expected churn against a gain in the low hundreds, which is a
# trade the optimiser can actually make.
ACTION_INTRUSION: Mapping[ActionType, float] = {
    ActionType.DO_NOTHING: 0.0,
    ActionType.RETRY: 0.0,
    ActionType.CARD_UPDATER: 0.0,
    ActionType.MANDATE_RE_REGISTER: 0.0,
    ActionType.RAIL_FALLBACK: 0.0,
    ActionType.EMAIL: 0.0008,
    ActionType.SMS: 0.0012,
    ActionType.WHATSAPP_UTILITY: 0.0010,
    ActionType.PAYMENT_LINK: 0.0009,
    ActionType.VOICE_CALL: 0.0045,
    ActionType.INSTALMENT_OFFER: 0.0030,
    ActionType.HUMAN_HANDOFF: 0.0050,
    ActionType.STATUTORY_NOTICE: 0.0150,
}

# Each contact already made this week raises the churn risk of the next one.
# Superlinear pressure is what makes the contact budget worth spending
# carefully rather than exhausting.
PRESSURE_EXPONENT = 1.3


class DropReason(StrEnum):
    """Why a claim is not in the pool. Every exclusion is logged with one."""

    CONTROL_ARM = "control_arm"
    GATE_INELIGIBLE = "gate_ineligible"
    NO_ELIGIBLE_ACTION = "no_eligible_action"
    INFEASIBLE_SHRINK = "infeasible_shrink"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class Drop:
    """One exclusion, kept so the treated set can be reconciled against the batch."""

    subject_token: str
    claim_id: UUID | None
    reason: DropReason
    detail: str = ""


@runtime_checkable
class EligibilitySource(Protocol):
    """The Gate, as the allocator is allowed to see it.

    Narrowed to `project` on purpose. The allocator can ask what is eligible;
    it has no way to ask for authorisation, because authorisation is not its
    to give (GI-6).
    """

    def project(
        self, ctx: GateContext, actions: Sequence[ActionType], at: datetime
    ) -> set[ActionType]: ...


@runtime_checkable
class UpliftSource(Protocol):
    """M7's forecaster, as the allocator uses it.

    Structural so the allocator can be exercised against a planted effect
    surface without training three models, and so a later model swap needs no
    change here.
    """

    def uplift(
        self, observation: object, action: ActionType, ctx: object, *, propensity: float
    ): ...


@dataclass(frozen=True)
class ClaimView:
    """One claim with everything the allocator needs to score it."""

    claim: Claim
    gate_ctx: GateContext
    observation: object = None
    feature_ctx: object = None
    contacts_7d: int = 0


@dataclass(frozen=True)
class SubjectPortfolio:
    """Every claim held by one person, and the arm they were assigned.

    THE SUBJECT IS THE UNIT. It is the unit of budget contention, the unit of
    contact-frequency accounting, the unit a message physically reaches, and
    therefore the unit at which experimental interference occurs. Three claims
    against one person compete here and two of them lose.
    """

    subject_token: str
    arm: Arm
    claims: tuple[ClaimView, ...]
    ltv_remaining_paise: Paise | None = None

    def __post_init__(self) -> None:
        if not self.claims:
            raise ValueError(f"{self.subject_token} holds no claims and is not a portfolio")


@dataclass(frozen=True)
class Candidate:
    """One (subject, claim, action) the optimiser may choose.

    `value` is an expected incremental recovery in paise-scale, net of direct
    cost and expected churn. It is a float and it is NOT money: it is an
    expectation over an outcome that has not happened, and GI-2 governs
    amounts rather than expectations of them.
    """

    subject_token: str
    claim_id: UUID | None
    action: ActionType
    value: float
    cost: CostVector
    uplift: float = 0.0
    is_silent: bool = True

    @property
    def is_do_nothing(self) -> bool:
        return self.action is ActionType.DO_NOTHING


@dataclass
class CandidatePool:
    """The pool, grouped by subject, with the exclusions that produced it."""

    candidates: list[Candidate] = field(default_factory=list)
    drops: list[Drop] = field(default_factory=list)

    def by_subject(self) -> dict[str, list[Candidate]]:
        grouped: dict[str, list[Candidate]] = {}
        for candidate in self.candidates:
            grouped.setdefault(candidate.subject_token, []).append(candidate)
        return grouped

    @property
    def subjects(self) -> int:
        return len({candidate.subject_token for candidate in self.candidates})


def ltv_weight(claim: Claim) -> float:
    """How much more a recovery is worth than the rupees it recovers.

    Logarithmic in the ratio of remaining lifetime value to the failed amount.
    A relationship worth twenty times the charge outranks one worth two, and no
    single account can dominate a cycle by being enormous.
    """
    amount = max(int(claim.amount_paise), 1)
    ratio = int(claim.ltv_remaining_paise) / amount
    return 1.0 + LTV_SENSITIVITY * log1p(max(ratio, 0.0))


def annoyance_cost(claim: Claim, action: ActionType, contacts_7d: int) -> float:
    """Expected lifetime value destroyed by being this intrusive, right now.

    WHY THE OBJECTIVE CARRIES IT AT ALL: without this term the optimiser
    maximises thirty-day recovery and buys it with churn it never sees. The
    cost scales with the LIFETIME VALUE rather than the failed amount, which is
    what stops the system winning a small charge and losing a large customer.
    """
    intrusion = ACTION_INTRUSION[action]
    if intrusion == 0.0:
        return 0.0
    pressure = float(max(contacts_7d, 0) + 1) ** PRESSURE_EXPONENT
    return intrusion * pressure * float(int(claim.ltv_remaining_paise))


def candidate_value(
    claim: Claim,
    action: ActionType,
    uplift: float,
    contacts_7d: int,
) -> float:
    """v_ia = tau * amount * ltv_weight - direct_cost - expected_annoyance.

    `uplift` is SIGNED. A negative tau makes the first term negative, and the
    two cost terms only subtract further, so an action that hurts can never
    outscore `do_nothing` at zero. That is the whole sleeping-dog rule: it is
    arithmetic here, not a threshold somewhere else.
    """
    gain = uplift * float(int(claim.amount_paise)) * ltv_weight(claim)
    direct = float(int(cost_of(action).rupee_paise))
    return gain - direct - annoyance_cost(claim, action, contacts_7d)


def build_candidates(
    portfolio: Sequence[SubjectPortfolio],
    gate: EligibilitySource,
    forecaster: UpliftSource,
    at: datetime,
    *,
    behaviour_propensity: float = 0.05,
) -> CandidatePool:
    """The pool the optimiser sees.

    `at` is the PLANNED EXECUTION TIME, and the Gate is asked about that moment
    rather than about now. A temporal rule evaluated at wall-clock gating time
    would veto an action whose propensity had already been logged for a
    different moment, which is the contamination that certificate windows and
    pinned times exist to prevent.
    """
    ensure_utc(at)
    pool = CandidatePool()

    for subject in portfolio:
        if subject.arm in CONTROL_ARMS:
            # Excluded from the POOL, not merely left untreated.
            pool.drops.append(
                Drop(
                    subject_token=subject.subject_token,
                    claim_id=None,
                    reason=DropReason.CONTROL_ARM,
                    detail=f"arm={subject.arm}",
                )
            )
            continue

        produced = 0
        for view in subject.claims:
            eligible = gate.project(view.gate_ctx, ALL_ACTIONS, at)
            actionable = sorted(eligible - {ActionType.DO_NOTHING})
            if not actionable:
                pool.drops.append(
                    Drop(
                        subject_token=subject.subject_token,
                        claim_id=view.claim.claim_id,
                        reason=DropReason.GATE_INELIGIBLE,
                        detail="no action survived projection",
                    )
                )
                continue

            for action in actionable:
                estimate = forecaster.uplift(
                    view.observation,
                    action,
                    view.feature_ctx,
                    propensity=behaviour_propensity,
                )
                tau = float(getattr(estimate, "value", estimate))
                pool.candidates.append(
                    Candidate(
                        subject_token=subject.subject_token,
                        claim_id=view.claim.claim_id,
                        action=action,
                        value=candidate_value(view.claim, action, tau, view.contacts_7d),
                        cost=cost_of(action),
                        uplift=tau,
                        is_silent=ACTION_COST[action].contact == 0,
                    )
                )
                produced += 1

        # Always, and exactly once per subject.
        pool.candidates.append(
            Candidate(
                subject_token=subject.subject_token,
                claim_id=None,
                action=ActionType.DO_NOTHING,
                value=0.0,
                cost=ZERO_COST,
                uplift=0.0,
                is_silent=True,
            )
        )
        if produced == 0:
            pool.drops.append(
                Drop(
                    subject_token=subject.subject_token,
                    claim_id=None,
                    reason=DropReason.NO_ELIGIBLE_ACTION,
                    detail="only do_nothing remained",
                )
            )

    return pool
