"""One allocation cycle, end to end.

Build the pool, price the budgets, sample one action per subject, and record
what was decided together with the probability it was decided with.

BOTH TIMES ARE PINNED HERE. `decision_time` is when the portfolio was scored;
`planned_execution_time` is when the action is meant to run, and it is the
moment the Gate was asked about. Pinning them is what keeps the propensity
honest: if the Gate were consulted at wall-clock gating time instead, a clock
tick between allocation and certification could veto an action whose
probability had already been written down, and the logged number would then
describe an action that could never have executed.

WHAT THIS MODULE DOES NOT DO. It does not commit anything. `certify` is
invoked because the certificate is part of the decision record, but writing the
FSM transition, reserving the budget, appending to the ledger and inserting the
outbox row are one transaction owned by the Conductor at M9. The allocator
proposes; it does not act.

ONE ACTION PER SUBJECT PER CYCLE. Three claims against one person compete in
the same softmax and two of them lose. That is where the shared contact budget
is actually enforced, and it is the operational consequence of arms being
assigned at the subject level: a rule that randomised per claim would have
nothing to enforce here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import numpy as np

from arc.allocator.budgets import PRICED_BUDGETS, BudgetKey, Budgets, Spend
from arc.allocator.candidates import (
    Candidate,
    CandidatePool,
    Drop,
    DropReason,
    EligibilitySource,
    SubjectPortfolio,
    UpliftSource,
    build_candidates,
)
from arc.allocator.lagrangian import BudgetRelaxed, Solution, solve
from arc.allocator.policy import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    adjusted_values,
    explore_mass,
    explore_spend,
    propensity_distribution,
    sleeping_dog,
    stochastic_policy,
)
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType


@dataclass(frozen=True)
class Decision:
    """What the allocator decided for one subject, and how sure it was.

    `pi_intended` is the probability of the action the allocator SAMPLED. The
    realized action and its propensity under the composed allocator-and-Gate
    policy are added downstream, because the Gate can still refuse at
    certification and the estimator must condition on what actually happened.
    """

    subject_token: str
    claim_id: UUID | None
    intended_action: ActionType
    pi_intended: float
    decision_time: datetime
    planned_execution_time: datetime
    shadow_prices: Mapping[BudgetKey, float]
    adjusted_value: float
    uplift: float
    # Keyed by the DECISION, which is (claim, action) and not an action
    # alone: a subject holding three claims can be offered the same action
    # three times, and keying by action would silently merge them and lose
    # most of the probability mass.
    propensities: Mapping[tuple[UUID | None, ActionType], float]
    sleeping_dog: bool = False
    sampled_action: ActionType | None = None
    budget_deferred: bool = False
    # Budget headroom this subject's draw actually faced, in PRICED_BUDGETS
    # order, at ITS OWN POSITION in the admission order.
    #
    # RECORDED BECAUSE THE PROPENSITY DEPENDS ON IT. Admission admits a draw
    # exactly when its cost fits here, so the set of branches that COULD have
    # been admitted is a function of this vector - and M11's composed
    # behaviour policy is wrong without it. M11 used to reconstruct it by
    # replaying the admission order from the decision record, which worked but
    # duplicated the ordering rule outside the module that owns it.
    residual_capacity: tuple[int, ...] = ()

    @property
    def action_marginals(self) -> dict[ActionType, float]:
        """Mass per action, summed across the claims offering it.

        The readable view for the console and the replay trace. The estimator
        wants `propensities`, which is keyed by the decision itself.
        """
        marginals: dict[ActionType, float] = {}
        for (_, action), probability in self.propensities.items():
            marginals[action] = marginals.get(action, 0.0) + probability
        return marginals

    def __post_init__(self) -> None:
        ensure_utc(self.decision_time)
        ensure_utc(self.planned_execution_time)
        if not 0.0 < self.pi_intended <= 1.0:
            raise ValueError(
                f"pi_intended {self.pi_intended} is outside (0, 1]; an action that could "
                "not have been sampled cannot have been sampled"
            )


@dataclass
class Allocation:
    """The cycle's output."""

    decisions: list[Decision] = field(default_factory=list)
    shadow_prices: Mapping[BudgetKey, float] = field(default_factory=dict)
    planned_spend: Spend | None = None
    sampled_spend: Spend | None = None
    drops: list[Drop] = field(default_factory=list)
    pool: CandidatePool | None = None
    solution: Solution | None = None
    # B_explore, reported as two numbers because they answer different
    # questions: what the epsilon floor COST, and how often it fired.
    explore_value_foregone: float = 0.0
    explore_mass_share: float = 0.0

    @property
    def treated(self) -> list[Decision]:
        return [d for d in self.decisions if d.intended_action is not ActionType.DO_NOTHING]

    def subjects(self) -> set[str]:
        return {decision.subject_token for decision in self.decisions}


def allocate(
    portfolio: Sequence[SubjectPortfolio],
    gate: EligibilitySource,
    forecaster: UpliftSource,
    budgets: Budgets,
    rng: np.random.Generator,
    *,
    decision_time: datetime,
    planned_execution_time: datetime | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> Allocation:
    """Score, price, sample. One action per subject, each with its propensity."""
    ensure_utc(decision_time)
    planned_at = planned_execution_time or decision_time
    ensure_utc(planned_at)

    pool = build_candidates(portfolio, gate, forecaster, planned_at)
    if not pool.candidates:
        return Allocation(drops=list(pool.drops), pool=pool)

    solution = solve(pool.candidates, budgets)

    allocation = Allocation(
        shadow_prices=solution.shadow_prices,
        planned_spend=solution.spend,
        drops=[*pool.drops, *solution.drops],
        pool=pool,
        solution=solution,
    )

    # Candidates the shrink step removed must not be sampled either, or the
    # policy would hand back mass to an action the optimiser already refused.
    dropped = {(drop.subject_token, drop.claim_id) for drop in solution.drops}
    grouped = pool.by_subject()
    explore_total = 0.0
    explore_mass_total = 0.0
    sampled: list[_Sampled] = []

    for subject_token in sorted(grouped):
        candidates = [
            candidate
            for candidate in grouped[subject_token]
            if (candidate.subject_token, candidate.claim_id) not in dropped
            or candidate.is_do_nothing
        ]
        chosen, propensity = stochastic_policy(
            candidates,
            solution.shadow_prices,
            rng,
            temperature=temperature,
            epsilon=epsilon,
        )
        distribution = propensity_distribution(
            candidates, solution.shadow_prices, temperature=temperature, epsilon=epsilon
        )
        explore_total += explore_spend(
            candidates, solution.shadow_prices, temperature=temperature, epsilon=epsilon
        )
        explore_mass_total += explore_mass(
            candidates, solution.shadow_prices, temperature=temperature, epsilon=epsilon
        )
        adjusted = adjusted_values(candidates, solution.shadow_prices)
        position = candidates.index(chosen)

        sampled.append(
            _Sampled(
                subject_token=subject_token,
                candidate=chosen,
                propensity=propensity,
                adjusted_value=float(adjusted[position]),
                propensities={
                    (candidate.claim_id, candidate.action): float(probability)
                    for candidate, probability in zip(candidates, distribution, strict=True)
                },
                sleeping_dog=sleeping_dog(candidates),
                fallback=_do_nothing_of(candidates, subject_token),
            )
        )

    allocation.decisions, admitted, deferrals = _admit(
        sampled,
        budgets,
        decision_time=decision_time,
        planned_at=planned_at,
        shadow_prices=dict(solution.shadow_prices),
    )
    allocation.drops.extend(deferrals)
    allocation.sampled_spend = _spend_of(admitted)
    subjects = max(len(allocation.decisions), 1)
    allocation.explore_value_foregone = explore_total / subjects
    allocation.explore_mass_share = explore_mass_total / subjects

    overruns = allocation.sampled_spend.overruns(budgets)
    if overruns:
        raise BudgetRelaxed(
            f"the sampled allocation exceeds its caps by {overruns}; admission "
            "must defer the marginal subject, never widen the budget"
        )
    return allocation


# Stand-in for an uncapped dimension in `residual_capacity`. A tuple of ints
# keeps the field cheap to store and compare; `inf` would make it a float
# vector and invite float arithmetic into a budget comparison.
_UNBOUNDED = 2**62


@dataclass(frozen=True)
class _Sampled:
    """One subject's draw, before admission decides whether it fits."""

    subject_token: str
    candidate: Candidate
    propensity: float
    adjusted_value: float
    propensities: Mapping[tuple[UUID | None, ActionType], float]
    sleeping_dog: bool
    fallback: Candidate


def _admit(
    sampled: Sequence[_Sampled],
    budgets: Budgets,
    *,
    decision_time: datetime,
    planned_at: datetime,
    shadow_prices: Mapping[BudgetKey, float],
) -> tuple[list[Decision], list[Candidate], list[Drop]]:
    """Fit the sampled draws inside the caps, most valuable first.

    WHY THIS EXISTS AT ALL. The solver prices the budgets against each
    subject's BEST action, but the policy deliberately does not always take the
    best one - that is the whole point of the epsilon floor. So a realisation
    can consume a little more than the plan did, and a cap that held for the
    plan can be breached by the draw.

    The response is the same one the infeasibility rule uses: SHRINK THE
    TREATED SET. Subjects are admitted in descending adjusted value, so if a
    dimension runs out it is the least valuable draws that fall back to
    `do_nothing`, each logged with a reason. No cap moves.

    THIS IS NOT THE RESERVATION. Durable, atomic, cross-cycle reservation is
    the Conductor's at M9, and check-then-act inside one process is not it.
    This is the in-cycle guard that keeps the allocator's own output legal, so
    that M9 receives a plan that already fits rather than one it has to repair.

    The logged propensity is untouched. It remains the probability the
    allocator sampled with, which is what M11 needs as pi_alloc; a subject
    deferred here has its intended action collapse onto `do_nothing` in exactly
    the way a Gate veto does, and the composed policy accounts for it there.
    """
    order = sorted(range(len(sampled)), key=lambda i: -sampled[i].adjusted_value)
    running = [0] * len(PRICED_BUDGETS)
    caps = budgets.as_vector()

    outcome: dict[str, tuple[Candidate, bool]] = {}
    deferrals: list[Drop] = []
    headroom: dict[str, tuple[int, ...]] = {}

    for index in order:
        draw = sampled[index]
        costs = draw.candidate.cost.as_tuple()
        # Captured BEFORE this draw consumes anything, so it is the capacity
        # this subject faced rather than what was left after it.
        headroom[draw.subject_token] = tuple(
            int(caps[k] - running[k]) if caps[k] != float("inf") else _UNBOUNDED
            for k in range(len(PRICED_BUDGETS))
        )
        if all(running[k] + costs[k] <= caps[k] for k in range(len(PRICED_BUDGETS))):
            for k in range(len(PRICED_BUDGETS)):
                running[k] += costs[k]
            outcome[draw.subject_token] = (draw.candidate, False)
            continue

        outcome[draw.subject_token] = (draw.fallback, True)
        if not draw.candidate.is_do_nothing:
            deferrals.append(
                Drop(
                    subject_token=draw.subject_token,
                    claim_id=draw.candidate.claim_id,
                    reason=DropReason.BUDGET_EXHAUSTED,
                    detail=(
                        f"{draw.candidate.action} sampled at adjusted value "
                        f"{draw.adjusted_value:.2f} did not fit the remaining caps"
                    ),
                )
            )

    decisions: list[Decision] = []
    admitted: list[Candidate] = []
    for draw in sampled:
        candidate, deferred = outcome[draw.subject_token]
        admitted.append(candidate)
        decisions.append(
            Decision(
                subject_token=draw.subject_token,
                claim_id=candidate.claim_id,
                intended_action=candidate.action,
                pi_intended=draw.propensity,
                decision_time=decision_time,
                planned_execution_time=planned_at,
                shadow_prices=shadow_prices,
                adjusted_value=draw.adjusted_value,
                uplift=candidate.uplift,
                propensities=draw.propensities,
                sleeping_dog=draw.sleeping_dog,
                sampled_action=draw.candidate.action,
                budget_deferred=deferred,
                residual_capacity=headroom[draw.subject_token],
            )
        )
    return decisions, admitted, deferrals


def _do_nothing_of(candidates: Sequence[Candidate], subject_token: str) -> Candidate:
    for candidate in candidates:
        if candidate.is_do_nothing:
            return candidate
    raise LookupError(
        f"{subject_token} has no do_nothing candidate; there is no decline-to-act option"
    )


def _spend_of(candidates: Sequence[Candidate]) -> Spend:
    return Spend.from_vector(
        [
            sum(candidate.cost.as_tuple()[index] for candidate in candidates)
            for index in range(len(PRICED_BUDGETS))
        ]
    )
