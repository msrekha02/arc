"""The optimisation: a multi-dimensional multi-choice knapsack, relaxed.

THE PROBLEM

    maximise    sum_i sum_a  v_ia * x_ia
    subject to  sum_i sum_a  c^k_ia * x_ia  <=  B_k     for each budget k
                sum_a x_ia <= 1                          for each subject i
                x_ia in {0, 1}

Exactly NP-hard, and the exact solution is not worth having: the LP
relaxation's integrality gap vanishes at portfolio scale, so with tens of
thousands of subjects the relaxed answer is effectively optimal and arrives in
seconds instead of never.

THE DECOMPOSITION IS THE WHOLE TRICK. Relax the budget constraints into the
objective with multipliers lambda_k >= 0:

    L(lambda) = max_x sum_ia ( v_ia - sum_k lambda_k * c^k_ia ) * x_ia
                + sum_k lambda_k * B_k

With the budgets priced rather than enforced, the only coupling left between
subjects is gone, and the inner maximisation SEPARATES: every subject
independently takes its best action by adjusted value. That is what turns a
joint optimisation over fifty thousand subjects into fifty thousand
independent one-line choices, and it is why this runs in seconds on one core.

The dual is convex, so each lambda_k is found by bisection, cycling over the
dimensions in coordinate ascent.

LAMBDA IS A SHADOW PRICE, AND IT IS THE EXPLAINABILITY ARTIFACT. `lambda_voice
= 340` means the marginal voice minute is worth three hundred and forty rupees
of foregone recovery elsewhere. `lambda_k = 0` means budget k is not binding.
"Voice lost here because voice was worth more elsewhere" is a real reason
derived from the optimisation rather than a rationalisation written afterwards.

STOP-EV FALLS OUT RATHER THAN BEING CONFIGURED. A subject whose best adjusted
value is not positive is not worth its budget consumption at the current
prices, and takes `do_nothing`. There is no threshold to tune.

INFEASIBILITY SHRINKS THE TREATED SET AND NEVER RELAXES A BUDGET. See
`solve`'s docstring: relaxing a constraint under pressure is precisely how
compliance systems fail, so the caps are frozen and the treated set moves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from arc.allocator.budgets import PRICED_BUDGETS, BudgetKey, Budgets, Spend
from arc.allocator.candidates import Candidate, Drop, DropReason

# Upper end of the bisection bracket. A price this high buys nothing: any
# candidate with a positive cost in that dimension is driven far below zero
# adjusted value, so only free actions survive and the dimension's spend goes
# to zero. It therefore always brackets the answer.
LAMBDA_MAX = 1e9

# Twenty-eight halvings of [0, 1e9] resolve the price to about four rupees,
# which is finer than any cost in the table.
BISECTION_STEPS = 28

# Coordinate ascent passes. Convergence is typically reached in well under ten;
# the cap exists so a pathological instance terminates rather than spins.
MAX_PASSES = 40


class BudgetRelaxed(AssertionError):
    """A solve returned a plan that exceeds a cap. Should be unreachable."""


@dataclass
class Solution:
    """Prices, the plan they imply, and everything dropped to get there."""

    shadow_prices: dict[BudgetKey, float]
    spend: Spend
    chosen: dict[str, Candidate]
    drops: list[Drop] = field(default_factory=list)
    passes: int = 0
    shrink_rounds: int = 0

    @property
    def treated(self) -> int:
        """Subjects receiving something other than `do_nothing`."""
        return sum(1 for candidate in self.chosen.values() if not candidate.is_do_nothing)


class _Problem:
    """Candidates flattened into arrays, grouped by subject.

    Built once per solve. Every bisection step is then a matrix-vector product
    and one segmented maximum over contiguous memory, which is what keeps four
    hundred thousand candidates inside a few milliseconds per pass.
    """

    def __init__(self, candidates: Sequence[Candidate]) -> None:
        order = sorted(range(len(candidates)), key=lambda i: candidates[i].subject_token)
        self.candidates = [candidates[i] for i in order]
        self.values = np.array([c.value for c in self.candidates], dtype=float)
        self.costs = np.array([c.cost.as_tuple() for c in self.candidates], dtype=float)

        tokens = [c.subject_token for c in self.candidates]
        boundary = np.ones(len(tokens), dtype=bool)
        if tokens:
            boundary[1:] = np.array(tokens[1:]) != np.array(tokens[:-1])
        self.starts = np.flatnonzero(boundary)
        self.subject_of = np.cumsum(boundary) - 1
        self.subject_tokens = [tokens[i] for i in self.starts]
        self.counts = np.diff(np.append(self.starts, len(tokens)))
        self.active = np.ones(len(self.candidates), dtype=bool)

    @property
    def n_subjects(self) -> int:
        return len(self.starts)

    def best_indices(self, lam: np.ndarray) -> np.ndarray:
        """Index of each subject's best candidate under the adjusted value.

        This is the decomposition, executed. `-inf` masks candidates the
        shrink step has dropped, so a dropped candidate can never be chosen
        without the arrays being rebuilt.
        """
        adjusted = self.values - self.costs @ lam
        adjusted = np.where(self.active, adjusted, -np.inf)

        segment_max = np.maximum.reduceat(adjusted, self.starts)
        expanded = np.repeat(segment_max, self.counts)
        flagged = np.flatnonzero(adjusted >= expanded)

        # Keep the first flagged index in each subject's block, so ties resolve
        # deterministically to the earliest candidate rather than to whichever
        # one floating point happened to favour.
        segments = self.subject_of[flagged]
        first = np.ones(len(segments), dtype=bool)
        first[1:] = segments[1:] != segments[:-1]
        return flagged[first]

    def spend_at(self, lam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Total cost per dimension, and the chosen index per subject."""
        best = self.best_indices(lam)
        adjusted = self.values[best] - self.costs[best] @ lam
        taken = best[adjusted > 0.0]
        return self.costs[taken].sum(axis=0), best


def _solve_prices(problem: _Problem, caps: np.ndarray, tol: float) -> tuple[np.ndarray, int]:
    """Coordinate ascent with bisection on each price."""
    lam = np.zeros(len(PRICED_BUDGETS), dtype=float)
    passes = 0

    for _ in range(MAX_PASSES):
        passes += 1
        previous = lam.copy()

        for dimension, cap in enumerate(caps):
            if not np.isfinite(cap):
                continue

            # A dimension that fits at price zero is not binding, and its
            # shadow price is exactly zero rather than a residue of the
            # bisection bracket.
            probe = lam.copy()
            probe[dimension] = 0.0
            if problem.spend_at(probe)[0][dimension] <= cap:
                lam[dimension] = 0.0
                continue

            low, high = 0.0, LAMBDA_MAX
            for _ in range(BISECTION_STEPS):
                middle = (low + high) / 2.0
                probe[dimension] = middle
                if problem.spend_at(probe)[0][dimension] > cap:
                    low = middle  # overspending, so raise the price
                else:
                    high = middle
            lam[dimension] = high

        if np.max(np.abs(lam - previous)) <= tol * max(1.0, float(np.max(np.abs(lam)))):
            break

    return lam, passes


def solve(
    candidates: Sequence[Candidate],
    budgets: Budgets,
    tol: float = 1e-4,
    *,
    max_shrink_rounds: int = 24,
) -> Solution:
    """Price the budgets, then let every subject choose independently.

    WHEN NO PRICE IS ENOUGH. Pricing alone satisfies most instances, because a
    high enough price drives every costly action below `do_nothing` at zero.
    It fails when the caps cannot accommodate even the cheapest positive-value
    plan the portfolio insists on - and then the answer is to TREAT FEWER
    SUBJECTS, not to spend more.

    So the fallback raises a value threshold and removes the least valuable
    candidates until the plan fits, logging every removal with its reason. The
    `Budgets` object is never touched; it is frozen, and the returned spend is
    asserted against the original caps before this function returns.
    """
    problem = _Problem(candidates)
    caps = np.array(budgets.as_vector(), dtype=float)

    shrink_rounds = 0
    drops: list[Drop] = []
    lam, passes = _solve_prices(problem, caps, tol)
    spend_vector, best = problem.spend_at(lam)

    while np.any(spend_vector > caps + 1e-6) and shrink_rounds < max_shrink_rounds:
        shrink_rounds += 1
        drops.extend(_shrink(problem, lam, caps))
        lam, extra = _solve_prices(problem, caps, tol)
        passes += extra
        spend_vector, best = problem.spend_at(lam)

    chosen: dict[str, Candidate] = {}
    adjusted = problem.values[best] - problem.costs[best] @ lam
    for position, index in enumerate(best):
        candidate = problem.candidates[index]
        # STOP-EV: not worth its budget consumption at the current prices.
        if adjusted[position] <= 0.0 and not candidate.is_do_nothing:
            candidate = _do_nothing_for(problem, candidate.subject_token)
        chosen[candidate.subject_token] = candidate

    spend = Spend.from_vector(
        [
            sum(candidate.cost.as_tuple()[index] for candidate in chosen.values())
            for index in range(len(PRICED_BUDGETS))
        ]
    )

    overruns = spend.overruns(budgets)
    if overruns:
        raise BudgetRelaxed(
            "the solver returned a plan that exceeds its caps, which means a budget "
            f"was relaxed rather than the treated set shrunk: {overruns}"
        )

    return Solution(
        shadow_prices={key: float(lam[index]) for index, key in enumerate(PRICED_BUDGETS)},
        spend=spend,
        chosen=chosen,
        drops=drops,
        passes=passes,
        shrink_rounds=shrink_rounds,
    )


def _do_nothing_for(problem: _Problem, subject_token: str) -> Candidate:
    for candidate in problem.candidates:
        if candidate.subject_token == subject_token and candidate.is_do_nothing:
            return candidate
    raise LookupError(
        f"{subject_token} has no do_nothing candidate; the knapsack has no "
        "decline-to-act option and will spend budget on negative value"
    )


def _shrink(problem: _Problem, lam: np.ndarray, caps: np.ndarray) -> list[Drop]:
    """Remove the least valuable treated candidates until the plan can fit.

    Raises the effective value threshold rather than lowering a cap. The
    subjects removed here keep their `do_nothing` candidate, so they are not
    erased from the portfolio - they are declined for this cycle, with a
    reason.
    """
    best = problem.best_indices(lam)
    adjusted = problem.values[best] - problem.costs[best] @ lam
    treated = [
        (adjusted[position], index)
        for position, index in enumerate(best)
        if not problem.candidates[index].is_do_nothing and adjusted[position] > 0.0
    ]
    if not treated:
        # Nothing left to shrink: the caps cannot be met even by treating
        # nobody, which means a cap is negative or a free action overruns.
        return []

    treated.sort(key=lambda item: item[0])
    cut = max(1, len(treated) // 10)
    drops: list[Drop] = []
    for _, index in treated[:cut]:
        candidate = problem.candidates[index]
        problem.active[index] = False
        drops.append(
            Drop(
                subject_token=candidate.subject_token,
                claim_id=candidate.claim_id,
                reason=DropReason.INFEASIBLE_SHRINK,
                detail=(
                    f"{candidate.action} dropped at adjusted value "
                    f"{problem.values[index] - float(problem.costs[index] @ lam):.2f}; "
                    f"caps {caps.tolist()} held"
                ),
            )
        )
    return drops
