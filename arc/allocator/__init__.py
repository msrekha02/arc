"""L4 - the Allocator. A portfolio decision, not a per-account workflow.

The unit of decision is the batch, which is what makes a batch-level metric
mean anything. Within it:

    candidates.py   what may be chosen - control arm removed, Gate consulted
    budgets.py      what things cost and what the caps are
    lagrangian.py   the relaxation, and the per-subject decomposition
    policy.py       softmax with an epsilon floor, returning pi(a|s)
    cycle.py        one cycle end to end, with both times pinned

This package may not import `arc.llm_service` and may not touch simulator
ground truth; both bans are enforced in CI. It also contains no compliance rule
of its own - eligibility comes from `gate.project()` and nothing here may
second-guess it (GI-6), which `tests/test_allocator.py` asserts by walking this
package's AST.
"""

from arc.allocator.budgets import (
    ACTION_COST,
    CONTACT_ACTIONS,
    PRICED_BUDGETS,
    SILENT_ACTIONS,
    BudgetKey,
    Budgets,
    CostVector,
    Spend,
    cost_of,
)
from arc.allocator.candidates import (
    ALL_ACTIONS,
    Candidate,
    CandidatePool,
    ClaimView,
    Drop,
    DropReason,
    EligibilitySource,
    SubjectPortfolio,
    UpliftSource,
    build_candidates,
    candidate_value,
    ltv_weight,
)
from arc.allocator.cycle import Allocation, Decision, allocate
from arc.allocator.lagrangian import BudgetRelaxed, Solution, solve
from arc.allocator.policy import (
    DEFAULT_EPSILON,
    DEFAULT_TEMPERATURE,
    DeterministicPolicy,
    adjusted_values,
    explore_spend,
    propensity_distribution,
    sleeping_dog,
    stochastic_policy,
)

__all__ = [
    "ACTION_COST",
    "ALL_ACTIONS",
    "CONTACT_ACTIONS",
    "DEFAULT_EPSILON",
    "DEFAULT_TEMPERATURE",
    "PRICED_BUDGETS",
    "SILENT_ACTIONS",
    "Allocation",
    "BudgetKey",
    "BudgetRelaxed",
    "Budgets",
    "Candidate",
    "CandidatePool",
    "ClaimView",
    "CostVector",
    "Decision",
    "DeterministicPolicy",
    "Drop",
    "DropReason",
    "EligibilitySource",
    "Solution",
    "Spend",
    "SubjectPortfolio",
    "UpliftSource",
    "adjusted_values",
    "allocate",
    "build_candidates",
    "candidate_value",
    "cost_of",
    "explore_spend",
    "ltv_weight",
    "propensity_distribution",
    "sleeping_dog",
    "solve",
    "stochastic_policy",
]
