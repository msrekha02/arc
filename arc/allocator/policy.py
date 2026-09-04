"""The stochastic policy, and why an argmax would destroy the headline number.

The dual gives every subject a deterministic best action. Taking it would be
the obvious thing to do and it would quietly make the whole system
unmeasurable, so this module converts that argmax into a distribution and
returns the exact probability it sampled with.

WHY. Off-policy evaluation asks what a different policy would have recovered,
using data this policy generated. That question only has an answer where the
logged policy had a chance of taking the other action - the overlap condition.
A deterministic policy assigns probability one to one action and zero to the
rest, so for every counterfactual action the importance weight is a division
by zero and the estimate is undefined. Not noisy. Undefined. The doubly-robust
estimator at M11, the X-learner's propensity weights at M7, and the confidence
interval on the headline number all rest on this being a distribution.

THREE THINGS THE FLOOR BUYS:

  1. pi(a|s) is KNOWN EXACTLY rather than estimated. Most industrial uplift
     work has to fit a propensity model and inherits the mis-specification as
     bias. Here the number is recorded because the policy drew with it.

  2. Overlap is guaranteed. Every eligible action has probability at least
     eps/n, so no region of the action space is unobserved and the importance
     ratios cannot blow up.

  3. Exploration is deliberate and priced separately. The epsilon spend is
     reported as `B_explore` rather than folded into the objective, so it reads
     as a four-or-five percent line item rather than as waste.

Temperature is the exploration dial. It does not change which action is best;
it changes how sharply the distribution concentrates on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from arc.allocator.budgets import PRICED_BUDGETS, BudgetKey
from arc.allocator.candidates import Candidate
from arc.core.types import ActionType

# The propensity floor. Five percent of the mass is spread uniformly across the
# eligible set, so the rarest action a subject could receive still has
# probability eps/n.
DEFAULT_EPSILON = 0.05

# Temperature is a FRACTION OF THE SUBJECT'S OWN VALUE SPREAD, not an absolute
# number of paise.
#
# WHY RELATIVE. Adjusted values are in paise-scale, and a portfolio of
# two-hundred-rupee subscriptions and one of two-lakh invoices differ by three
# orders of magnitude. An absolute temperature that explores sensibly on the
# first is a hard argmax on the second and a coin flip on a third, so the dial
# would have to be retuned per portfolio and would silently be wrong whenever
# nobody remembered to. Scaling by the spread between the best and worst
# adjusted value for that subject makes the same number mean the same thing
# everywhere: at 0.05, an action one twentieth of the spread below the best is
# about e times less likely to be drawn.
#
# CHOSEN BY MEASUREMENT, against the value actually given up rather than
# against how often the draw departs from the best action. Over a
# two-hundred-subject portfolio the mass landing anywhere but the best barely
# moves across the whole plausible range - 0.55 at a fraction of 0.02, 0.65 at
# 0.12 - because most of that mass sits on near-ties, where which one is drawn
# hardly matters. What does move is the cost: 514 paise per subject at 0.02,
# 619 at 0.05, 1253 at 0.12, against a best-action value of about 21,400. This
# setting gives up under three percent, which leaves room inside the four-to-
# five percent exploration line item for the epsilon floor's own share.
DEFAULT_TEMPERATURE = 0.05

# Below this spread the candidates are indistinguishable and the softmax would
# be dividing by noise, so the distribution falls back to uniform before the
# epsilon floor is applied.
SPREAD_FLOOR = 1e-9


class DeterministicPolicy(AssertionError):
    """A propensity distribution collapsed onto one action."""


def adjusted_values(
    candidates: Sequence[Candidate], shadow_prices: Mapping[BudgetKey, float]
) -> np.ndarray:
    """v_ia - sum_k lambda_k c^k_ia, the quantity the policy ranks by."""
    lam = np.array([shadow_prices.get(key, 0.0) for key in PRICED_BUDGETS], dtype=float)
    values = np.array([candidate.value for candidate in candidates], dtype=float)
    costs = np.array([candidate.cost.as_tuple() for candidate in candidates], dtype=float)
    return values - costs @ lam


def propensity_distribution(
    candidates: Sequence[Candidate],
    shadow_prices: Mapping[BudgetKey, float],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """pi(.|s) over one subject's candidates. Sums to one; floored at eps/n.

    Separated from sampling so that M11 can compute the composed
    allocator-and-Gate propensity in closed form without drawing anything, and
    so the floor can be asserted without a thousand samples.
    """
    if not candidates:
        raise ValueError("a subject with no candidates has no policy; do_nothing is missing")
    if not 0.0 <= epsilon < 1.0:
        raise ValueError(f"epsilon must be in [0, 1), got {epsilon}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    adjusted = adjusted_values(candidates, shadow_prices)
    count = len(candidates)

    spread = float(adjusted.max() - adjusted.min())
    if spread <= SPREAD_FLOOR:
        probabilities = np.full(count, 1.0 / count)
    else:
        # Subtracting the maximum before exponentiating is what keeps a
        # paise-scale value from overflowing to inf and turning the whole
        # distribution into a NaN.
        weights = np.exp((adjusted - adjusted.max()) / (temperature * spread))
        total = weights.sum()
        probabilities = (
            weights / total if np.isfinite(total) and total > 0 else np.full(count, 1.0 / count)
        )

    # THE PROPENSITY FLOOR.
    probabilities = (1.0 - epsilon) * probabilities + epsilon / count

    # Renormalise against float drift so the logged propensity is a real
    # probability rather than one that nearly is.
    return probabilities / probabilities.sum()


def stochastic_policy(
    subject_candidates: Sequence[Candidate],
    shadow_prices: Mapping[BudgetKey, float],
    rng: np.random.Generator,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[Candidate, float]:
    """Sample one action for one subject and return it with its exact pi(a|s).

    NEVER RETURNS AN ARGMAX. There is no `deterministic=True`, no temperature
    of zero, and no "just for the demo" path - a flag like that would be used
    under pressure and the headline number would silently stop meaning
    anything. The generator is injected rather than global so that a replay
    reproduces the same draw.
    """
    if rng is None:
        raise ValueError("an injected generator is required; there is no global rng in this repo")

    probabilities = propensity_distribution(
        subject_candidates,
        shadow_prices,
        temperature=temperature,
        epsilon=epsilon,
    )
    index = int(rng.choice(len(subject_candidates), p=probabilities))
    return subject_candidates[index], float(probabilities[index])


def explore_spend(
    candidates: Sequence[Candidate],
    shadow_prices: Mapping[BudgetKey, float],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Expected adjusted value given up by sampling instead of taking the best.

    `B_explore`, in paise-scale. THE VALUE FOREGONE AND NOT THE DISPLACED MASS,
    because those differ by a lot and only one of them is a cost. When two
    actions are worth almost the same, splitting the draw between them moves a
    great deal of probability and gives up almost nothing; reporting the mass
    would book that as expensive exploration when it was very nearly free.

    Naming this number is what stops the epsilon floor reading as a bug at M11.
    It is the price of being measurable, and it is small.
    """
    if not candidates:
        return 0.0
    adjusted = adjusted_values(candidates, shadow_prices)
    probabilities = propensity_distribution(
        candidates, shadow_prices, temperature=temperature, epsilon=epsilon
    )
    return float(adjusted.max() - float(probabilities @ adjusted))


def explore_mass(
    candidates: Sequence[Candidate],
    shadow_prices: Mapping[BudgetKey, float],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Probability the draw lands anywhere but the best action.

    Reported alongside `explore_spend` because the two answer different
    questions: this one is how often the policy departs from greedy, that one
    is what the departures cost.
    """
    if not candidates:
        return 0.0
    probabilities = propensity_distribution(
        candidates, shadow_prices, temperature=temperature, epsilon=epsilon
    )
    greedy = int(np.argmax(adjusted_values(candidates, shadow_prices)))
    return float(1.0 - probabilities[greedy])


def sleeping_dog(candidates: Sequence[Candidate]) -> bool:
    """Every contact action available to this subject scores negative uplift.

    A READING OF THE MODEL, NOT A RULE. Nothing here decides that contact is
    bad; it observes that M7's signed estimate is below zero on every channel
    the Gate left eligible. The optimiser needs no help acting on it - a
    negative-uplift action already scores below `do_nothing` at zero - so this
    exists for the audit trail and the console, not for the decision.
    """
    contact = [candidate for candidate in candidates if not candidate.is_silent]
    return bool(contact) and all(candidate.uplift < 0.0 for candidate in contact)


def distinct_actions(
    subject_candidates: Sequence[Candidate],
    shadow_prices: Mapping[BudgetKey, float],
    rng: np.random.Generator,
    draws: int,
    **kwargs: float,
) -> set[ActionType]:
    """Sample repeatedly - the shape of the stochasticity gate, in one place."""
    return {
        stochastic_policy(subject_candidates, shadow_prices, rng, **kwargs)[0].action
        for _ in range(draws)
    }
