"""Budgets, costs, and the line between a hard limit and a priced resource.

THE SPLIT THAT MATTERS. Cooldowns are hard limits and live in the Gate;
contact VOLUME is a budget and lives here. A forty-eight hour voice cooldown
must never be purchasable at any expected value, but which of this week's three
contact slots to spend on this subject rather than that one is exactly what the
knapsack should decide. A system where sufficient value can buy past a cooldown
will eventually harass someone, and a system where volume is a hard limit
spends its scarcest resource first-come-first-served.

So nothing in this module can refuse an action. It can only make one expensive.

MONEY STAYS INTEGER (GI-2). Every cost and every cap here is an integer -
contact slots, voice minutes, network attempts, agent minutes, and paise. The
Lagrange multipliers and the adjusted values computed from them are floats, and
that is not a violation: a shadow price is a ratio, not an amount, and an
expected value is an expectation of money rather than money. No float ever
holds a quantity that anybody is owed or charged.

`B_explore` is deliberately not priced. It records what the epsilon floor spent
on exploration so that exploration reads as a deliberate four-or-five percent
line item at M11 rather than as waste, but pricing it would make the optimiser
trade away the overlap that off-policy evaluation depends on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from arc.core.money import Paise, paise
from arc.core.types import ActionType


class BudgetKey(StrEnum):
    """The scarce resources a cycle spends."""

    CONTACT = "contact"  # contact slots, portfolio per cycle
    VOICE = "voice"  # voice minutes, portfolio per cycle
    RUPEE = "rupee"  # messaging and telephony spend, portfolio per day
    RETRY = "retry"  # network attempts
    HUMAN = "human"  # agent minutes, portfolio per day
    CONCESSION = "concession"  # waivers and discounts, portfolio per month
    EXPLORE = "explore"  # epsilon-floor spend - REPORTED, never priced


# Priced dimensions, in a fixed order. The order is part of the contract: the
# solver holds costs as a matrix whose columns are these, and inserting a key
# in the middle would silently re-map an already-computed shadow price.
PRICED_BUDGETS: tuple[BudgetKey, ...] = (
    BudgetKey.CONTACT,
    BudgetKey.VOICE,
    BudgetKey.RUPEE,
    BudgetKey.RETRY,
    BudgetKey.HUMAN,
    BudgetKey.CONCESSION,
)

BUDGET_INDEX: Mapping[BudgetKey, int] = MappingProxyType(
    {key: position for position, key in enumerate(PRICED_BUDGETS)}
)


@dataclass(frozen=True)
class CostVector:
    """What one action consumes. Integers throughout."""

    contact: int = 0
    voice_minutes: int = 0
    rupee_paise: Paise = paise(0)
    retry_attempts: int = 0
    human_minutes: int = 0
    concession_paise: Paise = paise(0)

    def __post_init__(self) -> None:
        # Validate the RAW attributes, not `as_mapping()`, which casts to int
        # and would launder a float past the check it exists to fail (GI-2).
        for name in (
            "contact",
            "voice_minutes",
            "rupee_paise",
            "retry_attempts",
            "human_minutes",
            "concession_paise",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"cost {name} must be an integer, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"cost {name} is negative ({value}); a refund is not an action")

    def as_mapping(self) -> dict[str, int]:
        return {
            "contact": self.contact,
            "voice_minutes": self.voice_minutes,
            "rupee_paise": int(self.rupee_paise),
            "retry_attempts": self.retry_attempts,
            "human_minutes": self.human_minutes,
            "concession_paise": int(self.concession_paise),
        }

    def as_tuple(self) -> tuple[int, ...]:
        """In `PRICED_BUDGETS` order, for the solver's cost matrix."""
        return (
            self.contact,
            self.voice_minutes,
            int(self.rupee_paise),
            self.retry_attempts,
            self.human_minutes,
            int(self.concession_paise),
        )

    @property
    def is_free(self) -> bool:
        return not any(self.as_tuple())


ZERO_COST = CostVector()

# ---------------------------------------------------------------------------
# What each action costs.
#
# source: Indian messaging and telephony rate cards - WhatsApp utility
# conversations priced well below the marketing category, transactional SMS at
# a fraction of that again, and outbound voice billed by the minute. The
# absolute levels matter less than the ORDER OF MAGNITUDE between tiers, which
# is what makes the escalation ladder an economic argument rather than a
# preference: each tier costs roughly ten times the one below it, so skipping
# to a call for a claim a retry would have fixed is both wasteful and more
# intrusive than necessary.
#
# A rail action costs no contact slot because it reaches nobody. That is the
# whole point of the SELF_HEALING path: money recovered without a message.
# ---------------------------------------------------------------------------
ACTION_COST: Mapping[ActionType, CostVector] = MappingProxyType(
    {
        ActionType.DO_NOTHING: ZERO_COST,
        # Silent. No contact slot, but a network attempt is a scarce,
        # network-capped resource in its own right.
        ActionType.RETRY: CostVector(retry_attempts=1, rupee_paise=paise(30)),
        ActionType.CARD_UPDATER: CostVector(retry_attempts=1, rupee_paise=paise(250)),
        ActionType.MANDATE_RE_REGISTER: CostVector(retry_attempts=1, rupee_paise=paise(400)),
        ActionType.RAIL_FALLBACK: CostVector(retry_attempts=1, rupee_paise=paise(60)),
        # Digital. One contact slot each; the money differs by an order of
        # magnitude across the three.
        ActionType.WHATSAPP_UTILITY: CostVector(contact=1, rupee_paise=paise(115)),
        ActionType.SMS: CostVector(contact=1, rupee_paise=paise(18)),
        ActionType.EMAIL: CostVector(contact=1, rupee_paise=paise(4)),
        ActionType.PAYMENT_LINK: CostVector(contact=1, rupee_paise=paise(90)),
        # Assisted. Voice minutes are the tightest dimension in most cycles,
        # which is why lambda_voice is the shadow price worth showing a judge.
        ActionType.VOICE_CALL: CostVector(contact=1, voice_minutes=3, rupee_paise=paise(210)),
        ActionType.INSTALMENT_OFFER: CostVector(
            contact=1,
            voice_minutes=4,
            rupee_paise=paise(280),
            concession_paise=paise(1),
        ),
        # Escalated. Human time is the most expensive thing the system spends.
        ActionType.HUMAN_HANDOFF: CostVector(contact=1, human_minutes=8, rupee_paise=paise(900)),
        ActionType.STATUTORY_NOTICE: CostVector(contact=1, rupee_paise=paise(4_500)),
    }
)


def cost_of(action: ActionType) -> CostVector:
    return ACTION_COST[action]


# Actions that reach nobody. The sleeping-dog rule needs this set: when every
# contact action scores negative, these are what remains available, and the
# optimiser choosing one of them is what "silent actions only" looks like from
# the inside.
SILENT_ACTIONS: frozenset[ActionType] = frozenset(
    action for action, cost in ACTION_COST.items() if cost.contact == 0
)
CONTACT_ACTIONS: frozenset[ActionType] = frozenset(ACTION_COST) - SILENT_ACTIONS


@dataclass(frozen=True)
class Budgets:
    """The caps for one cycle. Frozen, because nothing may relax one.

    A cap of zero is meaningful and different from absent: zero voice minutes
    is a decision to run no calls this cycle, and the optimiser must respect it
    rather than treating it as unset.
    """

    caps: Mapping[BudgetKey, int]

    def __post_init__(self) -> None:
        for key, cap in self.caps.items():
            if key not in PRICED_BUDGETS:
                raise ValueError(f"{key} is not a priced budget dimension")
            if isinstance(cap, bool) or not isinstance(cap, int):
                raise TypeError(f"budget {key} must be an integer, got {type(cap).__name__}")
            if cap < 0:
                raise ValueError(f"budget {key} is negative ({cap})")
        object.__setattr__(self, "caps", MappingProxyType(dict(self.caps)))

    def cap(self, key: BudgetKey) -> int | None:
        """None means unconstrained, which is not the same as zero."""
        return self.caps.get(key)

    def as_vector(self) -> tuple[float, ...]:
        """Caps in `PRICED_BUDGETS` order; an absent cap becomes infinity."""
        return tuple(
            float(self.caps[key]) if key in self.caps else float("inf") for key in PRICED_BUDGETS
        )

    @property
    def constrained(self) -> tuple[BudgetKey, ...]:
        return tuple(key for key in PRICED_BUDGETS if key in self.caps)


@dataclass(frozen=True)
class Spend:
    """What a plan consumes, in the same units as the caps."""

    amounts: Mapping[BudgetKey, int]

    @classmethod
    def from_vector(cls, vector: tuple[float, ...] | list[float]) -> Spend:
        return cls({key: int(round(vector[index])) for key, index in BUDGET_INDEX.items()})

    def of(self, key: BudgetKey) -> int:
        return self.amounts.get(key, 0)

    def within(self, budgets: Budgets) -> bool:
        return all(self.of(key) <= cap for key, cap in budgets.caps.items())

    def overruns(self, budgets: Budgets) -> dict[BudgetKey, int]:
        """By how much each cap was exceeded. Empty when the plan is legal."""
        return {key: self.of(key) - cap for key, cap in budgets.caps.items() if self.of(key) > cap}
