"""The five arms, as decision rules over one subject-cycle.

WHAT IS HELD CONSTANT AND WHAT VARIES. Every arm sees the same batch, the same
claims and the SAME uplift estimates. What differs is only the decision rule.
That separation is the whole point: if the arms disagreed about the effect
sizes as well as about what to do with them, a win would be attributable to
either, and it would be attributable to whichever one flattered the result.

    WHERE THE SHARED SURFACE STOPS, AND WHY IT STOPS THERE. The obvious way to
    make ARC win is to hand it a forecaster that already knows about annoyance
    and payday and hand the baselines one that does not. That is the
    circularity attack wearing a lab coat. So `SharedUplift` is one object,
    passed to every arm that scores anything, and it conditions ONLY on
    `ObservableState` - the facts a merchant has in its own database. It knows
    nothing about annoyance, contact history, payday timing or churn intent,
    all four of which the world implements and never discloses. Any advantage
    ARC shows therefore comes from the decision machinery - the portfolio
    objective, the budgets, the Gate - and not from being told more.

THE GATE APPLIES TO FOUR ARMS, NOT FIVE. Null, naive dunning, gateway default
and ARC all certify before acting. Greedy does not, and that is what
"unconstrained" means: it is the arm that shows what the constraints were
buying, and it can only show that by not having them. It is a measurement
instrument, and it never touches a real channel - it runs against the
simulator alone, which is the only place it is safe to let something act
ungated.

WHY GREEDY IS THE ARM THAT MATTERS. It will recover more GROSS rupees than
ARC. It should: it contacts everyone, escalates immediately, and pays whatever
a contact costs. Beating a weak baseline on gross recovery proves nothing.
Beating this one on NET value, while it blows the complaint and opt-out
guardrails and spends several times the money, is the result worth presenting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol
from uuid import UUID

import numpy as np

from arc.allocator.budgets import Budgets, cost_of
from arc.allocator.candidates import (
    ClaimView,
    SubjectPortfolio,
    candidate_value,
)
from arc.allocator.cycle import allocate
from arc.allocator.policy import DEFAULT_EPSILON, DEFAULT_TEMPERATURE
from arc.core.types import ActionType, Claim
from arc.gate.context import GateContext
from arc.proving_ground.arms import Arm
from arc.proving_ground.composed import DO_NOTHING, DecisionKey

# ---------------------------------------------------------------------------
# The shared effect surface.
#
# source: published payment-recovery benchmarks - card retry recovery in the
# high single digits to low teens per attempt, credential-refresh recovery
# materially above a blind retry, digital reminder response in the mid single
# digits, and voice contact the strongest per-attempt lever and the most
# expensive. Levels are indicative; the ORDER between them is what any policy
# reasons over, and it is the same order every arm is handed.
#
# The BASE rate per action. Conditioning on observable state happens in
# `SharedUplift` below, from the observation alone.
# ---------------------------------------------------------------------------
BASE_UPLIFT: Mapping[ActionType, float] = {
    ActionType.DO_NOTHING: 0.0,
    ActionType.RETRY: 0.085,
    ActionType.CARD_UPDATER: 0.130,
    ActionType.MANDATE_RE_REGISTER: 0.120,
    ActionType.RAIL_FALLBACK: 0.075,
    ActionType.WHATSAPP_UTILITY: 0.070,
    ActionType.SMS: 0.042,
    ActionType.EMAIL: 0.026,
    ActionType.PAYMENT_LINK: 0.078,
    ActionType.VOICE_CALL: 0.115,
    ActionType.INSTALMENT_OFFER: 0.098,
    ActionType.HUMAN_HANDOFF: 0.125,
    ActionType.STATUTORY_NOTICE: 0.055,
}


# Repairs only help where there is something to repair. A card updater against
# a healthy credential is a wasted network attempt, and a forecaster that could
# not tell the difference would be a bad forecaster rather than a neutral one.
REPAIR_MISS = 0.12  # multiplier when the fault this action fixes is absent


@dataclass(frozen=True)
class _Estimate:
    value: float


class SharedUplift:
    """The one effect surface, handed to every arm that scores anything.

    WHAT IT MAY READ, AND WHY THAT LINE IS WHERE IT IS. Only `ObservableState`
    - the reissue timestamp, the mandate status and cap, the decline history.
    Those are facts a real merchant has in its own database, and conditioning
    on them is what any competent forecaster would do.

    WHAT IT DELIBERATELY DOES NOT READ. Anything about annoyance, contact
    history, payday timing or churn intent. The world implements all four and
    the policy is never told, so ARC has to discover them through the
    allocator's own objective and the outcomes it observes.

        THAT OMISSION IS THE ANTI-CIRCULARITY MEASURE. The easy way to make
        ARC win is to hand it a forecaster that already knows the mechanisms
        the simulator uses to generate harm. Then the win is the author's, not
        the system's. Leaving them out means the guardrail columns are earned
        by the annoyance term in M8's objective and by the Gate's cooldowns,
        which are the things actually under test.

    Satisfies M8's `UpliftSource` protocol, so the real X-learner drops into
    the same socket when forecast accuracy rather than allocation machinery is
    what is being measured.
    """

    def uplift(self, observation: object, action: ActionType, ctx: object, *, propensity: float):
        base = float(BASE_UPLIFT.get(action, 0.0))
        if base == 0.0 or observation is None:
            return _Estimate(base)
        return _Estimate(base * _fault_multiplier(observation, action))


def _fault_multiplier(observation: object, action: ActionType) -> float:
    """How much a repair is worth given what the records actually say.

    Reads the merchant's own view, which for the orphaned cohort says the
    mandate is active and is wrong. Being misled by that is correct: the
    Sentinel is what discovers it, and a forecaster that saw through the
    records would be reading the answer key.
    """
    reissued = getattr(observation, "instrument_reissued_at", None)
    status = getattr(observation, "mandate_status", "active")
    cap = getattr(observation, "mandate_cap_paise", None)
    amount = getattr(observation, "plan_value_paise", None)
    over_cap = cap is not None and amount is not None and int(amount) > int(cap)

    if action is ActionType.CARD_UPDATER:
        return 1.0 if reissued is not None else REPAIR_MISS
    if action is ActionType.MANDATE_RE_REGISTER:
        return 1.0 if (status != "active" or over_cap) else REPAIR_MISS
    if action is ActionType.RAIL_FALLBACK:
        return 1.0 if (status != "active" or over_cap) else 0.6
    if action is ActionType.RETRY:
        # A retry against a known blocking fault presents into the same wall.
        return REPAIR_MISS if (reissued is not None or over_cap) else 1.0
    return 1.0


@dataclass(frozen=True)
class ClaimCase:
    """One claim, with everything any arm needs to act on it."""

    claim: Claim
    account_id: str
    gate_ctx: GateContext
    contacts_7d: int = 0
    # The agent-facing view of the account. An `ObservableState` and never an
    # `Account`, which is the boundary M4 froze and the type system enforces.
    observation: object = None

    @property
    def claim_id(self) -> UUID:
        return self.claim.claim_id

    def view(self) -> ClaimView:
        return ClaimView(
            claim=self.claim,
            gate_ctx=self.gate_ctx,
            observation=self.observation,
            contacts_7d=self.contacts_7d,
        )


@dataclass(frozen=True)
class SubjectCase:
    """One subject's whole position in one cycle."""

    subject_token: str
    claims: tuple[ClaimCase, ...]
    stratum: str = ""

    def __post_init__(self) -> None:
        if not self.claims:
            raise ValueError(f"{self.subject_token} holds no claims and is not a subject")

    @property
    def contexts(self) -> Mapping[UUID | None, GateContext]:
        return {case.claim_id: case.gate_ctx for case in self.claims}

    def account_for(self, claim_id: UUID | None) -> str | None:
        for case in self.claims:
            if case.claim_id == claim_id:
                return case.account_id
        return None

    def case_for(self, claim_id: UUID | None) -> ClaimCase | None:
        for case in self.claims:
            if case.claim_id == claim_id:
                return case
        return None


@dataclass(frozen=True)
class Draw:
    """What an arm decided for one subject, before the Gate sees it.

    `pi_alloc` is a real distribution for ARC and a point mass for every other
    arm. That asymmetry is honest and load-bearing: a deterministic arm cannot
    be evaluated off-policy, and none of them needs to be, because each is run
    directly against the simulator and measured on-policy. Only ARC's logs
    feed the doubly-robust estimator, and only ARC draws stochastically.
    """

    subject_token: str
    intended: DecisionKey
    pi_alloc: Mapping[DecisionKey, float]
    adjusted_value: float = 0.0
    sleeping_dog: bool = False
    # Budget headroom this subject's draw actually faced, in PRICED_BUDGETS
    # order. The Allocator's in-cycle admission admits a draw exactly when its
    # cost fits here, so this is what makes admission composable per branch
    # rather than only observable after the fact.
    slack: tuple[float, ...] = ()
    budget_deferred: bool = False

    @property
    def is_do_nothing(self) -> bool:
        return self.intended[1] is ActionType.DO_NOTHING

    def admissible(self, key: DecisionKey) -> bool:
        """Would the Allocator's admission step have let this branch through?

        The same inequality `_admit` applies, evaluated against the headroom
        this subject faced. Conditioning on the rest of the batch as realised
        is the same treatment the Gate composition already gives the subject's
        own contact history.
        """
        if not self.slack:
            return True
        costs = cost_of(key[1]).as_tuple()
        return all(cost <= room for cost, room in zip(costs, self.slack, strict=True))


class ArmPolicy(Protocol):
    """One arm's decision rule."""

    arm: Arm
    gated: bool

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]: ...


def _point_mass(subject: SubjectCase, key: DecisionKey) -> Draw:
    return Draw(subject_token=subject.subject_token, intended=key, pi_alloc={key: 1.0})


@dataclass
class NullArm:
    """Arm A. No intervention. Measures what arrives anyway.

    The baseline everything else is incremental against, and the only honest
    answer to "how much of this would we have got for free".
    """

    arm: Arm = Arm.NULL
    gated: bool = True

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]:
        return [_point_mass(subject, DO_NOTHING) for subject in subjects]


@dataclass
class NaiveDunningArm:
    """Arm B. Fixed-schedule dunning at T+1, T+3, T+7. The bar to beat.

    The industry default: contact on a calendar, not on a state. It does not
    look at cause, at whether the issuer is down, at whether the customer
    already paid, or at what a contact is worth. It is the comparator the
    headline is stated against because it is what most systems actually do.
    """

    arm: Arm = Arm.NAIVE_DUNNING
    gated: bool = True
    schedule: tuple[ActionType, ...] = (
        ActionType.SMS,
        ActionType.WHATSAPP_UTILITY,
        ActionType.VOICE_CALL,
    )

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]:
        draws: list[Draw] = []
        for subject in subjects:
            if cycle >= len(self.schedule):
                draws.append(_point_mass(subject, DO_NOTHING))
                continue
            # Always the first claim. A fixed-schedule dunner has no notion of
            # choosing between a subject's claims, which is part of what makes
            # it the naive arm.
            target = subject.claims[0]
            draws.append(_point_mass(subject, (target.claim_id, self.schedule[cycle])))
        return draws


@dataclass
class GatewayDefaultArm:
    """Arm C. Retry once the next day, then halt. The incumbent behaviour.

    What the payment gateway does on its own if nobody builds anything. It
    recovers real money - a retry on a temporary decline often works - and
    then it stops, which is why it is a floor rather than a strategy.
    """

    arm: Arm = Arm.GATEWAY_DEFAULT
    gated: bool = True
    retry_cycle: int = 0

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]:
        draws: list[Draw] = []
        for subject in subjects:
            if cycle != self.retry_cycle:
                draws.append(_point_mass(subject, DO_NOTHING))
                continue
            draws.append(_point_mass(subject, (subject.claims[0].claim_id, ActionType.RETRY)))
        return draws


@dataclass
class GreedyUnconstrainedArm:
    """Arm D. Maximum expected value, no budget and no guardrail.

    THE OBJECTIVE IS DELIBERATELY THE NAIVE ONE: expected recovery minus the
    direct cost of the action. No lifetime-value weight, so it will spend a
    relationship worth fifteen thousand rupees to win one worth thirteen
    hundred. No annoyance term, so contacting someone for the fourth time this
    week looks exactly as good as the first. No budget, so every subject gets
    the best action rather than the portfolio getting the best allocation. No
    Gate, so cooldowns and contact windows do not apply.

    It will win on gross rupees. That is the point. Everything it costs to get
    them shows up in the guardrail columns beside it.
    """

    arm: Arm = Arm.GREEDY_UNCONSTRAINED
    gated: bool = False
    uplift: SharedUplift = field(default_factory=SharedUplift)

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]:
        draws: list[Draw] = []
        for subject in subjects:
            best_key: DecisionKey = DO_NOTHING
            best_value = 0.0
            for case in subject.claims:
                for action in BASE_UPLIFT:
                    if action is ActionType.DO_NOTHING:
                        continue
                    tau = self.uplift.uplift(case.observation, action, None, propensity=1.0).value
                    gain = tau * float(int(case.claim.amount_paise))
                    value = gain - float(int(cost_of(action).rupee_paise))
                    if value > best_value:
                        best_value, best_key = value, (case.claim_id, action)
            draws.append(
                Draw(
                    subject_token=subject.subject_token,
                    intended=best_key,
                    pi_alloc={best_key: 1.0},
                    adjusted_value=best_value,
                )
            )
        return draws


@dataclass
class ArcArm:
    """Arm E. The system under test.

    Everything the other arms do not do: the Gate prunes the eligible set
    before anything is scored, the objective carries lifetime value and
    annoyance, budgets are priced rather than checked, the treated set shrinks
    when the caps cannot be met, and the output is a DISTRIBUTION with a
    logged propensity rather than an argmax.

    That last one is what makes this arm the only one the doubly-robust
    estimator can evaluate off-policy, and it is why it exists.
    """

    gate: object
    budgets: Budgets
    arm: Arm = Arm.ARC
    gated: bool = True
    uplift: SharedUplift = field(default_factory=SharedUplift)
    temperature: float = DEFAULT_TEMPERATURE
    epsilon: float = DEFAULT_EPSILON
    last_shadow_prices: Mapping[str, float] = field(default_factory=dict)
    last_explore_mass: float = 0.0

    def decide(
        self, subjects: Sequence[SubjectCase], cycle: int, at: datetime, rng: np.random.Generator
    ) -> list[Draw]:
        portfolio = [
            SubjectPortfolio(
                subject_token=subject.subject_token,
                arm=Arm.ARC,
                claims=tuple(case.view() for case in subject.claims),
            )
            for subject in subjects
        ]
        allocation = allocate(
            portfolio,
            self.gate,
            self.uplift,
            self.budgets,
            rng,
            decision_time=at,
            temperature=self.temperature,
            epsilon=self.epsilon,
        )
        self.last_shadow_prices = {
            str(key): value for key, value in allocation.shadow_prices.items()
        }
        self.last_explore_mass = allocation.explore_mass_share

        decided = {decision.subject_token: decision for decision in allocation.decisions}
        assert_headroom_explains_admission(allocation.decisions)

        draws: list[Draw] = []
        for subject in subjects:
            decision = decided.get(subject.subject_token)
            if decision is None:
                # Dropped before sampling - no eligible action survived
                # projection. Declining is a decision and it is logged as one.
                draws.append(_point_mass(subject, DO_NOTHING))
                continue
            draws.append(
                Draw(
                    subject_token=subject.subject_token,
                    intended=_sampled_key(decision),
                    pi_alloc=dict(decision.propensities),
                    adjusted_value=decision.adjusted_value,
                    sleeping_dog=decision.sleeping_dog,
                    slack=tuple(float(v) for v in decision.residual_capacity),
                    budget_deferred=decision.budget_deferred,
                )
            )
        return draws


class AdmissionRecordInconsistent(AssertionError):
    """The recorded residual capacity disagrees with the admission verdict."""


def assert_headroom_explains_admission(decisions: Sequence[object]) -> None:
    """The recorded headroom must imply the admission outcome it came with.

    M8 now records `residual_capacity` on every `Decision`, so M11 no longer
    replays the admission order to recover it - which is the right split, since
    the ordering rule belongs to the module that applies it.

    WHAT REMAINS WORTH CHECKING is that the recorded vector and the recorded
    verdict agree: a draw whose cost fits its headroom must have been admitted,
    and one that does not must have been deferred. If they ever disagree, the
    composed propensity is being built from a headroom that did not govern the
    decision, and every importance ratio for that subject is wrong. Cheap to
    check, silent if omitted.
    """
    mismatches: list[str] = []
    for decision in decisions:
        headroom = decision.residual_capacity  # type: ignore[attr-defined]
        if not headroom:
            continue
        sampled = decision.sampled_action or decision.intended_action  # type: ignore[attr-defined]
        costs = cost_of(sampled).as_tuple()
        fits = all(cost <= room for cost, room in zip(costs, headroom, strict=True))
        if fits == bool(decision.budget_deferred):  # type: ignore[attr-defined]
            mismatches.append(
                f"{decision.subject_token}: headroom says "  # type: ignore[attr-defined]
                f"{'admissible' if fits else 'not admissible'}, allocator says "
                f"{'deferred' if decision.budget_deferred else 'admitted'}"  # type: ignore[attr-defined]
            )
    if mismatches:
        raise AdmissionRecordInconsistent(
            f"{len(mismatches)} decision(s) carry a residual capacity that does not "
            "explain their own admission outcome, so the composed propensity would "
            "be wrong for them: " + "; ".join(mismatches[:5])
        )


def _sampled_key(decision) -> DecisionKey:
    """The branch the policy actually DREW, before admission moved it.

    `Decision` records the sampled action but not the sampled claim, so for a
    subject holding several claims the claim is recovered by matching the
    action inside the logged distribution. Single-claim subjects - the large
    majority - resolve exactly; the tie-break is the lowest claim id, so it is
    deterministic across replays rather than dependent on dict ordering.

    The alternative, keying the intended branch on the POST-admission action,
    is what makes the collapse invisible: a deferred subject would appear to
    have drawn `do_nothing` rather than to have been refused, and the refused
    mass would never be attributed.
    """
    sampled = decision.sampled_action or decision.intended_action
    if not decision.budget_deferred:
        # Admission left the draw alone, so the recorded claim IS the sampled
        # claim. Exact, and it covers every subject admission did not touch.
        return (decision.claim_id, sampled)

    matches = sorted(
        (key for key in decision.propensities if key[1] is sampled),
        key=lambda key: (key[0] is not None, str(key[0])),
    )
    if matches:
        return matches[0]
    return (decision.claim_id, decision.intended_action)


def greedy_value(claim: Claim, action: ActionType, observation: object = None) -> float:
    """Greedy's objective, exposed so the difference is readable, not implied.

    Compare with `arc.allocator.candidates.candidate_value`, which multiplies
    by the lifetime-value weight and subtracts the annoyance term. The gap
    between these two functions is most of what the guardrail columns measure.
    """
    tau = SharedUplift().uplift(observation, action, None, propensity=1.0).value
    return tau * float(int(claim.amount_paise)) - float(int(cost_of(action).rupee_paise))


def arc_value(
    claim: Claim, action: ActionType, contacts_7d: int, observation: object = None
) -> float:
    """ARC's objective, for the same side-by-side reading."""
    tau = SharedUplift().uplift(observation, action, None, propensity=1.0).value
    return candidate_value(claim, action, tau, contacts_7d)


def build_arms(gate: object, budgets: Budgets) -> list[ArmPolicy]:
    """All five, in reporting order."""
    return [
        NullArm(),
        NaiveDunningArm(),
        GatewayDefaultArm(),
        GreedyUnconstrainedArm(),
        ArcArm(gate=gate, budgets=budgets),
    ]
