"""P(pay | account, action, t), and what happens when they do not pay.

FROZEN at `simulator-frozen-v1`. Every constant below was written before any
policy code existed, and none of them is touched after the tag. That ordering
is the defence against the circularity attack: a world whose constants were
adjusted until the policy looked good measures nothing.

The seven-term logit, exactly as specified:

    P(pay) = sigmoid( b0
                    + b1 * ability_to_pay(t)
                    + b2 * responsiveness[channel]
                    + b3 * timing_fit(t, salary_day)
                    + b4 * issuer_health(issuer, t)
                    - b5 * annoyance_sensitivity * contacts_7d
                    - b6 * friction(action)
                    + b7 * amount_affordability(amount, income) )

Two terms carry the whole result.

`b5 > 0` is the sleeping-dog term. Because `contacts_7d` counts the contact
being considered, an account with high annoyance sensitivity and low channel
responsiveness has a genuinely negative treatment effect: contacting it
destroys value. A policy that contacts everyone loses to one that does not,
and the uplift model at M7 has to find which is which.

`b3` is the payday term. `salary_day` is latent and jittered per month, so a
fixed-calendar policy - dun on T+1, T+3, T+7 - cannot align with it. It is
inferable from `prior_payment_timestamps`, which ARE observable, so a learned
policy can. That gap between the naive arm and ARC is structural, not tuned.

On sourcing: every constant names the published figure it is anchored to.
Where a published rate had to be turned into a logit coefficient, the comment
says so - the anchor is the observable rate the model reproduces, and the
transformation from that rate to a beta is ours, not the source's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from arc.core.types import ActionType

# ---------------------------------------------------------------------------
# Channels, as the world sees them. Deliberately a separate vocabulary from
# `arc.gate.context.Channel`: the simulator must not import the gate, and a
# world that shared the policy's enum would be sharing the policy's view.
# ---------------------------------------------------------------------------


class SimChannel(StrEnum):
    NONE = "none"
    RAIL = "rail"  # silent, no human involved
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"
    PAYMENT_LINK = "payment_link"
    VOICE = "voice"
    HUMAN = "human"
    POSTAL = "postal"


ACTION_CHANNEL: Mapping[ActionType, SimChannel] = MappingProxyType(
    {
        ActionType.DO_NOTHING: SimChannel.NONE,
        ActionType.RETRY: SimChannel.RAIL,
        ActionType.CARD_UPDATER: SimChannel.RAIL,
        ActionType.MANDATE_RE_REGISTER: SimChannel.RAIL,
        ActionType.RAIL_FALLBACK: SimChannel.RAIL,
        ActionType.WHATSAPP_UTILITY: SimChannel.WHATSAPP,
        ActionType.SMS: SimChannel.SMS,
        ActionType.EMAIL: SimChannel.EMAIL,
        ActionType.PAYMENT_LINK: SimChannel.PAYMENT_LINK,
        ActionType.VOICE_CALL: SimChannel.VOICE,
        ActionType.INSTALMENT_OFFER: SimChannel.VOICE,
        ActionType.HUMAN_HANDOFF: SimChannel.HUMAN,
        ActionType.STATUTORY_NOTICE: SimChannel.POSTAL,
    }
)

# Channels that reach a person. Only these accrue annoyance, and only these
# can produce an opt-out or a complaint.
CONTACT_CHANNELS: frozenset[SimChannel] = frozenset(
    {
        SimChannel.WHATSAPP,
        SimChannel.SMS,
        SimChannel.EMAIL,
        SimChannel.PAYMENT_LINK,
        SimChannel.VOICE,
        SimChannel.HUMAN,
        SimChannel.POSTAL,
    }
)

CONTACT_ACTIONS: frozenset[ActionType] = frozenset(
    action for action, channel in ACTION_CHANNEL.items() if channel in CONTACT_CHANNELS
)

# Actions that present a debit to a rail. Their success depends on funds and
# issuer health and not at all on whether anyone read a message. Three of the
# four repair something first: that is the silent, zero-contact recovery path.
DEBIT_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.RETRY,
        ActionType.CARD_UPDATER,
        ActionType.RAIL_FALLBACK,
        ActionType.MANDATE_RE_REGISTER,
    }
)

# Channels through which a person can commit to a date. A promise needs a
# conversation, which is why an SMS does not produce one.
PROMISE_CHANNELS: frozenset[SimChannel] = frozenset({SimChannel.VOICE, SimChannel.HUMAN})

# Automated digital outreach - the nudge family a cost-blind policy sprays.
# The sleeping-dog test is defined over these rather than over every contact
# action, because a human handoff and a statutory notice are escalations with
# their own economics, not nudges, and lumping them in would define the cohort
# out of existence.
DIGITAL_NUDGE_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.WHATSAPP_UTILITY,
        ActionType.SMS,
        ActionType.EMAIL,
        ActionType.PAYMENT_LINK,
    }
)


# ---------------------------------------------------------------------------
# The seven betas
# ---------------------------------------------------------------------------

# source: NPCI monthly NACH debit statistics, where roughly a third of
# presented debits are returned, and the observation that most returns are
# funds-related. b0 is set so an untouched failed claim recovers on its own at
# ~16% within a cycle - the natural-recovery baseline arm A exists to measure.
# It absorbs the healthy-issuer level of the b4 term, so the two move together.
# The published rate is the anchor; the logit intercept is our transformation.
B0_BASELINE = -5.64

# source: NPCI NACH return-reason mix, where funds-insufficient dominates the
# return population. b1 makes ability-to-pay the largest single driver, so a
# debit presented against an empty account fails whatever the channel.
B1_ABILITY = 2.10

# source: published messaging engagement benchmarks - WhatsApp utility
# templates read at rates well above SMS and email open rates. b2 spreads
# per-account channel responsiveness across that observed range.
B2_RESPONSIVENESS = 0.95

# source: card-network and PSP re-presentment guidance, which reports
# materially higher authorisation when a retry is aligned to the payer's
# salary credit rather than to a fixed calendar. b3 reproduces that lift at
# zero days past the credit, decaying over TIMING_SIGMA_DAYS.
B3_TIMING_FIT = 1.05

# source: RBI system-outage disclosures and PSP status pages, where an issuer
# incident collapses authorisation for the duration of its window rather than
# degrading it gently. b4 is large because that collapse is large: at the
# residual health of an outage this term removes about three logits, taking a
# 35% retry to under 3%. It also carries the steady-state quality spread
# between a large private issuer and a cooperative, which is the baseline the
# cohort detector has to distinguish a real incident from.
B4_ISSUER_HEALTH = 3.20

# source: collections research on contact frequency and disengagement, which
# finds response falling once contact volume passes a few per week, and
# reports a segment for whom outreach reduces recovery outright.
#
# b5 > 0 IS THE SLEEPING-DOG TERM, and its size is what decides whether that
# segment exists. It has to exceed the friction relief a contact provides -
# b6 * (friction(do_nothing) - friction(contact)) - plus the responsiveness
# bonus, or every account is better off contacted and the uplift model has
# nothing to find. At this value a high-annoyance, low-responsiveness account
# is strictly worse off on every digital channel, which puts the cohort at
# roughly a sixth of the population.
B5_ANNOYANCE = 1.90

# source: payment-funnel drop-off between a prompt and a completed payment.
# Friction is the effort the action demands of the customer: a rail retry
# demands none, an email demands opening it and then acting on it.
B6_FRICTION = 1.31

# source: affordability practice in Indian lending, which treats an obligation
# above roughly a quarter of monthly income as materially harder to clear in
# one payment. b7 reproduces that curvature.
B7_AFFORDABILITY = 0.90

# source: payment-funnel drop-off between a prompt and a completed payment,
# and account-updater and mandate-repair success reporting for the silent
# actions. Effort the action demands of the customer, in [0, 1]. do_nothing
# sits at the top of the scale: nothing prompts them, so only a self-initiated payment
# lands. Its distance from the other actions is what makes contact worth
# anything at all.
ACTION_FRICTION: Mapping[ActionType, float] = MappingProxyType(
    {
        ActionType.DO_NOTHING: 0.90,
        ActionType.RETRY: 0.05,
        ActionType.CARD_UPDATER: 0.12,
        ActionType.MANDATE_RE_REGISTER: 0.20,
        ActionType.RAIL_FALLBACK: 0.15,
        ActionType.WHATSAPP_UTILITY: 0.35,
        ActionType.SMS: 0.45,
        ActionType.EMAIL: 0.55,
        ActionType.PAYMENT_LINK: 0.30,
        ActionType.VOICE_CALL: 0.25,
        ActionType.INSTALMENT_OFFER: 0.18,
        ActionType.HUMAN_HANDOFF: 0.20,
        ActionType.STATUTORY_NOTICE: 0.40,
    }
)

# source: affordability practice as above - the share of monthly income at
# which an obligation stops being clearable in one payment.
AFFORDABLE_INCOME_SHARE = 0.25

# source: salary-credit clustering in Indian payroll, where balances peak on
# the credit date and are materially drawn down within the first week.
TIMING_SIGMA_DAYS = 3.0
SALARY_TROUGH = 0.55
SALARY_DECAY_DAYS = 9.0


# ---------------------------------------------------------------------------
# Adverse-outcome hazards. The guardrail metrics at M11 have no source without
# these, and a recovery number reported without them is not a result.
# ---------------------------------------------------------------------------

# source: published messaging opt-out benchmarks, which sit in the single
# digits per thousand for transactional and utility traffic. The exponent
# above one is the point: opt-out is superlinear in contact volume, so an
# unconstrained policy destroys the channel it depends on.
OPT_OUT_BASE = 0.020
OPT_OUT_EXPONENT = 1.4

# source: card-network complaint thresholds and collections-conduct guidance,
# where sustained complaint rates in the low single digits per thousand
# contacts are already a supervisory concern.
COMPLAINT_BASE = 0.0125
COMPLAINT_EXPONENT = 1.8

# source: subscription churn benchmarks for involuntary-failure cohorts, where
# a failed charge followed by pressure is a leading cancellation cause.
CANCEL_BASE = 0.028

# source: card-network dispute ratios, which stay below one percent of
# transactions before a monitoring programme engages.
DISPUTE_BASE = 0.006

# source: dispute and complaint reporting for repeated failed debit
# presentation, which is an order of magnitude below the rates for outbound
# contact. A silent rail action reaches nobody, so it cannot produce an
# opt-out at all, and only a repeated failed debit produces a complaint.
SILENT_HARM_SCALE = 0.08

# source: collections practice, where promise-to-pay capture on a connected
# call sits well under half of connects.
PROMISE_ELICIT_RATE = 0.34


@dataclass(frozen=True, slots=True)
class ResponseInputs:
    """Everything the seven terms need, assembled by the world.

    A flat record of primitives rather than an account object, so the response
    model cannot reach anything the seven terms do not name.
    """

    ability_to_pay: float
    responsiveness: float
    timing_fit: float
    issuer_health: float
    annoyance_sensitivity: float
    contacts_7d: int
    friction: float
    affordability: float


def sigmoid(x: float) -> float:
    """Numerically stable logistic. Overflow on a large negative logit would
    otherwise turn an impossible payment into a NaN."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def logit(inputs: ResponseInputs) -> float:
    """The seven terms, in the order they are specified. Nothing else."""
    return (
        B0_BASELINE
        + B1_ABILITY * inputs.ability_to_pay
        + B2_RESPONSIVENESS * inputs.responsiveness
        + B3_TIMING_FIT * inputs.timing_fit
        + B4_ISSUER_HEALTH * inputs.issuer_health
        - B5_ANNOYANCE * inputs.annoyance_sensitivity * inputs.contacts_7d
        - B6_FRICTION * inputs.friction
        + B7_AFFORDABILITY * inputs.affordability
    )


def p_pay(inputs: ResponseInputs) -> float:
    return sigmoid(logit(inputs))


def timing_fit(days_since_salary: float) -> float:
    """Alignment of the attempt with the salary credit. Peaks at the credit.

    An attempt landing before the credit scores zero rather than a mirrored
    value, because money that has not arrived yet is not money.
    """
    if days_since_salary < 0:
        return 0.0
    return math.exp(-0.5 * (days_since_salary / TIMING_SIGMA_DAYS) ** 2)


def funds_cycle(days_since_salary: float) -> float:
    """Within-month balance decay, from 1.0 at the credit to SALARY_TROUGH."""
    if days_since_salary < 0:
        return SALARY_TROUGH
    return SALARY_TROUGH + (1.0 - SALARY_TROUGH) * math.exp(-days_since_salary / SALARY_DECAY_DAYS)


def affordability(amount_paise: int, monthly_income_paise: int) -> float:
    """1.0 when the amount is trivial against income, falling toward 0.

    Hyperbolic rather than linear: the difference between 5% and 10% of income
    matters far less than the difference between 40% and 80%.
    """
    if monthly_income_paise <= 0:
        return 0.0
    ceiling = monthly_income_paise * AFFORDABLE_INCOME_SHARE
    return float(1.0 / (1.0 + amount_paise / ceiling))


def friction_of(action: ActionType) -> float:
    return ACTION_FRICTION[action]


def is_contact(action: ActionType) -> bool:
    return action in CONTACT_ACTIONS


# ---------------------------------------------------------------------------
# What happens when they do not pay
# ---------------------------------------------------------------------------


class OutcomeKind(StrEnum):
    """The six outcomes of one attempt.

    Not binary, because the guardrails at M11 need a source: an opt-out and a
    complaint are results, not noise.

    NO_RESPONSE is the residual - no payment and no adverse signal. A promise
    can accompany it, because a promise is a commitment that resolves later
    rather than an outcome of this attempt.
    """

    PAID = "paid"
    NO_RESPONSE = "no_response"
    OPT_OUT = "opt_out"
    COMPLAINT = "complaint"
    VOLUNTARY_CANCEL = "voluntary_cancel"
    DISPUTE = "dispute"


ADVERSE_OUTCOMES: frozenset[OutcomeKind] = frozenset(
    {
        OutcomeKind.OPT_OUT,
        OutcomeKind.COMPLAINT,
        OutcomeKind.VOLUNTARY_CANCEL,
        OutcomeKind.DISPUTE,
    }
)


@dataclass(frozen=True, slots=True)
class HarmHazards:
    opt_out: float
    complaint: float
    voluntary_cancel: float
    dispute: float

    def total(self) -> float:
        return self.opt_out + self.complaint + self.voluntary_cancel + self.dispute


def harm_hazards(
    *,
    action: ActionType,
    annoyance_sensitivity: float,
    intent_to_churn: float,
    contacts_7d: int,
    prior_attempts: int,
) -> HarmHazards:
    """Probability of each adverse outcome, conditional on not paying.

    Contact pressure enters superlinearly, so the fifth message in a week is
    far more damaging than the first. That curvature is what makes the contact
    budget worth spending carefully rather than exhausting.
    """
    contacted = is_contact(action)
    scale = 1.0 if contacted else SILENT_HARM_SCALE
    pressure = float(max(contacts_7d, 1))

    opt_out = 0.0
    if contacted:
        opt_out = OPT_OUT_BASE * annoyance_sensitivity * pressure**OPT_OUT_EXPONENT

    complaint = COMPLAINT_BASE * annoyance_sensitivity**2 * pressure**COMPLAINT_EXPONENT * scale
    cancel = CANCEL_BASE * intent_to_churn * (1.0 + annoyance_sensitivity * pressure) * scale
    dispute = DISPUTE_BASE * (1.0 + 0.25 * prior_attempts) * scale

    return HarmHazards(
        opt_out=min(opt_out, 0.45),
        complaint=min(complaint, 0.25),
        voluntary_cancel=min(cancel, 0.25),
        dispute=min(dispute, 0.20),
    )


def sample_outcome(
    p: float,
    hazards: HarmHazards,
    generator: np.random.Generator,
) -> OutcomeKind:
    """Draw one of the six: payment, then the adverse hazards, then residual.

    Hazards are scaled down rather than truncated when they would exceed the
    non-payment mass, so no outcome silently becomes impossible.
    """
    draw = float(generator.random())
    if draw < p:
        return OutcomeKind.PAID

    remaining = 1.0 - p
    total = hazards.total()
    shrink = 1.0 if total <= remaining else remaining / total

    cursor = p
    for kind, hazard in (
        (OutcomeKind.OPT_OUT, hazards.opt_out),
        (OutcomeKind.COMPLAINT, hazards.complaint),
        (OutcomeKind.VOLUNTARY_CANCEL, hazards.voluntary_cancel),
        (OutcomeKind.DISPUTE, hazards.dispute),
    ):
        cursor += hazard * shrink
        if draw < cursor:
            return kind
    return OutcomeKind.NO_RESPONSE
