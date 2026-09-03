"""The simulated world: latent state, population, and ground-truth outcomes.

FROZEN at `simulator-frozen-v1`.

Two rules govern this module.

1. THE OBSERVABILITY BOUNDARY. `World.observe()` returns an `ObservableState`
   and nothing else. `LatentState` is never reachable from it - not by
   attribute, not through `__dict__` (there is none), not by dataclass
   introspection, not through the pickled bytes, and not by walking the object
   graph. `tests/test_simulator.py` tries all of those routes and asserts each
   one fails. The type boundary is the claim, so it is a tested one.

2. THE ANTI-CIRCULARITY GUARD. This package does not import `arc.allocator`,
   `arc.forecaster` or `arc.gate`. The world does not know about the policy
   that will be measured against it, and `World.counterfactual()` - which
   returns ground truth - is for the evaluation harness alone. A forecaster or
   allocator that called it would be reading the answer key; both the module
   ban and a call-level ban on the name are enforced in CI.

Structure injected here that the agent is never told about, and has to find:

  * two issuer outages, one two hours long and one forty minutes
  * a festival week with suppressed payment activity
  * a cohort of about 3% whose mandates silently orphan after a card reissue
  * salary clustering on the 1st and the last working day of the month
  * 5% wrong or remapped decline codes, and 3% stale phone numbers

None of it appears in `ObservableState`. The outage is visible only as a burst
of correlated declines, which is exactly what the Sentinel's cohort detector
has to discover at M6.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from zoneinfo import ZoneInfo

import numpy as np

from arc.core.money import Paise, paise
from arc.core.types import ActionType, ClaimType, Rail
from arc.simulator import codes as code_book
from arc.simulator import response_model as rm
from arc.simulator.codes import Semantic
from arc.simulator.seeds import (
    BATCH_DAYS,
    BATCH_START,
    EPOCH,
    HISTORY_START,
    Stream,
    rng,
    stable_hash,
    unit_hash,
)

# Every simulated subject is in India, so subject-local time is one zone. The
# Gate still resolves a timezone basis per subject; the world simply has one.
IST = ZoneInfo("Asia/Kolkata")

DEFAULT_POPULATION = 2_000


# ---------------------------------------------------------------------------
# Injected structure. None of it is exposed in ObservableState.
# ---------------------------------------------------------------------------


class IssuerClass(StrEnum):
    LARGE_PRIVATE = "large_private"
    PSU = "psu"
    SMALL_PRIVATE = "small_private"
    COOPERATIVE = "cooperative"


@dataclass(frozen=True, slots=True)
class Issuer:
    issuer_id: str
    bank_class: IssuerClass
    share: float  # fraction of the population banking here
    base_health: float  # steady-state authorisation health, [0, 1]


# Issuer identifiers are deliberately generic. Attaching a real bank's name to
# a synthetic outage would be an assertion about that bank, and the simulator
# has no evidence for one.
#
# source: the share profile follows the concentration in Indian retail
# payments, where a handful of large private banks carry most authorisation
# volume and a long tail of cooperatives carries very little. That tail is
# what produces genuine INSUFFICIENT_POWER at M6: a cohort detector cannot
# find a signal in eleven transactions, and pretending otherwise is the bug
# the Sentinel is designed around.
ISSUERS: tuple[Issuer, ...] = (
    Issuer("ISS_LP01", IssuerClass.LARGE_PRIVATE, 0.26, 0.97),
    Issuer("ISS_LP02", IssuerClass.LARGE_PRIVATE, 0.21, 0.96),
    Issuer("ISS_LP03", IssuerClass.LARGE_PRIVATE, 0.14, 0.95),
    Issuer("ISS_PS01", IssuerClass.PSU, 0.12, 0.92),
    Issuer("ISS_PS02", IssuerClass.PSU, 0.09, 0.90),
    Issuer("ISS_SP01", IssuerClass.SMALL_PRIVATE, 0.06, 0.93),
    Issuer("ISS_SP02", IssuerClass.SMALL_PRIVATE, 0.045, 0.91),
    Issuer("ISS_SP03", IssuerClass.SMALL_PRIVATE, 0.025, 0.89),
    Issuer("ISS_CO01", IssuerClass.COOPERATIVE, 0.02, 0.86),
    Issuer("ISS_CO02", IssuerClass.COOPERATIVE, 0.012, 0.85),
    Issuer("ISS_CO03", IssuerClass.COOPERATIVE, 0.008, 0.84),
)

ISSUER_BY_ID: Mapping[str, Issuer] = MappingProxyType({i.issuer_id: i for i in ISSUERS})


@dataclass(frozen=True, slots=True)
class Outage:
    """A correlated authorisation collapse at one issuer, for one window.

    Half-open `[start, end)`, like every other window in ARC.
    """

    issuer_id: str
    start: datetime
    end: datetime
    residual_health: float

    def covers(self, moment: datetime) -> bool:
        return self.start <= moment < self.end


# Two outages, as specified: one two hours, one forty minutes. Both land
# inside the batch window and inside Indian business hours, which is when a
# real incident does most of its damage.
#
# source: RBI and PSP outage disclosures, where issuer-side incidents are
# measured in tens of minutes to a few hours and collapse authorisation for
# their duration rather than degrading it gently.
OUTAGES: tuple[Outage, ...] = (
    # 07:00-09:00 IST on 23 October: a large private issuer, two hours, laid
    # across the morning presentation peak. This is the one with enough volume
    # behind it to be found in a single bucket, and it is the demo beat.
    Outage(
        "ISS_LP02",
        datetime(2025, 10, 23, 1, 30, tzinfo=UTC),
        datetime(2025, 10, 23, 3, 30, tzinfo=UTC),
        0.05,
    ),
    # 09:40-10:20 IST on 29 October: a PSU issuer, forty minutes. Deliberately
    # short and on a smaller issuer, so the cell is thin and the Sentinel has
    # to climb its back-off ladder rather than answer from one bucket. An
    # outage that were always easy to see would exercise nothing.
    Outage(
        "ISS_PS01",
        datetime(2025, 10, 29, 4, 10, tzinfo=UTC),
        datetime(2025, 10, 29, 4, 50, tzinfo=UTC),
        0.10,
    ),
)

# A festival week suppresses payment activity: discretionary spend rises and
# balances fall. Diwali 2025 fell on 20 October, so the week around it sits at
# the head of the batch window.
#
# source: seasonal consumption studies of the Indian festive season, which
# report a pronounced spending peak in the Diwali week.
FESTIVAL_START = datetime(2025, 10, 18, 0, 0, tzinfo=UTC)
FESTIVAL_END = datetime(2025, 10, 25, 0, 0, tzinfo=UTC)
FESTIVAL_ABILITY_MULTIPLIER = 0.72

# The silently-orphaning mandate cohort. After a card reissue, the mandate is
# still registered against a token that no longer resolves, and every debit
# returns as though the mandate were missing. The merchant's own view still
# says "active" - that is what makes it silent, and why `mandate_status` in
# ObservableState keeps saying active for these accounts. What IS observable
# is the reissue date, the registration date and an unbroken run of failures;
# the Sentinel has to put those three together at M6.
#
# source: card-on-file reissuance rates reported by networks after the move to
# tokenised credentials.
ORPHANED_MANDATE_RATE = 0.03

# source: contact-data decay studies, which put annual mobile-number churn in
# the low single digits.
STALE_PHONE_RATE = 0.03

# Salary credit dates. The mass on the last working day and the 1st is the
# payday clustering; the spread is what stops a fixed calendar from working.
#
# source: Indian payroll convention, where salary is credited on the last
# working day of the month or in the first days of the next.
LAST_WORKING_DAY = 0
SALARY_DAYS: tuple[int, ...] = (LAST_WORKING_DAY, 1, 5, 7, 10, 15)
SALARY_DAY_WEIGHTS: tuple[float, ...] = (0.34, 0.28, 0.08, 0.12, 0.10, 0.08)


# ---------------------------------------------------------------------------
# Calendar helpers. Pure: they take the moment, they never read a clock.
# ---------------------------------------------------------------------------


def _days_in_month(year: int, month: int) -> int:
    if month == 12:
        return 31
    return (date(year, month + 1, 1) - date(year, month, 1)).days


def last_working_day(year: int, month: int) -> date:
    """Last weekday of the month. Bank holidays belong to the Time Authority,
    which the simulator does not consult - a payroll run is not a settlement."""
    day = date(year, month, _days_in_month(year, month))
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day


def _month_jitter(account_id: str, year: int, month: int, variance: float) -> int:
    """Deterministic per-month salary jitter, in whole days.

    Derived from a stable hash rather than drawn from a generator, so
    `counterfactual()` stays a pure function of (account, action, time). A
    generator draw here would make the ground truth depend on call order.
    """
    spread = int(round(variance))
    if spread <= 0:
        return 0
    offset = unit_hash(account_id, str(year), str(month))
    return int(round((offset * 2.0 - 1.0) * spread))


def salary_date(account_id: str, year: int, month: int, salary_day: int, variance: float) -> date:
    base = (
        last_working_day(year, month)
        if salary_day == LAST_WORKING_DAY
        else date(year, month, min(salary_day, _days_in_month(year, month)))
    )
    return base + timedelta(days=_month_jitter(account_id, year, month, variance))


def days_since_salary(account_id: str, at: datetime, salary_day: int, variance: float) -> float:
    """Days since the most recent salary credit at or before `at`, in local time.

    Looks back through the previous month so a credit on the last working day
    of October is found from the 2nd of November.
    """
    local = at.astimezone(IST).date()
    year, month = local.year, local.month
    for _ in range(3):
        credited = salary_date(account_id, year, month, salary_day, variance)
        if credited <= local:
            return float((local - credited).days)
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return float(BATCH_DAYS)


def in_festival_week(at: datetime) -> bool:
    return FESTIVAL_START <= at < FESTIVAL_END


def issuer_health(issuer_id: str, at: datetime) -> float:
    """Authorisation health of one issuer at one moment, in [0, 1].

    Baseline when nothing is wrong; collapsed for the duration of an outage.
    This is the ONLY place the outages are readable, and nothing that reads it
    is exposed to the agent.
    """
    for outage in OUTAGES:
        if outage.issuer_id == issuer_id and outage.covers(at):
            return outage.residual_health
    issuer = ISSUER_BY_ID.get(issuer_id)
    return issuer.base_health if issuer else 0.90


# ---------------------------------------------------------------------------
# The observability boundary
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LatentState:
    """ARC NEVER SEES THIS.

    `slots=True` is load-bearing rather than cosmetic: it removes `__dict__`,
    so there is no bag of attributes to enumerate and no place for a latent
    field to be attached to something observable later.

    It holds what is true about the PERSON. What is true about the contract
    and the instrument - a closed card, a stale credential, a silently
    orphaned mandate - lives on `Account`, which is equally invisible to the
    agent. Splitting them that way keeps this record readable as a person.

    `issuer_id` is deliberately NOT here even though the specification lists
    it, because the issuer is observable by construction - it is printed on
    the instrument. Carrying it in both records would make the two field sets
    overlap, and the disjointness of those sets is what the boundary test
    checks. It lives on `Account`, and reaches the agent through
    `ObservableState.issuer_id`.
    """

    ability_to_pay: float
    monthly_income_paise: Paise
    salary_day: int
    salary_variance: float
    responsiveness: Mapping[rm.SimChannel, float]
    annoyance_sensitivity: float
    intent_to_churn: float
    promise_reliability: float
    digital_literacy: float
    phone_stale: bool

    def responsiveness_for(self, channel: rm.SimChannel) -> float:
        return self.responsiveness.get(channel, 0.0)


@dataclass(frozen=True, slots=True)
class ObservableState:
    """ARC's ENTIRE view of an account.

    Every field is a scalar, a string, or a flat tuple of those. Nothing here
    holds a reference to an `Account`, a `LatentState` or the `World`, which
    is what makes the object graph reachable from this record latent-free.

    `mandate_status` reports what the merchant's own records say. For the
    orphaned cohort those records say `active`, and they are wrong. Discovering
    that is the Sentinel's job, not this record's.
    """

    account_id: str
    issuer_id: str
    rail: Rail
    plan_value_paise: Paise
    tenure_days: int
    invoice_ageing_bucket: str
    prior_bounces_90d: int
    prior_payment_timestamps: tuple[datetime, ...]
    decline_code_history: tuple[str, ...]
    mac_history: tuple[str, ...]
    mandate_status: str
    mandate_cap_paise: Paise | None
    mandate_registered_at: datetime | None
    instrument_reissued_at: datetime | None
    channel_consent_state: tuple[tuple[str, str], ...]
    contact_history_7d: int
    contact_history_30d: int
    prior_ptp_outcomes: tuple[str, ...]

    def consent_for(self, channel: str) -> str:
        for name, state in self.channel_consent_state:
            if name == channel:
                return state
        return "unknown"


LATENT_FIELD_NAMES: frozenset[str] = frozenset(LatentState.__slots__)
OBSERVABLE_FIELD_NAMES: frozenset[str] = frozenset(ObservableState.__slots__)


# ---------------------------------------------------------------------------
# The account, the events it emits, and the outcomes it produces
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Person:
    """The identifying data a real gateway would carry.

    It exists so the redaction boundary at M5 and the PII write-guard at M2
    have something real to stop. These strings go on the wire and into the
    subject store; nothing derived from them may reach the Decision Ledger.
    """

    name: str
    email: str
    phone: str


@dataclass(frozen=True, slots=True)
class Account:
    """One paying relationship. Holds the latent state, so it is never
    returned to the agent - `observe()` builds an `ObservableState` instead."""

    account_id: str
    customer_ref: str
    person: Person
    issuer_id: str
    rail: Rail
    claim_type: ClaimType
    plan_value_paise: Paise
    opened_at: datetime
    mandate_status: str
    mandate_cap_paise: Paise | None
    mandate_registered_at: datetime | None
    instrument_reissued_at: datetime | None
    # The three instrument-layer faults, each with exactly one repair:
    #   terminal_instrument  lost, stolen or closed  -> nothing repairs it
    #   credential_stale     card reissued           -> card_updater
    #   mandate_orphaned     mandate lost its token  -> mandate_re_register
    # A cap set below the plan value is the fourth, and re-registration or a
    # rail fallback clears that too.
    terminal_instrument: bool
    credential_stale: bool
    mandate_orphaned: bool
    consent: tuple[tuple[str, str], ...]
    latent: LatentState

    prior_payments: tuple[datetime, ...] = ()
    prior_declines: tuple[tuple[datetime, str], ...] = ()
    prior_mac: tuple[str, ...] = ()
    prior_ptp: tuple[str, ...] = ()
    prior_contacts: tuple[datetime, ...] = ()
    invoice_ageing_bucket: str = "current"

    def tenure_days(self, at: datetime) -> int:
        return max(0, (at - self.opened_at).days)


class EventKind(StrEnum):
    """What the gateway is telling us happened."""

    PRESENTATION = "presentation"  # a debit was presented; it captured or failed
    INVOICE_OVERDUE = "invoice_overdue"
    CHECKOUT_ABANDON = "checkout_abandon"


class Initiator(StrEnum):
    """Who presented the debit. The gateway retries on its own schedule, and
    those attempts count against the network cap whether we issued them or not."""

    MERCHANT = "merchant"
    GATEWAY = "gateway"


@dataclass(frozen=True, slots=True)
class BatchEvent:
    """One thing that happened in the world, before any adapter sees it."""

    event_id: str
    kind: EventKind
    account_id: str
    at: datetime
    rail: Rail
    claim_type: ClaimType
    amount_paise: Paise
    succeeded: bool
    attempt: int
    initiated_by: Initiator
    decline_code: str | None = None
    advice_code: str | None = None
    true_semantic: Semantic | None = None

    def as_record(self) -> dict[str, object]:
        """Canonical form for the batch digest. Ground truth is included on
        purpose: a change to the injected structure must change the digest."""
        return {
            "event_id": self.event_id,
            "kind": str(self.kind),
            "account_id": self.account_id,
            "at": self.at.isoformat(),
            "rail": str(self.rail),
            "claim_type": str(self.claim_type),
            "amount_paise": int(self.amount_paise),
            "succeeded": self.succeeded,
            "attempt": self.attempt,
            "initiated_by": str(self.initiated_by),
            "decline_code": self.decline_code,
            "advice_code": self.advice_code,
            "true_semantic": str(self.true_semantic) if self.true_semantic else None,
        }


class PromiseStatus(StrEnum):
    """UNRESOLVED is a distinct answer and is never coerced to BROKEN.

    A promise dated the 20th is neither kept nor broken on the 18th. Coding it
    broken is what biases a promise-to-pay model pessimistic, so the world
    refuses to do it and M7 gets genuinely censored data.
    """

    KEPT = "kept"
    BROKEN = "broken"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Promise:
    made_at: datetime
    due_at: datetime
    amount_paise: Paise


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one action actually produced."""

    kind: rm.OutcomeKind
    at: datetime
    action: ActionType
    paid_paise: Paise
    decline_code: str | None = None
    true_semantic: Semantic | None = None
    wrong_party: bool = False
    promise: Promise | None = None


# ---------------------------------------------------------------------------
# Population generation
# ---------------------------------------------------------------------------

# Synthetic name pools. They exist so a bank narration can contain a real-
# looking name for the PII write-guard to refuse and the redaction boundary to
# strip. No person is represented.
_FIRST_NAMES: tuple[str, ...] = (
    "Aarav",
    "Priya",
    "Rohan",
    "Ananya",
    "Vikram",
    "Meera",
    "Arjun",
    "Kavya",
    "Rahul",
    "Divya",
    "Karthik",
    "Sneha",
    "Imran",
    "Fatima",
    "Joseph",
    "Neha",
    "Sanjay",
    "Pooja",
    "Aditya",
    "Ritu",
)
_LAST_NAMES: tuple[str, ...] = (
    "Sharma",
    "Iyer",
    "Nair",
    "Reddy",
    "Patel",
    "Banerjee",
    "Khan",
    "Gupta",
    "Menon",
    "Desai",
    "Chauhan",
    "Rao",
    "Fernandes",
    "Singh",
    "Joshi",
    "Bose",
)

# Rail mix. source: the payments mix for Indian recurring collections, where
# cards and eNACH carry most subscription debits and UPI Autopay has taken a
# growing share of low-ticket mandates.
_RAIL_WEIGHTS: Mapping[Rail, float] = MappingProxyType(
    {Rail.CARD: 0.38, Rail.ENACH: 0.30, Rail.UPI_AUTOPAY: 0.22, Rail.INVOICE: 0.10}
)

# Claims per subject. source: the long tail of multi-subscription customers -
# most people hold one failing obligation, a minority hold several. This is
# the distribution the stratified arm assignment at M5 balances on, and the
# reason contact budget is contended at subject level rather than claim level.
_CLAIMS_PER_SUBJECT: tuple[int, ...] = (1, 2, 3)
_CLAIMS_PER_SUBJECT_WEIGHTS: tuple[float, ...] = (0.72, 0.20, 0.08)

# source: Indian subscription price points, from low-ticket content plans to
# utility and insurance mandates. Lognormal, because plan value is.
_PLAN_VALUE_LOG_MEAN = 6.9  # exp(6.9) ~ 992 rupees
_PLAN_VALUE_LOG_SIGMA = 0.85

# source: household income distribution for the salaried segment that holds
# recurring mandates. The ratio to plan value is what drives affordability.
_INCOME_LOG_MEAN = 10.85  # exp(10.85) ~ 51,600 rupees a month
_INCOME_LOG_SIGMA = 0.55

_CONSENT_CHANNELS: tuple[str, ...] = ("whatsapp", "sms", "email", "voice")

# source: consent capture rates for transactional channels, where e-mail and
# SMS consent ride on the account opening and WhatsApp and voice are opted
# into separately and less often.
_CONSENT_GRANT_RATE: Mapping[str, float] = MappingProxyType(
    {"whatsapp": 0.62, "sms": 0.88, "email": 0.91, "voice": 0.55}
)

# source: B2B receivables ageing profiles, where most overdue value sits in
# the first thirty days and the tail beyond ninety is small but expensive.
_INVOICE_BUCKETS: tuple[str, ...] = ("current", "1_30", "31_60", "61_90", "90_plus")
_INVOICE_BUCKET_WEIGHTS: tuple[float, ...] = (0.18, 0.42, 0.22, 0.11, 0.07)

# source: card-on-file reissuance following network tokenisation mandates.
_REISSUE_RATE = 0.11

# source: the hard-decline share of the card return population - lost, stolen,
# closed and stop-payment categories that must never be retried. Around a
# tenth of declines, and declines are a sixth of presentations, which puts the
# share of the portfolio at under two percent.
_TERMINAL_INSTRUMENT_RATE = 0.018

# Of the reissued cards that did not orphan their mandate, the share whose
# stored credential is now out of date. source: account-updater hit-rate
# reporting, where a material fraction of stored credentials go stale between
# reissue and the next presentation.
_CREDENTIAL_STALE_RATE = 0.15

# Share of reissued mandate-rail accounts that orphan. Scaled by the reissue
# rate and by how much of the population is on a mandate rail at all, so the
# cohort comes out at ORPHANED_MANDATE_RATE of the WHOLE population - which is
# the denominator the specification names.
_MANDATE_RAIL_SHARE = 0.52  # eNACH plus UPI Autopay, per _RAIL_WEIGHTS
_ORPHAN_GIVEN_REISSUE = ORPHANED_MANDATE_RATE / (_REISSUE_RATE * _MANDATE_RAIL_SHARE)


def _weighted_choice(generator: np.random.Generator, values: Sequence, weights: Sequence) -> object:
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return values[int(generator.choice(len(values), p=probabilities))]


def _person(generator: np.random.Generator, index: int) -> Person:
    first = _FIRST_NAMES[int(generator.integers(len(_FIRST_NAMES)))]
    last = _LAST_NAMES[int(generator.integers(len(_LAST_NAMES)))]
    # A ten-digit mobile starting 6-9, which is what the PII detector matches.
    phone = f"+91{int(generator.integers(6, 10))}{int(generator.integers(10**8, 10**9)):09d}"
    return Person(
        name=f"{first} {last}",
        email=f"{first.lower()}.{last.lower()}{index % 997}@example.com",
        phone=phone,
    )


def _latent_state(generator: np.random.Generator, account_id: str) -> LatentState:
    """Draw one account's hidden traits.

    The two that decide the result are `annoyance_sensitivity` and
    `salary_day`. Everything else colours the world; those two make it a
    problem worth solving.
    """
    responsiveness = {
        rm.SimChannel.WHATSAPP: float(generator.beta(2.2, 3.4)),
        rm.SimChannel.SMS: float(generator.beta(1.8, 4.0)),
        rm.SimChannel.EMAIL: float(generator.beta(1.6, 4.4)),
        rm.SimChannel.PAYMENT_LINK: float(generator.beta(2.4, 3.2)),
        rm.SimChannel.VOICE: float(generator.beta(2.6, 3.0)),
        rm.SimChannel.HUMAN: float(generator.beta(3.0, 2.6)),
        rm.SimChannel.POSTAL: float(generator.beta(1.3, 5.0)),
        rm.SimChannel.RAIL: 0.0,
        rm.SimChannel.NONE: 0.0,
    }
    salary_day = int(
        _weighted_choice(generator, SALARY_DAYS, SALARY_DAY_WEIGHTS)  # type: ignore[arg-type]
    )
    return LatentState(
        ability_to_pay=float(generator.beta(3.2, 2.4)),
        monthly_income_paise=paise(
            int(round(float(generator.lognormal(_INCOME_LOG_MEAN, _INCOME_LOG_SIGMA)) * 100))
        ),
        salary_day=salary_day,
        salary_variance=float(generator.gamma(1.6, 0.9)),
        responsiveness=MappingProxyType(responsiveness),
        annoyance_sensitivity=float(generator.beta(2.0, 3.0)),
        intent_to_churn=float(generator.beta(1.7, 5.0)),
        promise_reliability=float(generator.beta(3.0, 2.2)),
        digital_literacy=float(generator.beta(3.4, 2.0)),
        phone_stale=bool(generator.random() < STALE_PHONE_RATE),
    )


def _consent(
    generator: np.random.Generator, digital_literacy: float
) -> tuple[tuple[str, str], ...]:
    """Per-channel consent, correlated with digital literacy.

    Absent is never granted: a channel the record does not mention comes back
    as `unknown`, and the Gate fails that closed.
    """
    states: list[tuple[str, str]] = []
    for channel in _CONSENT_CHANNELS:
        rate = _CONSENT_GRANT_RATE[channel] * (0.75 + 0.35 * digital_literacy)
        draw = float(generator.random())
        if draw < min(rate, 0.98):
            states.append((channel, "granted"))
        elif draw < min(rate, 0.98) + 0.06:
            states.append((channel, "withdrawn"))
        else:
            states.append((channel, "never_given"))
    return tuple(states)


def _payment_history(
    generator: np.random.Generator, account_id: str, latent: LatentState
) -> tuple[tuple[datetime, ...], int]:
    """Successful payments over the observable history window.

    Timestamps cluster a day or two after the salary credit, which is how the
    latent salary day leaks into observables. A learner can recover it; a
    fixed T+1/T+3/T+7 calendar cannot use it. That gap is the whole reason
    arm B exists to be beaten.
    """
    payments: list[datetime] = []
    misses = 0
    local_epoch = EPOCH.astimezone(IST)
    for back in (3, 2, 1):
        month = local_epoch.month - back
        year = local_epoch.year
        while month <= 0:
            month += 12
            year -= 1
        credited = salary_date(account_id, year, month, latent.salary_day, latent.salary_variance)
        if float(generator.random()) < 0.35 + 0.55 * latent.ability_to_pay:
            offset_days = int(generator.integers(0, 3))
            hour = int(generator.integers(6, 22))
            minute = int(generator.integers(0, 60))
            moment = datetime(
                credited.year, credited.month, credited.day, hour, minute, tzinfo=IST
            ) + timedelta(days=offset_days)
            if HISTORY_START <= moment.astimezone(UTC) < BATCH_START:
                payments.append(moment.astimezone(UTC))
        else:
            misses += 1
    return tuple(sorted(payments)), misses


def _decline_history(
    generator: np.random.Generator,
    rail: Rail,
    misses: int,
    terminal_instrument: bool,
) -> tuple[tuple[tuple[datetime, str], ...], tuple[str, ...]]:
    """Declines and merchant advice codes seen before the batch window."""
    declines: list[tuple[datetime, str]] = []
    advice: list[str] = []
    for _ in range(misses):
        semantic = Semantic.HARD_DECLINE if terminal_instrument else Semantic.INSUFFICIENT_FUNDS
        code = code_book.emit_code(rail, semantic, generator)
        if code is None:
            continue
        offset = int(generator.integers(0, (BATCH_START - HISTORY_START).days))
        moment = HISTORY_START + timedelta(days=offset, hours=int(generator.integers(0, 24)))
        declines.append((moment, code))
        if terminal_instrument and float(generator.random()) < 0.5:
            advice.append("MAC03")
        elif float(generator.random()) < 0.22:
            advice.append("MAC02")
    return tuple(sorted(declines)), tuple(advice)


def _promise_history(generator: np.random.Generator, latent: LatentState) -> tuple[str, ...]:
    """Prior promise outcomes, including unresolved ones.

    An unresolved promise stays unresolved in the record. Recording it as
    broken would hand M7 a pre-biased label set.
    """
    count = int(generator.integers(0, 3))
    outcomes: list[str] = []
    for _ in range(count):
        draw = float(generator.random())
        if draw < 0.08:
            outcomes.append(str(PromiseStatus.UNRESOLVED))
        elif draw < 0.08 + 0.92 * latent.promise_reliability:
            outcomes.append(str(PromiseStatus.KEPT))
        else:
            outcomes.append(str(PromiseStatus.BROKEN))
    return tuple(outcomes)


_CLAIM_TYPE_BY_RAIL: Mapping[Rail, ClaimType] = MappingProxyType(
    {
        Rail.CARD: ClaimType.CARD_DECLINE,
        Rail.ENACH: ClaimType.MANDATE_FAILURE,
        Rail.UPI_AUTOPAY: ClaimType.MANDATE_FAILURE,
        Rail.INVOICE: ClaimType.INVOICE_OVERDUE,
    }
)

# A slice of the card population abandoned a checkout rather than failing a
# debit. Same object, different leak surface - which is the point of
# normalising all four onto one claim type at M5.
# source: published checkout-abandonment rates for Indian e-commerce, taken as
# a share of the card-rail population rather than of sessions.
_CHECKOUT_ABANDON_RATE = 0.12


def _build_account(generator: np.random.Generator, index: int, customer_ref: str) -> Account:
    account_id = f"acct_{index:07d}"
    latent = _latent_state(generator, account_id)

    issuer = _weighted_choice(generator, ISSUERS, [i.share for i in ISSUERS])
    rail = _weighted_choice(generator, list(_RAIL_WEIGHTS), list(_RAIL_WEIGHTS.values()))
    claim_type = _CLAIM_TYPE_BY_RAIL[rail]
    if rail is Rail.CARD and float(generator.random()) < _CHECKOUT_ABANDON_RATE:
        claim_type = ClaimType.CHECKOUT_ABANDON

    plan_value = paise(
        int(round(float(generator.lognormal(_PLAN_VALUE_LOG_MEAN, _PLAN_VALUE_LOG_SIGMA)) * 100))
    )
    opened_at = EPOCH - timedelta(days=int(generator.integers(30, 1400)))

    mandate_status = "none"
    mandate_cap: Paise | None = None
    mandate_registered_at: datetime | None = None
    if rail in (Rail.ENACH, Rail.UPI_AUTOPAY, Rail.CARD):
        mandate_status = "active"
        # A cap set close to the plan value is how a price rise turns into a
        # merchant-layer failure that no amount of customer contact fixes.
        # A cap set below the plan value is a merchant-layer failure no
        # amount of customer contact fixes. The lower bound puts about one
        # percent of mandates there, which is what a price rise looks like.
        multiplier = float(generator.uniform(0.98, 3.0))
        mandate_cap = paise(int(round(plan_value * multiplier)))
        mandate_registered_at = opened_at + timedelta(days=int(generator.integers(0, 10)))

    instrument_reissued_at: datetime | None = None
    orphaned = False
    credential_stale = False
    if mandate_registered_at is not None and float(generator.random()) < _REISSUE_RATE:
        span = max((EPOCH - mandate_registered_at).days - 1, 1)
        instrument_reissued_at = mandate_registered_at + timedelta(
            days=int(generator.integers(1, span + 1))
        )
        # Of the reissued, this share orphan silently, giving the overall
        # ORPHANED_MANDATE_RATE across the population. The rest merely carry a
        # stale stored credential, which the card updater can refresh.
        # A mandate orphans on the rails that hold one. A card has no mandate
        # of its own - what goes wrong there is a stale stored credential,
        # which is the account updater's job rather than re-registration.
        if rail in (Rail.ENACH, Rail.UPI_AUTOPAY):
            orphaned = float(generator.random()) < _ORPHAN_GIVEN_REISSUE
        elif rail is Rail.CARD:
            credential_stale = bool(generator.random() < _CREDENTIAL_STALE_RATE)

    terminal_instrument = bool(generator.random() < _TERMINAL_INSTRUMENT_RATE)
    payments, misses = _payment_history(generator, account_id, latent)
    declines, advice = _decline_history(generator, rail, misses, terminal_instrument)

    ageing = "current"
    if rail is Rail.INVOICE:
        ageing = str(_weighted_choice(generator, _INVOICE_BUCKETS, _INVOICE_BUCKET_WEIGHTS))

    prior_contacts: list[datetime] = []
    for _ in range(int(generator.integers(0, 4))):
        prior_contacts.append(
            EPOCH
            - timedelta(days=int(generator.integers(1, 30)), hours=int(generator.integers(24)))
        )

    return Account(
        account_id=account_id,
        customer_ref=customer_ref,
        person=_person(generator, index),
        issuer_id=issuer.issuer_id,
        rail=rail,
        claim_type=claim_type,
        plan_value_paise=plan_value,
        opened_at=opened_at,
        mandate_status=mandate_status,
        mandate_cap_paise=mandate_cap,
        mandate_registered_at=mandate_registered_at,
        instrument_reissued_at=instrument_reissued_at,
        terminal_instrument=terminal_instrument,
        credential_stale=credential_stale,
        mandate_orphaned=orphaned,
        consent=_consent(generator, latent.digital_literacy),
        latent=latent,
        prior_payments=payments,
        prior_declines=declines,
        prior_mac=advice,
        prior_ptp=_promise_history(generator, latent),
        prior_contacts=tuple(sorted(prior_contacts)),
        invoice_ageing_bucket=ageing,
    )


def build_population(seed: int, size: int) -> tuple[Account, ...]:
    """Generate `size` accounts, grouped into subjects.

    Accounts belonging to one subject share a `customer_ref`, which is what
    lets the normaliser at M5 collapse them onto one subject token and assign
    a single experiment arm to all of them. Randomising below that grouping
    would violate SUTVA under a shared contact budget.
    """
    if size <= 0:
        raise ValueError(f"population size must be positive, got {size}")

    generator = rng(seed, Stream.POPULATION)
    accounts: list[Account] = []
    subject_index = 0
    while len(accounts) < size:
        subject_index += 1
        customer_ref = f"cust_{subject_index:07d}"
        held = int(_weighted_choice(generator, _CLAIMS_PER_SUBJECT, _CLAIMS_PER_SUBJECT_WEIGHTS))
        for _ in range(held):
            if len(accounts) >= size:
                break
            accounts.append(_build_account(generator, len(accounts) + 1, customer_ref))
    return tuple(accounts)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------

# Preconditions, not response terms. A debit that cannot be presented is not a
# probability question, so these gate whether the seven-term model is consulted
# at all rather than adding an eighth term to it. They are also the source of
# the merchant-layer SELF_HEALING path: an orphaned mandate is repaired at the
# rail, with zero customer contact.
_REPAIR_ACTIONS: Mapping[ActionType, str] = MappingProxyType(
    {
        ActionType.MANDATE_RE_REGISTER: "mandate",
        ActionType.RAIL_FALLBACK: "mandate",
        ActionType.CARD_UPDATER: "instrument",
    }
)

# source: card-updater hit rates published by the networks - a reissued card
# is usually recoverable, a closed account never is.
_CARD_UPDATER_HIT_RATE = 0.62

# Presentation hours, IST. Recurring debit files are submitted in the morning
# and card retries run through the working day.
# source: NACH settlement cycles and PSP retry scheduling practice.
_PRESENTATION_HOUR_WEIGHTS: tuple[float, ...] = (
    0.005,
    0.005,
    0.005,
    0.01,
    0.02,
    0.05,
    0.09,
    0.10,
    0.09,
    0.085,
    0.08,
    0.075,
    0.06,
    0.055,
    0.05,
    0.045,
    0.04,
    0.035,
    0.03,
    0.02,
    0.015,
    0.01,
    0.008,
    0.007,
)

# source: PSP smart-retry behaviour - the gateway re-presents a failed debit
# on its own schedule, and those attempts count against the network cap
# whether or not we issued them.
_GATEWAY_RETRY_RATE = 0.35

# Invoices are B2B and an order of magnitude larger than a consumer plan.
# source: MSME invoice ticket sizes against consumer subscription price
# points; the range spans a small services invoice to a goods order.
_INVOICE_MULTIPLIER_LOW = 3.0
_INVOICE_MULTIPLIER_HIGH = 20.0


# A FIRST PRESENTATION IS NOT A RETRY, and must not be scored as one. The
# seven-term response model answers "given this obligation has already failed,
# what does this action recover" - a population conditioned on failure. The
# scheduled debit that creates the failure is drawn from every account, so it
# has its own, much higher success curve.
#
# source: NPCI monthly NACH debit statistics, where roughly two thirds to
# seven tenths of presented debits settle on first presentation, and card
# recurring authorisation rates, which run materially higher than that. The
# rail offsets carry that difference. p2 is large for the same reason b4 is:
# an issuer incident collapses authorisation rather than denting it.
_PRESENT_BASE = -3.45
_PRESENT_ABILITY = 1.40
_PRESENT_ISSUER_HEALTH = 3.60
_PRESENT_AFFORDABILITY = 0.70
_PRESENT_RAIL_OFFSET: Mapping[Rail, float] = MappingProxyType(
    {Rail.CARD: 1.25, Rail.UPI_AUTOPAY: 0.45, Rail.ENACH: 0.0, Rail.INVOICE: 0.0}
)


def presentation_success_probability(inputs: rm.ResponseInputs, rail: Rail) -> float:
    """P(a scheduled debit settles). The batch's failures are its complement,
    and its successes are the denominator the cohort detector needs."""
    return rm.sigmoid(
        _PRESENT_BASE
        + _PRESENT_ABILITY * inputs.ability_to_pay
        + _PRESENT_ISSUER_HEALTH * inputs.issuer_health
        + _PRESENT_AFFORDABILITY * inputs.affordability
        + _PRESENT_RAIL_OFFSET[rail]
    )


class World:
    """The simulated population and its ground truth.

    `observe()` is the only public read of an account. `counterfactual()` is
    ground truth and belongs to the evaluation harness alone - a forecaster or
    allocator calling it is reading the answer key, which CI refuses by name.
    """

    def __init__(
        self,
        *,
        seed: int,
        size: int = DEFAULT_POPULATION,
        epoch: datetime = EPOCH,
    ) -> None:
        self._seed = seed
        self._size = size
        self._epoch = epoch
        self._accounts: tuple[Account, ...] = build_population(seed, size)
        self._by_id: Mapping[str, Account] = MappingProxyType(
            {account.account_id: account for account in self._accounts}
        )
        # Interaction state. Mutated as the policy acts, which is what makes
        # the sleeping-dog term bite: contacts accumulate and the world reacts.
        self._contacts: dict[str, list[datetime]] = {}
        self._attempts: dict[str, int] = {}
        self._repaired: dict[str, set[str]] = {}
        self._batch: tuple[BatchEvent, ...] | None = None

    # -- population ------------------------------------------------------

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def epoch(self) -> datetime:
        return self._epoch

    @property
    def account_ids(self) -> tuple[str, ...]:
        return tuple(account.account_id for account in self._accounts)

    def customer_ref(self, account_id: str) -> str:
        """The gateway's own customer identifier. Accounts sharing one belong
        to a single subject, and therefore to a single experiment arm."""
        return self._account(account_id).customer_ref

    def _account(self, account_id: str) -> Account:
        try:
            return self._by_id[account_id]
        except KeyError:
            raise KeyError(f"unknown account {account_id!r}") from None

    def _latent(self, account_id: str) -> LatentState:
        """PRIVATE. Test and generator use only, never the agent path.

        Kept as a named method rather than a public attribute so that any
        reader of the agent path can see there is exactly one door and that it
        is marked.
        """
        return self._account(account_id).latent

    def fork(self) -> World:
        """A fresh world over the same population, with no interaction history.

        The five arms at M11 run on the same batch. If they shared one world,
        arm B's contacts would raise arm E's annoyance and the comparison
        would be between arms plus contamination.
        """
        twin = object.__new__(World)
        twin._seed = self._seed
        twin._size = self._size
        twin._epoch = self._epoch
        twin._accounts = self._accounts
        twin._by_id = self._by_id
        twin._contacts = {}
        twin._attempts = {}
        twin._repaired = {}
        twin._batch = self._batch
        return twin

    # -- the observability boundary ---------------------------------------

    def observe(self, account_id: str, at: datetime) -> ObservableState:
        """ARC's entire view of one account at one moment.

        Takes `at` rather than reading a clock, because nothing in this repo
        reads a clock except the Time Authority, and because a view of the
        world that depends on when you asked cannot be replayed.

        The returned record holds no reference to the account, the latent
        state or the world: every field is a scalar or a flat tuple of them.
        """
        account = self._account(account_id)
        contacts = self._all_contacts(account)
        declines = [entry for entry in account.prior_declines if entry[0] < at]

        return ObservableState(
            account_id=account.account_id,
            issuer_id=account.issuer_id,
            rail=account.rail,
            plan_value_paise=account.plan_value_paise,
            tenure_days=account.tenure_days(at),
            invoice_ageing_bucket=account.invoice_ageing_bucket,
            prior_bounces_90d=len(declines),
            prior_payment_timestamps=tuple(t for t in account.prior_payments if t < at),
            decline_code_history=tuple(code for _, code in declines),
            mac_history=account.prior_mac,
            mandate_status=account.mandate_status,
            mandate_cap_paise=account.mandate_cap_paise,
            mandate_registered_at=account.mandate_registered_at,
            instrument_reissued_at=account.instrument_reissued_at,
            channel_consent_state=account.consent,
            contact_history_7d=_count_within(contacts, at, timedelta(days=7)),
            contact_history_30d=_count_within(contacts, at, timedelta(days=30)),
            prior_ptp_outcomes=account.prior_ptp,
        )

    def _all_contacts(self, account: Account) -> list[datetime]:
        return list(account.prior_contacts) + self._contacts.get(account.account_id, [])

    def contacts_7d(self, account_id: str, at: datetime) -> int:
        return _count_within(self._all_contacts(self._account(account_id)), at, timedelta(days=7))

    # -- ground truth ------------------------------------------------------

    def _presentation(
        self, account: Account, action: ActionType, at: datetime
    ) -> tuple[float, Semantic | None]:
        """Probability the debit reaches the issuer, and why it would not.

        Preconditions, not response terms. A debit that cannot be presented is
        not a probability question about the customer, so these gate whether
        the seven-term model is consulted at all rather than adding an eighth
        term to it. They are also where the merchant-layer SELF_HEALING path
        comes from: an orphaned mandate is repaired at the rail, and the
        customer is never contacted.

        The blocking semantic is what a truthful gateway would return. It is
        attributable to the instrument or to our own setup, never to the
        person, which is the distinction the Sentinel exists to make.
        """
        if action not in rm.DEBIT_ACTIONS:
            return 1.0, None

        repaired = self._repaired.get(account.account_id, frozenset())
        repairs = _REPAIR_ACTIONS.get(action)

        # Nothing repairs a closed, lost or stolen instrument. Retrying one is
        # exactly the network-punished behaviour the Gate blocks permanently.
        if account.terminal_instrument and "instrument" not in repaired:
            return 0.0, Semantic.HARD_DECLINE

        if account.mandate_orphaned and "mandate" not in repaired and repairs != "mandate":
            return 0.0, Semantic.MANDATE_MISSING

        if account.credential_stale and "instrument" not in repaired:
            if repairs != "instrument":
                return 0.0, Semantic.CARD_EXPIRED
            # The account updater does not always find the new credential.
            return _CARD_UPDATER_HIT_RATE, None

        cap = account.mandate_cap_paise
        over_cap = cap is not None and account.plan_value_paise > cap
        if over_cap and "mandate" not in repaired and repairs != "mandate":
            return 0.0, Semantic.MANDATE_CAP

        return 1.0, None

    def _response_inputs(
        self, account: Account, action: ActionType, at: datetime, contacts_7d: int
    ) -> rm.ResponseInputs:
        latent = account.latent
        channel = rm.ACTION_CHANNEL[action]

        responsiveness = latent.responsiveness_for(channel)
        # A stale number reaches nobody. The message is delivered to whoever
        # holds it now, which is a wrong-party contact and worth nothing.
        if latent.phone_stale and channel in (rm.SimChannel.VOICE, rm.SimChannel.SMS):
            responsiveness = 0.0

        since = days_since_salary(account.account_id, at, latent.salary_day, latent.salary_variance)
        ability = latent.ability_to_pay * rm.funds_cycle(since)
        if in_festival_week(at):
            ability *= FESTIVAL_ABILITY_MULTIPLIER

        return rm.ResponseInputs(
            ability_to_pay=min(ability, 1.0),
            responsiveness=responsiveness,
            timing_fit=rm.timing_fit(since),
            issuer_health=issuer_health(account.issuer_id, at),
            annoyance_sensitivity=latent.annoyance_sensitivity,
            contacts_7d=contacts_7d + (1 if rm.is_contact(action) else 0),
            friction=rm.friction_of(action),
            affordability=rm.affordability(
                int(account.plan_value_paise), int(latent.monthly_income_paise)
            ),
        )

    def counterfactual(self, account_id: str, action: ActionType, at: datetime) -> float:
        """GROUND TRUTH P(pay). EVALUATION HARNESS ONLY.

        This is the answer key. It validates the doubly-robust estimator at
        M11 against the truth the estimator is trying to recover, and it is
        the only honest way to report the estimator's own error.

        Nothing in `arc/forecaster/`, `arc/allocator/`, `arc/sentinel/` or
        `arc/gate/` may call it. That is enforced by name in CI, not by
        convention, because a policy that reads the answer key produces a
        headline number that measures nothing.

        Pure: no sampling, no recording, no advance of world state.
        """
        account = self._account(account_id)
        reachable, _ = self._presentation(account, action, at)
        if reachable <= 0.0:
            return 0.0

        contacts = self.contacts_7d(account_id, at)
        return reachable * rm.p_pay(self._response_inputs(account, action, at, contacts))

    # -- sampling reality --------------------------------------------------

    def outcome(
        self,
        account_id: str,
        action: ActionType,
        at: datetime,
        generator: np.random.Generator,
    ) -> Outcome:
        """Take one action and sample what the world does about it.

        Unlike `counterfactual`, this ADVANCES the world: the contact is
        recorded, the attempt counter moves, a repair sticks. That is what
        makes the sleeping-dog term bite - annoyance accumulates because the
        policy accumulated it.
        """
        account = self._account(account_id)
        contacts = self.contacts_7d(account_id, at)
        attempts = self._attempts.get(account_id, 0)

        if rm.is_contact(action):
            self._contacts.setdefault(account_id, []).append(at)
        if action in rm.DEBIT_ACTIONS:
            self._attempts[account_id] = attempts + 1

        if action is ActionType.DO_NOTHING:
            reachable, blocking = 1.0, None
        else:
            reachable, blocking = self._presentation(account, action, at)

        repairs = _REPAIR_ACTIONS.get(action)
        if repairs is not None and blocking is None and float(generator.random()) < reachable:
            self._repaired.setdefault(account_id, set()).add(repairs)

        # A stale number reaches whoever holds it now. Nothing is disclosed,
        # nothing is recovered, and the attempt is still a contact.
        channel = rm.ACTION_CHANNEL[action]
        wrong_party = account.latent.phone_stale and channel in (
            rm.SimChannel.VOICE,
            rm.SimChannel.SMS,
        )

        if blocking is not None or float(generator.random()) >= reachable:
            semantic = blocking if blocking is not None else Semantic.TECHNICAL
            return Outcome(
                kind=rm.OutcomeKind.NO_RESPONSE,
                at=at,
                action=action,
                paid_paise=paise(0),
                decline_code=code_book.emit_code(account.rail, semantic, generator),
                true_semantic=semantic,
                wrong_party=wrong_party,
            )

        inputs = self._response_inputs(account, action, at, contacts)
        hazards = rm.harm_hazards(
            action=action,
            annoyance_sensitivity=account.latent.annoyance_sensitivity,
            intent_to_churn=account.latent.intent_to_churn,
            contacts_7d=inputs.contacts_7d,
            prior_attempts=attempts,
        )
        kind = rm.sample_outcome(rm.p_pay(inputs), hazards, generator)

        if kind is rm.OutcomeKind.PAID:
            return Outcome(
                kind=kind,
                at=at,
                action=action,
                paid_paise=account.plan_value_paise,
            )

        semantic = self._failure_semantic(account, inputs, at, generator)
        promise = None
        if kind is rm.OutcomeKind.NO_RESPONSE and not wrong_party:
            promise = self._maybe_promise(account, action, at, generator)

        return Outcome(
            kind=kind,
            at=at,
            action=action,
            paid_paise=paise(0),
            decline_code=(
                code_book.emit_code(account.rail, semantic, generator)
                if action in rm.DEBIT_ACTIONS
                else None
            ),
            true_semantic=semantic if action in rm.DEBIT_ACTIONS else None,
            wrong_party=wrong_party,
            promise=promise,
        )

    def _failure_semantic(
        self,
        account: Account,
        inputs: rm.ResponseInputs,
        at: datetime,
        generator: np.random.Generator,
    ) -> Semantic:
        """Why a presentable debit still failed.

        Issuer health first, because a systemic failure is not a delinquent
        customer. That ordering is the same reason the Sentinel checks cohorts
        before it opens the code map.
        """
        if inputs.issuer_health < 0.5:
            return Semantic.ISSUER_UNAVAILABLE
        # source: NPCI NACH return-reason mix and card decline-reason
        # reporting, both dominated by funds-insufficient, with technical and
        # risk-driven declines making up most of the remainder.
        draw = float(generator.random())
        if draw < 0.80:
            return Semantic.INSUFFICIENT_FUNDS
        if draw < 0.89:
            return Semantic.TECHNICAL
        if draw < 0.95:
            return Semantic.RISK_DECLINE
        return Semantic.DO_NOT_RETRY

    def _maybe_promise(
        self,
        account: Account,
        action: ActionType,
        at: datetime,
        generator: np.random.Generator,
    ) -> Promise | None:
        """A promise needs a conversation, so only a call produces one."""
        if rm.ACTION_CHANNEL[action] not in rm.PROMISE_CHANNELS:
            return None
        latent = account.latent
        engaged = latent.responsiveness_for(rm.ACTION_CHANNEL[action])
        if float(generator.random()) >= rm.PROMISE_ELICIT_RATE * (0.4 + engaged):
            return None

        # People promise for the day they expect to be paid, which is why the
        # gap between the promise date and the salary day predicts whether it
        # is kept - and why M7 finds that feature without being told.
        since = days_since_salary(account.account_id, at, latent.salary_day, latent.salary_variance)
        days_to_next = max(1, int(round(30 - since)))
        horizon = min(days_to_next + int(generator.integers(-2, 3)), 21)
        return Promise(
            made_at=at,
            due_at=at + timedelta(days=max(horizon, 1)),
            amount_paise=account.plan_value_paise,
        )

    def resolve_promise(
        self,
        account_id: str,
        promise: Promise,
        at: datetime,
        generator: np.random.Generator,
    ) -> PromiseStatus:
        """KEPT, BROKEN, or UNRESOLVED because the date has not arrived.

        A promise dated the 20th is neither kept nor broken on the 18th.
        Returning BROKEN there is what biases a promise model pessimistic, so
        the world refuses to and M7 gets genuinely censored labels.
        """
        if at < promise.due_at:
            return PromiseStatus.UNRESOLVED

        account = self._account(account_id)
        latent = account.latent
        since = days_since_salary(
            account.account_id, promise.due_at, latent.salary_day, latent.salary_variance
        )
        # A promise dated just before the salary credit is materially less
        # likely to be kept than one dated just after it.
        alignment = rm.timing_fit(since)
        kept = latent.promise_reliability * (0.55 + 0.45 * alignment)
        return PromiseStatus.KEPT if float(generator.random()) < kept else PromiseStatus.BROKEN

    # -- the batch ---------------------------------------------------------

    def batch_events(self) -> tuple[BatchEvent, ...]:
        """Everything that happened in the batch window, in event-time order.

        Both captures and failures. The cohort detector at M6 needs the
        denominator: a burst of declines means nothing without the volume it
        was drawn from, and a system that only sees failures cannot tell a
        busy hour from a broken issuer.

        Deterministic and cached, so `same seed, same batch` is a property of
        the world rather than of how many times it was asked.
        """
        if self._batch is None:
            self._batch = self._generate_batch()
        return self._batch

    def _generate_batch(self) -> tuple[BatchEvent, ...]:
        generator = rng(self._seed, Stream.FAILURES)
        hours = np.arange(24)
        hour_weights = np.asarray(_PRESENTATION_HOUR_WEIGHTS, dtype=float)
        hour_weights = hour_weights / hour_weights.sum()

        events: list[BatchEvent] = []
        for account in self._accounts:
            if account.rail is Rail.INVOICE:
                events.append(self._invoice_event(account, generator))
                continue
            if account.claim_type is ClaimType.CHECKOUT_ABANDON:
                events.append(self._checkout_event(account, generator, hours, hour_weights))
                continue
            events.extend(self._presentation_events(account, generator, hours, hour_weights))

        events.sort(key=lambda event: (event.at, event.event_id))
        return tuple(events)

    def _moment(
        self,
        generator: np.random.Generator,
        hours: np.ndarray,
        hour_weights: np.ndarray,
    ) -> datetime:
        """A presentation time inside the batch window, clustered in the
        working day. Drawn in IST because that is when the file is submitted."""
        day = int(generator.integers(0, BATCH_DAYS))
        hour = int(generator.choice(hours, p=hour_weights))
        minute = int(generator.integers(0, 60))
        local = BATCH_START.astimezone(IST).replace(hour=0, minute=0, second=0, microsecond=0)
        return (local + timedelta(days=day, hours=hour, minutes=minute)).astimezone(UTC)

    def _sample_presentation(
        self, account: Account, at: datetime, generator: np.random.Generator
    ) -> tuple[bool, Semantic | None]:
        """Did this debit capture, and if not, what truthfully went wrong."""
        reachable, blocking = self._presentation(account, ActionType.RETRY, at)
        if blocking is not None or float(generator.random()) >= reachable:
            return False, blocking or Semantic.TECHNICAL

        inputs = self._response_inputs(account, ActionType.RETRY, at, 0)
        if float(generator.random()) < presentation_success_probability(inputs, account.rail):
            return True, None
        return False, self._failure_semantic(account, inputs, at, generator)

    def _presentation_events(
        self,
        account: Account,
        generator: np.random.Generator,
        hours: np.ndarray,
        hour_weights: np.ndarray,
    ) -> list[BatchEvent]:
        at = self._moment(generator, hours, hour_weights)
        succeeded, semantic = self._sample_presentation(account, at, generator)
        amount = paise(int(round(account.plan_value_paise * float(generator.uniform(0.98, 1.02)))))
        events = [
            self._debit_event(
                account, at, amount, succeeded, semantic, 1, Initiator.MERCHANT, generator
            )
        ]

        # The gateway re-presents on its own schedule. Those attempts count
        # against the network cap whether or not we issued them, which is why
        # they are in the batch rather than invented later.
        if not succeeded and float(generator.random()) < _GATEWAY_RETRY_RATE:
            later = at + timedelta(hours=int(generator.integers(20, 40)))
            if later < self._epoch:
                retry_ok, retry_semantic = self._sample_presentation(account, later, generator)
                events.append(
                    self._debit_event(
                        account,
                        later,
                        amount,
                        retry_ok,
                        retry_semantic,
                        2,
                        Initiator.GATEWAY,
                        generator,
                    )
                )
        return events

    def _debit_event(
        self,
        account: Account,
        at: datetime,
        amount: Paise,
        succeeded: bool,
        semantic: Semantic | None,
        attempt: int,
        initiated_by: Initiator,
        generator: np.random.Generator,
    ) -> BatchEvent:
        code = None
        if not succeeded and semantic is not None:
            code = code_book.emit_code(account.rail, semantic, generator)

        advice = None
        if not succeeded and semantic in (Semantic.HARD_DECLINE, Semantic.DO_NOT_RETRY):
            advice = "MAC03"
        elif not succeeded and float(generator.random()) < 0.18:
            advice = "MAC02"

        return BatchEvent(
            event_id=_event_id(account.account_id, at, attempt),
            kind=EventKind.PRESENTATION,
            account_id=account.account_id,
            at=at,
            rail=account.rail,
            claim_type=account.claim_type,
            amount_paise=amount,
            succeeded=succeeded,
            attempt=attempt,
            initiated_by=initiated_by,
            decline_code=code,
            advice_code=advice,
            true_semantic=None if succeeded else semantic,
        )

    def _invoice_event(self, account: Account, generator: np.random.Generator) -> BatchEvent:
        at = BATCH_START + timedelta(
            days=int(generator.integers(0, BATCH_DAYS)), hours=int(generator.integers(0, 24))
        )
        multiplier = float(generator.uniform(_INVOICE_MULTIPLIER_LOW, _INVOICE_MULTIPLIER_HIGH))
        return BatchEvent(
            event_id=_event_id(account.account_id, at, 1),
            kind=EventKind.INVOICE_OVERDUE,
            account_id=account.account_id,
            at=at,
            rail=Rail.INVOICE,
            claim_type=ClaimType.INVOICE_OVERDUE,
            amount_paise=paise(int(round(account.plan_value_paise * multiplier))),
            succeeded=False,
            attempt=1,
            initiated_by=Initiator.MERCHANT,
        )

    def _checkout_event(
        self,
        account: Account,
        generator: np.random.Generator,
        hours: np.ndarray,
        hour_weights: np.ndarray,
    ) -> BatchEvent:
        at = self._moment(generator, hours, hour_weights)
        return BatchEvent(
            event_id=_event_id(account.account_id, at, 1),
            kind=EventKind.CHECKOUT_ABANDON,
            account_id=account.account_id,
            at=at,
            rail=account.rail,
            claim_type=ClaimType.CHECKOUT_ABANDON,
            amount_paise=account.plan_value_paise,
            succeeded=False,
            attempt=1,
            initiated_by=Initiator.MERCHANT,
        )

    # -- determinism -------------------------------------------------------

    def batch_digest(self) -> str:
        """SHA-256 over the canonical batch and the population that produced it.

        Ground truth is inside the digest on purpose: changing an injected
        outage or a latent trait must change the digest, so the freeze is
        checkable rather than asserted.
        """
        payload = {
            "seed": self._seed,
            "size": self._size,
            "epoch": self._epoch.isoformat(),
            "population": [_account_fingerprint(a) for a in self._accounts],
            "events": [event.as_record() for event in self.batch_events()],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _count_within(moments: Sequence[datetime], at: datetime, span: timedelta) -> int:
    """Events inside the half-open window `[at - span, at)`.

    Half-open like every other window in ARC: an event at exactly `at` belongs
    to the next window, not this one, so boundary behaviour is defined.
    """
    start = at - span
    return sum(1 for moment in moments if start <= moment < at)


def _event_id(account_id: str, at: datetime, attempt: int) -> str:
    """Derived, not random, so a redelivered webhook carries the same id and
    the adapter's dedupe at M5 is idempotent rather than merely likely."""
    return f"evt_{stable_hash(account_id, at.isoformat(), str(attempt)):016x}"


def _account_fingerprint(account: Account) -> dict[str, object]:
    """Canonical form of one account, ground truth included, for the digest."""
    latent = account.latent
    return {
        "account_id": account.account_id,
        "customer_ref": account.customer_ref,
        "issuer_id": account.issuer_id,
        "rail": str(account.rail),
        "claim_type": str(account.claim_type),
        "plan_value_paise": int(account.plan_value_paise),
        "opened_at": account.opened_at.isoformat(),
        "mandate_status": account.mandate_status,
        "mandate_cap_paise": None
        if account.mandate_cap_paise is None
        else int(account.mandate_cap_paise),
        "terminal_instrument": account.terminal_instrument,
        "credential_stale": account.credential_stale,
        "mandate_orphaned": account.mandate_orphaned,
        "consent": [list(pair) for pair in account.consent],
        "latent": {
            "ability_to_pay": latent.ability_to_pay,
            "monthly_income_paise": int(latent.monthly_income_paise),
            "salary_day": latent.salary_day,
            "salary_variance": latent.salary_variance,
            "annoyance_sensitivity": latent.annoyance_sensitivity,
            "intent_to_churn": latent.intent_to_churn,
            "promise_reliability": latent.promise_reliability,
            "digital_literacy": latent.digital_literacy,
            "phone_stale": latent.phone_stale,
            "responsiveness": {str(k): v for k, v in latent.responsiveness.items()},
        },
    }


def sleeping_dogs(world: World, at: datetime) -> tuple[str, ...]:
    """Accounts made worse off by every digital nudge.

    EVALUATION HARNESS ONLY - it reads ground truth. It exists so the M7
    uplift model can be scored on whether it found the accounts that were
    genuinely planted, rather than on whether its output looks plausible.

    Defined over the digital nudge family rather than over every contact
    action: a human handoff is an escalation with its own economics, and
    including it would define the cohort out of existence.
    """
    found: list[str] = []
    for account_id in world.account_ids:
        base = world.counterfactual(account_id, ActionType.DO_NOTHING, at)
        if all(
            world.counterfactual(account_id, action, at) < base
            for action in sorted(rm.DIGITAL_NUDGE_ACTIONS)
        ):
            found.append(account_id)
    return tuple(found)
