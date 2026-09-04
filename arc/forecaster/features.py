"""Feature extraction, and the staleness contract that governs it.

Two jobs, deliberately in one module because they are the same decision made
twice: what the models are allowed to look at, and for how long that look
stays valid.

THE OBSERVABLE BOUNDARY. Nothing here imports the simulator. Extraction is
defined against `ObservableLike`, a structural protocol that the simulator's
`ObservableState` happens to satisfy, so the forecaster can be trained on a
simulated population and served on a real one without changing a line. The
ground-truth surface is banned by name in CI from this package, and that ban is
what makes the uplift number at M11 mean anything: a model that reads the
answer key measures nothing.

THE ENGAGEMENT FAMILY is worth its own note, because it looks at first like a
back door and is not. It carries what ARC ITSELF already did to an account and
what came back: how many nudges it sent, how many landed, how many produced an
opt-out or a complaint. That is the Decision Ledger's own content, available in
production on day one, and it is the only honest handle the system has on how a
person responds to being contacted. Removing it would not make the model purer,
it would make it blind to the one thing it is entitled to learn from.

PER-FAMILY STALENESS, not one global timeout. Issuer health goes stale in
minutes and account attributes in weeks; a single TTL is either too loose for
the fast family or too tight for the slow one. Past its TTL a family is not
quietly extrapolated - the estimate falls back to a segment prior, sets
`degraded`, and widens its interval, because silent extrapolation on stale
features is exactly how these systems produce confident nonsense.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from arc.core.money import Paise, paise
from arc.core.time_authority import TimezoneBasis, TzBasisKind, ensure_utc, to_local
from arc.core.types import ActionType, Rail

# Every simulated subject is in India. A real deployment resolves this per
# subject; the default exists so a feature vector is never silently built
# against a timezone nobody chose.
DEFAULT_TZ_BASIS = TimezoneBasis(TzBasisKind.BILLING_ADDRESS, "Asia/Kolkata")

# Stand-in for "no observation", used where zero would be a lie. LightGBM
# handles NaN natively as a branch of its own, which is one reason the bounce
# model is a GBDT: an account with no payment history is a different thing
# from an account whose last payment was today.
MISSING = float("nan")


# ---------------------------------------------------------------------------
# What the models may see
# ---------------------------------------------------------------------------
@runtime_checkable
class ObservableLike(Protocol):
    """ARC's entire view of an account.

    Structural rather than nominal on purpose: the forecaster must not import
    the package that defines the concrete record, so the shape is the contract.
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


# ---------------------------------------------------------------------------
# Staleness families and their TTLs
# ---------------------------------------------------------------------------
class FeatureFamily(StrEnum):
    """The four families, each with its own decay rate."""

    ISSUER_HEALTH = "issuer_health"
    CONTACT_HISTORY = "contact_history"
    PAYMENT_HISTORY = "payment_history"
    ACCOUNT_ATTRIBUTES = "account_attributes"


TTL: Mapping[FeatureFamily, timedelta] = MappingProxyType(
    {
        # Changes fast. An issuer that was healthy an hour ago tells you
        # nothing about an issuer mid-incident now.
        FeatureFamily.ISSUER_HEALTH: timedelta(minutes=15),
        # Drives compliance decisions, so an hour is already generous.
        FeatureFamily.CONTACT_HISTORY: timedelta(hours=1),
        FeatureFamily.PAYMENT_HISTORY: timedelta(hours=24),
        # Near-static: a tenure or a rail does not move in a week.
        FeatureFamily.ACCOUNT_ATTRIBUTES: timedelta(days=7),
    }
)


@dataclass(frozen=True)
class FeatureFreshness:
    """When each family was last observed. Absent means never observed.

    A family with no timestamp is stale by definition rather than fresh by
    default: unknown fails closed (GI-5).
    """

    observed_at: Mapping[FeatureFamily, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for family, moment in self.observed_at.items():
            if not isinstance(family, FeatureFamily):
                raise TypeError(f"{family!r} is not a feature family")
            ensure_utc(moment)
        object.__setattr__(self, "observed_at", MappingProxyType(dict(self.observed_at)))

    @classmethod
    def fresh_at(cls, at: datetime) -> FeatureFreshness:
        """Everything observed now - the ordinary online case."""
        ensure_utc(at)
        return cls({family: at for family in FeatureFamily})

    def age(self, family: FeatureFamily, at: datetime) -> timedelta | None:
        moment = self.observed_at.get(family)
        return None if moment is None else at - moment

    def stale(self, at: datetime) -> tuple[FeatureFamily, ...]:
        """Families at or past their own TTL at `at`, in declared order.

        The comparison is `>=` because every window in ARC is half-open: an
        observation exactly one TTL old has left the window it was fresh in.
        """
        ensure_utc(at)
        late: list[FeatureFamily] = []
        for family in FeatureFamily:
            age = self.age(family, at)
            if age is None or age >= TTL[family]:
                late.append(family)
        return tuple(late)


# ---------------------------------------------------------------------------
# Context the observable record does not carry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IssuerSignal:
    """What the Sentinel's cohort view says about this issuer right now."""

    decline_rate_7d: float = MISSING
    degraded: bool = False


NO_ISSUER_SIGNAL = IssuerSignal()

# Action families, named here because the feature vector groups by them and
# `ActionType` itself is deliberately flat.
NUDGE_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.WHATSAPP_UTILITY,
        ActionType.SMS,
        ActionType.EMAIL,
        ActionType.PAYMENT_LINK,
    }
)
VOICE_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.VOICE_CALL, ActionType.INSTALMENT_OFFER, ActionType.HUMAN_HANDOFF}
)
SILENT_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.RETRY,
        ActionType.CARD_UPDATER,
        ActionType.MANDATE_RE_REGISTER,
        ActionType.RAIL_FALLBACK,
    }
)
CONTACT_ACTIONS: frozenset[ActionType] = (
    NUDGE_ACTIONS | VOICE_ACTIONS | frozenset({ActionType.STATUTORY_NOTICE})
)


@dataclass(frozen=True)
class EngagementHistory:
    """What ARC did to this account, and what came back.

    Sourced from the Decision Ledger, never from the world. `adverse_events`
    is the load-bearing field: an opt-out or a complaint is the single
    strongest observable evidence that further contact will not help, and it
    is how the uplift model reaches a segment nothing else exposes.
    """

    nudges_sent: int = 0
    voice_attempts: int = 0
    silent_attempts: int = 0
    payments_after_contact: int = 0
    payments_after_silent: int = 0
    adverse_events: int = 0
    opt_outs: int = 0
    complaints: int = 0
    promises_made: int = 0
    promises_kept: int = 0
    last_contact_at: datetime | None = None

    @property
    def contacts(self) -> int:
        return self.nudges_sent + self.voice_attempts

    def with_outcome(
        self,
        *,
        action: ActionType,
        paid: bool,
        adverse: bool = False,
        opted_out: bool = False,
        complained: bool = False,
        promised: bool = False,
        at: datetime | None = None,
    ) -> EngagementHistory:
        """Fold one logged decision in. Pure; returns a new record."""
        contact = action in CONTACT_ACTIONS
        silent = action in SILENT_ACTIONS
        return EngagementHistory(
            nudges_sent=self.nudges_sent + (1 if action in NUDGE_ACTIONS else 0),
            voice_attempts=self.voice_attempts + (1 if action in VOICE_ACTIONS else 0),
            silent_attempts=self.silent_attempts + (1 if silent else 0),
            payments_after_contact=self.payments_after_contact + (1 if paid and contact else 0),
            payments_after_silent=self.payments_after_silent + (1 if paid and silent else 0),
            adverse_events=self.adverse_events + (1 if adverse else 0),
            opt_outs=self.opt_outs + (1 if opted_out else 0),
            complaints=self.complaints + (1 if complained else 0),
            promises_made=self.promises_made + (1 if promised else 0),
            promises_kept=self.promises_kept,
            last_contact_at=at if contact and at is not None else self.last_contact_at,
        )


NO_ENGAGEMENT = EngagementHistory()


@dataclass(frozen=True)
class FeatureContext:
    """Everything extraction needs beyond the observable record itself."""

    at: datetime
    amount_paise: Paise | None = None
    portfolio_median_amount_paise: Paise = paise(100_000)
    issuer: IssuerSignal = NO_ISSUER_SIGNAL
    engagement: EngagementHistory = NO_ENGAGEMENT
    freshness: FeatureFreshness | None = None
    tz_basis: TimezoneBasis = DEFAULT_TZ_BASIS

    def __post_init__(self) -> None:
        ensure_utc(self.at)
        if self.freshness is None:
            object.__setattr__(self, "freshness", FeatureFreshness.fresh_at(self.at))

    def stale_families(self) -> tuple[FeatureFamily, ...]:
        freshness = self.freshness
        assert freshness is not None  # set in __post_init__
        return freshness.stale(self.at)


# ---------------------------------------------------------------------------
# The feature vector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """One column: its name, and the family whose TTL governs it."""

    name: str
    family: FeatureFamily
    categorical: bool = False


def _days(delta: timedelta) -> float:
    return delta.total_seconds() / 86400.0


def _mode_day_of_month(moments: Sequence[datetime], tz_basis: TimezoneBasis) -> float:
    """The day of the month payments cluster on - the inferred salary day.

    The world clusters credits a day or two after a latent salary day and
    never tells anyone what it is. Recovering it from the payment record is
    the whole reason a learned policy beats a fixed T+1/T+3/T+7 calendar.
    """
    if not moments:
        return MISSING
    counts: dict[int, int] = {}
    for moment in moments:
        day = to_local(moment, tz_basis).day
        counts[day] = counts.get(day, 0) + 1
    best = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return float(best[0])


def _days_to_day_of_month(at_local: datetime, day: float) -> float:
    if math.isnan(day):
        return MISSING
    ahead = int(day) - at_local.day
    return float(ahead if ahead >= 0 else ahead + 30)


def _bounce_streak(obs: ObservableLike) -> float:
    """Declines beyond the payments that offset them.

    A count of lifetime bounces says how bad the account has been; a streak
    says whether it is bad right now, and those are different questions.
    """
    if not obs.prior_payment_timestamps:
        return float(len(obs.decline_code_history))
    return float(max(len(obs.decline_code_history) - len(obs.prior_payment_timestamps), 0))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else MISSING


# Declared once, in one order. `FEATURE_NAMES` is what every model is trained
# and served on; a column inserted in the middle would silently re-map an
# already fitted model, so the order is part of the contract.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    # account
    FeatureSpec("prior_bounces_90d", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("bounce_streak", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("tenure_days", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("rail", FeatureFamily.ACCOUNT_ATTRIBUTES, categorical=True),
    FeatureSpec("plan_value_paise", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("invoice_ageing_bucket", FeatureFamily.ACCOUNT_ATTRIBUTES, categorical=True),
    FeatureSpec("consents_granted", FeatureFamily.ACCOUNT_ATTRIBUTES),
    # timing
    FeatureSpec("days_since_last_credit", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("payments_90d", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("day_of_month", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("is_month_end", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("inferred_salary_day", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("days_to_inferred_salary_day", FeatureFamily.PAYMENT_HISTORY),
    # mandate
    FeatureSpec("mandate_headroom_paise", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("mandate_over_cap", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("mandate_age_days", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("reissue_flag", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("days_since_reissue", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("mandate_active", FeatureFamily.ACCOUNT_ATTRIBUTES),
    # issuer
    FeatureSpec("issuer_id", FeatureFamily.ISSUER_HEALTH, categorical=True),
    FeatureSpec("issuer_7d_decline_rate", FeatureFamily.ISSUER_HEALTH),
    FeatureSpec("live_degradation_flag", FeatureFamily.ISSUER_HEALTH),
    # amount
    FeatureSpec("amount_paise", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("amount_vs_median_ratio", FeatureFamily.ACCOUNT_ATTRIBUTES),
    FeatureSpec("is_first_charge_of_cycle", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("hard_decline_seen", FeatureFamily.PAYMENT_HISTORY),
    FeatureSpec("do_not_retry_seen", FeatureFamily.PAYMENT_HISTORY),
    # engagement - ARC's own ledger, and its only handle on how a person
    # responds to being contacted
    FeatureSpec("contact_history_7d", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("contact_history_30d", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("nudges_sent", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("voice_attempts", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("silent_attempts", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("payments_after_contact", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("payments_after_silent", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("contact_conversion", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("adverse_events", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("adverse_per_contact", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("opt_outs", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("complaints", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("hours_since_last_contact", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("prior_promises", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("prior_promises_kept", FeatureFamily.CONTACT_HISTORY),
    FeatureSpec("prior_kept_ratio", FeatureFamily.CONTACT_HISTORY),
)

FEATURE_NAMES: tuple[str, ...] = tuple(spec.name for spec in FEATURE_SPECS)
FEATURE_INDEX: Mapping[str, int] = MappingProxyType(
    {name: position for position, name in enumerate(FEATURE_NAMES)}
)
CATEGORICAL_INDICES: tuple[int, ...] = tuple(
    position for position, spec in enumerate(FEATURE_SPECS) if spec.categorical
)
FAMILY_OF: Mapping[str, FeatureFamily] = MappingProxyType(
    {spec.name: spec.family for spec in FEATURE_SPECS}
)

# Categorical columns are hashed to a stable bucket rather than fitted to a
# vocabulary, so an issuer or a rail unseen at training time lands somewhere
# real instead of raising at serving time.
_CATEGORY_BUCKETS = 64


def _category(value: str) -> float:
    """Deterministic across processes - `hash()` on a str is salted per run,
    and a feature that moves between runs cannot be replayed."""
    if not value:
        return MISSING
    digest = sum(byte * (31**index % 1_000_003) for index, byte in enumerate(value.encode()))
    return float(digest % _CATEGORY_BUCKETS)


def extract(obs: ObservableLike, ctx: FeatureContext) -> tuple[float, ...]:
    """One observable record plus its context to one fixed-width vector.

    Pure. Takes `at` through the context and reads no clock, for the same
    reason the Gate does not: a feature vector that depends on when it was
    built cannot be replayed, and a decision that cannot be replayed cannot be
    audited.
    """
    at = ctx.at
    local = to_local(at, ctx.tz_basis)
    payments = obs.prior_payment_timestamps
    engagement = ctx.engagement
    amount = int(ctx.amount_paise if ctx.amount_paise is not None else obs.plan_value_paise)

    cap = obs.mandate_cap_paise
    salary_day = _mode_day_of_month(payments, ctx.tz_basis)
    contacts = engagement.contacts
    ptp = obs.prior_ptp_outcomes

    values: dict[str, float] = {
        "prior_bounces_90d": float(obs.prior_bounces_90d),
        "bounce_streak": _bounce_streak(obs),
        "tenure_days": float(obs.tenure_days),
        "rail": _category(str(obs.rail)),
        "plan_value_paise": float(int(obs.plan_value_paise)),
        "invoice_ageing_bucket": _category(obs.invoice_ageing_bucket),
        "consents_granted": float(
            sum(1 for _, state in obs.channel_consent_state if state == "granted")
        ),
        "days_since_last_credit": _days(at - payments[-1]) if payments else MISSING,
        "payments_90d": float(len(payments)),
        "day_of_month": float(local.day),
        "is_month_end": 1.0 if local.day >= 26 else 0.0,
        "inferred_salary_day": salary_day,
        "days_to_inferred_salary_day": _days_to_day_of_month(local, salary_day),
        "mandate_headroom_paise": float(int(cap) - amount) if cap is not None else MISSING,
        "mandate_over_cap": (MISSING if cap is None else (1.0 if amount > int(cap) else 0.0)),
        "mandate_age_days": (
            _days(at - obs.mandate_registered_at) if obs.mandate_registered_at else MISSING
        ),
        "reissue_flag": 1.0 if obs.instrument_reissued_at else 0.0,
        "days_since_reissue": (
            _days(at - obs.instrument_reissued_at) if obs.instrument_reissued_at else MISSING
        ),
        "mandate_active": 1.0 if obs.mandate_status == "active" else 0.0,
        "issuer_id": _category(obs.issuer_id),
        "issuer_7d_decline_rate": float(ctx.issuer.decline_rate_7d),
        "live_degradation_flag": 1.0 if ctx.issuer.degraded else 0.0,
        "amount_paise": float(amount),
        "amount_vs_median_ratio": _ratio(
            float(amount), float(int(ctx.portfolio_median_amount_paise))
        ),
        "is_first_charge_of_cycle": 1.0 if not obs.decline_code_history else 0.0,
        "hard_decline_seen": 1.0 if "MAC03" in obs.mac_history else 0.0,
        "do_not_retry_seen": 1.0 if obs.mac_history else 0.0,
        "contact_history_7d": float(obs.contact_history_7d),
        "contact_history_30d": float(obs.contact_history_30d),
        "nudges_sent": float(engagement.nudges_sent),
        "voice_attempts": float(engagement.voice_attempts),
        "silent_attempts": float(engagement.silent_attempts),
        "payments_after_contact": float(engagement.payments_after_contact),
        "payments_after_silent": float(engagement.payments_after_silent),
        "contact_conversion": _ratio(float(engagement.payments_after_contact), float(contacts)),
        "adverse_events": float(engagement.adverse_events),
        "adverse_per_contact": _ratio(float(engagement.adverse_events), float(contacts)),
        "opt_outs": float(engagement.opt_outs),
        "complaints": float(engagement.complaints),
        "hours_since_last_contact": (
            (at - engagement.last_contact_at).total_seconds() / 3600.0
            if engagement.last_contact_at
            else MISSING
        ),
        "prior_promises": float(engagement.promises_made + len(ptp)),
        "prior_promises_kept": float(
            engagement.promises_kept + sum(1 for outcome in ptp if outcome == "kept")
        ),
        "prior_kept_ratio": _ratio(
            float(sum(1 for outcome in ptp if outcome == "kept")), float(len(ptp))
        ),
    }

    missing = set(FEATURE_NAMES) - set(values)
    if missing:
        raise KeyError(f"feature specs without an extractor: {sorted(missing)}")
    return tuple(values[name] for name in FEATURE_NAMES)
