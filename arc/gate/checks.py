"""The executable half of a rule. Pure functions over the context and `at`.

A rule in YAML names a check and supplies its parameters, so the thresholds a
compliance reviewer argues about (48 hours, 3 per week, 15 per 30 days) are
data, and the logic is a small library of primitives that several rules share.
That is what stops the registry becoming thirty-three bespoke code paths that
drift apart.

Every check answers two questions: is this violated, and if so, when does it
stop being violated. The second answer is what a DEFER carries. A check that
cannot compute it returns `None`, and the evaluator downgrades the remedy to
BLOCK rather than emitting a DEFER nobody can sleep on.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

from arc.core.time_authority import next_contact_window, rolling_window, to_local
from arc.core.types import ActionType
from arc.gate.context import (
    CONTACT_CHANNELS,
    DISCLOSING_ACTIONS,
    DO_NOT_RETRY_CODES,
    HARD_DECLINE_CATEGORIES,
    TIME_BOUNDED_CHANNELS,
    Channel,
    ConsentState,
    ContactOutcome,
    GateContext,
    TargetRelationship,
)


@dataclass(frozen=True)
class CheckOutcome:
    """Violated or not, and when it clears. `until` is None when unknowable."""

    violated: bool
    until: datetime | None = None
    detail: str = ""


CLEAR = CheckOutcome(violated=False)

Check = Callable[[GateContext, ActionType, datetime, Mapping[str, Any]], CheckOutcome]


class UnknownCheck(KeyError):
    """A rule names a check that does not exist. The registry refuses to load."""


# ---------------------------------------------------------------------------
# Absolute prohibitions and subject state
# ---------------------------------------------------------------------------
def subject_flag(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Violated when a named subject flag is set. No end time; these are BLOCKs."""
    flag = str(params["flag"])
    if getattr(ctx.flags, flag):
        return CheckOutcome(True, None, f"subject flag {flag} is set")
    return CLEAR


def target_is_obligor(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Only the obligor or a guarantor may be contacted. UNKNOWN fails closed."""
    permitted = {TargetRelationship.OBLIGOR, TargetRelationship.GUARANTOR}
    if ctx.target in permitted:
        return CLEAR
    return CheckOutcome(True, None, f"contact target is {ctx.target}, not the obligor")


def target_is_employer(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    if ctx.target is TargetRelationship.EMPLOYER:
        return CheckOutcome(True, None, "contact target is the employer or workplace")
    return CLEAR


def channel_opted_out(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    channel = ctx.channel_for(action)
    if channel in ctx.opted_out_channels:
        return CheckOutcome(True, None, f"{channel} is opted out")
    return CLEAR


def channel_consent(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Anything but an explicit grant blocks. A missing record is not consent."""
    channel = ctx.channel_for(action)
    state = ctx.consent_for(channel)
    if state is ConsentState.GRANTED:
        return CLEAR
    return CheckOutcome(True, None, f"consent for {channel} is {state}")


def identity_verified(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """No account detail is stated to anyone before they are verified."""
    if action in DISCLOSING_ACTIONS and not ctx.flags.identity_verified:
        return CheckOutcome(True, None, "identity not verified before disclosure")
    return CLEAR


# ---------------------------------------------------------------------------
# State-based freezes
# ---------------------------------------------------------------------------
def freeze_until(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """A freeze that ends at a known time defers; one that does not, blocks.

    `extra_hours` covers the grace period after a promise date and the
    stand-down after an issuer outage resolves.
    """
    flag = str(params["flag"])
    if not getattr(ctx.flags, flag):
        return CLEAR

    until_field = str(params["until_field"])
    until = getattr(ctx.flags, until_field)
    if until is None:
        return CheckOutcome(True, None, f"{flag} is set with no known end")

    until = until + timedelta(hours=float(params.get("extra_hours", 0)))
    if at >= until:
        return CLEAR
    return CheckOutcome(True, until, f"{flag} until {until.isoformat()}")


# ---------------------------------------------------------------------------
# Network and payment rules
# ---------------------------------------------------------------------------
def hard_decline_category(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    if ctx.decline_category in HARD_DECLINE_CATEGORIES:
        return CheckOutcome(
            True, None, f"{ctx.decline_category} is a do-not-retry decline category"
        )
    return CLEAR


def do_not_retry_advice(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    code = (ctx.advice_code or "").strip().upper()
    if code and code in DO_NOT_RETRY_CODES:
        return CheckOutcome(True, None, f"advice code {code} instructs no further presentment")
    return CLEAR


def attempt_budget(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Attempts in a half-open rolling window, gateway-initiated ones included.

    The next-eligible time is when the oldest counted attempt falls out of the
    window, which is computable, so this defers rather than blocking outright
    unless the caller asked for a block.
    """
    window_days = int(params["window_days"])
    limit = int(params["max_attempts"])
    window = rolling_window(at, timedelta(days=window_days))

    counted = sorted(event.at for event in ctx.retries if window.contains(event.at))
    if len(counted) < limit:
        return CLEAR

    # Enough must age out that one more attempt fits.
    oldest_to_expire = counted[len(counted) - limit]
    until = oldest_to_expire + timedelta(days=window_days)
    return CheckOutcome(True, until, f"{len(counted)} attempts in {window_days}d, cap is {limit}")


def attempts_per_day(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    limit = int(params["max_per_day"])
    window = rolling_window(at, timedelta(days=1))
    counted = sorted(event.at for event in ctx.retries if window.contains(event.at))
    if len(counted) < limit:
        return CLEAR
    until = counted[len(counted) - limit] + timedelta(days=1)
    return CheckOutcome(True, until, f"{len(counted)} attempts in 24h, cap is {limit}")


def amount_within_mandate_cap(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    if ctx.mandate_cap_paise is None:
        return CheckOutcome(True, None, "mandate cap unknown")
    if ctx.amount_paise > ctx.mandate_cap_paise:
        return CheckOutcome(
            True,
            None,
            f"amount exceeds the mandate cap by {ctx.amount_paise - ctx.mandate_cap_paise} paise",
        )
    return CLEAR


def predebit_notice_lead(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """A debit may not be presented until the notice has had its lead time."""
    lead = timedelta(hours=float(params["hours"]))
    if ctx.predebit_notice_at is None:
        return CheckOutcome(True, None, "no pre-debit notification was sent")
    earliest = ctx.predebit_notice_at + lead
    if at >= earliest:
        return CLEAR
    return CheckOutcome(True, earliest, f"pre-debit notice needs {lead} of lead time")


# ---------------------------------------------------------------------------
# Time of day and calendar
# ---------------------------------------------------------------------------
def _parse_clock(value: Any) -> time:
    if isinstance(value, time):
        return value
    hours, _, minutes = str(value).partition(":")
    return time(int(hours), int(minutes or 0))


def contact_window(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """The statutory window, in the subject's local time.

    An unresolved timezone blocks rather than defaulting to a zone, because
    guessing produces out-of-hours contact with a clean-looking audit trail.
    """
    channel = ctx.channel_for(action)
    if channel not in TIME_BOUNDED_CHANNELS:
        return CLEAR
    if ctx.tz_basis is None:
        return CheckOutcome(True, None, "subject timezone unresolved")

    opens = _parse_clock(params.get("open", "08:00"))
    closes = _parse_clock(params.get("close", "19:00"))
    local = to_local(at, ctx.tz_basis)

    if opens <= local.time() < closes:
        return CLEAR
    return CheckOutcome(
        True,
        next_contact_window(at, ctx.tz_basis),
        f"local time {local.time().isoformat()} is outside [{opens}, {closes})",
    )


def quiet_hours(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Subject-declared quiet hours, which may wrap past midnight."""
    channel = ctx.channel_for(action)
    if channel not in TIME_BOUNDED_CHANNELS or ctx.quiet_hours is None:
        return CLEAR
    if ctx.tz_basis is None:
        return CheckOutcome(True, None, "subject timezone unresolved")

    start, end = ctx.quiet_hours
    local = to_local(at, ctx.tz_basis)
    now = local.time()
    inside = start <= now or now < end if start > end else start <= now < end
    if not inside:
        return CLEAR

    day = local.date() if now < end else local.date() + timedelta(days=1)
    ends = datetime.combine(day, end, tzinfo=local.tzinfo)
    return CheckOutcome(True, ends.astimezone(at.tzinfo), f"inside quiet hours {start}-{end}")


def blocked_days(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Sundays and gazetted holidays. The calendar is injected, never fetched."""
    channel = ctx.channel_for(action)
    if channel not in TIME_BOUNDED_CHANNELS:
        return CLEAR
    if ctx.tz_basis is None:
        return CheckOutcome(True, None, "subject timezone unresolved")

    weekdays = {int(day) for day in params.get("weekdays", [])}
    include_holidays = bool(params.get("include_bank_holidays", True))
    local = to_local(at, ctx.tz_basis)

    def is_blocked(day: Any) -> bool:
        return day.weekday() in weekdays or (include_holidays and day in ctx.bank_holidays)

    if not is_blocked(local.date()):
        return CLEAR

    day = local.date()
    for _ in range(30):
        day = day + timedelta(days=1)
        if not is_blocked(day):
            opens = _parse_clock(params.get("open", "08:00"))
            resumes = datetime.combine(day, opens, tzinfo=local.tzinfo)
            return CheckOutcome(True, resumes.astimezone(at.tzinfo), f"{local.date()} is blocked")
    return CheckOutcome(True, None, "no permitted day within 30 days")


def certificate_window(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """RUNTIME. An authorisation that has expired is not nearly valid.

    Only fires when the context carries an already-issued certificate, which is
    the case at the dispatch and wake touchpoints.
    """
    if ctx.certificate_valid_until is None:
        return CLEAR
    if at <= ctx.certificate_valid_until:
        return CLEAR
    return CheckOutcome(
        True, None, f"certificate expired at {ctx.certificate_valid_until.isoformat()}"
    )


# ---------------------------------------------------------------------------
# Cooldowns and frequency
# ---------------------------------------------------------------------------
def _last_contact(
    ctx: GateContext,
    *,
    at: datetime,
    channels: frozenset[Channel],
    outcomes: frozenset[ContactOutcome] | None = None,
) -> datetime | None:
    relevant = [
        event.at
        for event in ctx.contacts
        if event.channel in channels
        and event.at < at
        and (outcomes is None or event.outcome in outcomes)
    ]
    return max(relevant, default=None)


def channel_cooldown(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Minimum gap since the last contact on this channel.

    A cooldown is a hard limit, never a budget. Sufficient expected value must
    not be able to buy past it, or the system will eventually harass someone.
    """
    channel = ctx.channel_for(action)
    scoped = Channel(params["channel"]) if "channel" in params else channel
    if channel is not scoped:
        return CLEAR

    outcomes = (
        frozenset(ContactOutcome(o) for o in params["outcomes"]) if "outcomes" in params else None
    )
    last = _last_contact(ctx, at=at, channels=frozenset({scoped}), outcomes=outcomes)
    if last is None:
        return CLEAR

    gap = timedelta(hours=float(params["hours"]))
    eligible = last + gap
    if at >= eligible:
        return CLEAR
    qualifier = f" after {'/'.join(sorted(outcomes))}" if outcomes else ""
    return CheckOutcome(True, eligible, f"{scoped} cooldown of {gap}{qualifier}")


def cross_channel_cooldown(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """A gap between ANY two outbound contacts.

    Without it, one WhatsApp, one SMS and one email inside ten minutes each
    pass their own cooldown while the subject experiences a burst.
    """
    if ctx.channel_for(action) not in CONTACT_CHANNELS:
        return CLEAR

    last = _last_contact(ctx, at=at, channels=CONTACT_CHANNELS)
    if last is None:
        return CLEAR

    gap = timedelta(hours=float(params["hours"]))
    eligible = last + gap
    if at >= eligible:
        return CLEAR
    return CheckOutcome(True, eligible, f"cross-channel cooldown of {gap}")


def contact_frequency(
    ctx: GateContext, action: ActionType, at: datetime, params: Mapping[str, Any]
) -> CheckOutcome:
    """Total contacts per subject in a half-open rolling window.

    Counted at the subject, not the claim: the subject is the unit a message
    physically reaches, so three claims must not buy three times the contact.
    """
    if ctx.channel_for(action) not in CONTACT_CHANNELS:
        return CLEAR

    hours = float(params["window_hours"])
    limit = int(params["max_contacts"])
    window = rolling_window(at, timedelta(hours=hours))

    counted = sorted(
        event.at
        for event in ctx.contacts
        if event.channel in CONTACT_CHANNELS and window.contains(event.at)
    )
    if len(counted) < limit:
        return CLEAR

    until = counted[len(counted) - limit] + timedelta(hours=hours)
    return CheckOutcome(True, until, f"{len(counted)} contacts in {hours:g}h, cap is {limit}")


CHECKS: Mapping[str, Check] = {
    "subject_flag": subject_flag,
    "target_is_obligor": target_is_obligor,
    "target_is_employer": target_is_employer,
    "channel_opted_out": channel_opted_out,
    "channel_consent": channel_consent,
    "identity_verified": identity_verified,
    "freeze_until": freeze_until,
    "hard_decline_category": hard_decline_category,
    "do_not_retry_advice": do_not_retry_advice,
    "attempt_budget": attempt_budget,
    "attempts_per_day": attempts_per_day,
    "amount_within_mandate_cap": amount_within_mandate_cap,
    "predebit_notice_lead": predebit_notice_lead,
    "contact_window": contact_window,
    "quiet_hours": quiet_hours,
    "blocked_days": blocked_days,
    "certificate_window": certificate_window,
    "channel_cooldown": channel_cooldown,
    "cross_channel_cooldown": cross_channel_cooldown,
    "contact_frequency": contact_frequency,
}


def get_check(name: str) -> Check:
    try:
        return CHECKS[name]
    except KeyError as exc:
        raise UnknownCheck(f"no check named {name!r}") from exc
