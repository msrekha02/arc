"""The only clock in the system.

Nothing else in the repo may read wall-clock time. `tests/test_core.py` walks
the AST of every file and fails the build on any other caller.

WHY: it is what makes the Gate a pure function, makes replay possible, and
makes tests deterministic. Every function that needs "now" receives it as a
parameter; in particular the Gate takes `at: datetime` and never reads a clock.

All timestamps crossing a module boundary are timezone-aware UTC. All rolling
windows are half-open, `[t - 7d, t)`, so boundary behaviour is defined rather
than accidental.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from zoneinfo import ZoneInfo

# Statutory contact window, subject-local. The Gate composes calendar rules
# (Sundays, gazetted holidays, festivals) on top of this; they are policy
# choices with different bases and do not belong in the Time Authority.
CONTACT_WINDOW_OPEN = time(8, 0)
CONTACT_WINDOW_CLOSE = time(19, 0)


class NotUTC(ValueError):
    """A naive or non-UTC datetime was offered where UTC was required."""


def ensure_utc(value: datetime) -> datetime:
    """Reject naive datetimes and anything not at zero offset.

    Fails closed: a naive datetime is not "probably UTC", it is unknown.
    """
    if not isinstance(value, datetime):
        raise NotUTC(f"expected datetime, got {type(value).__name__}")
    if value.tzinfo is None:
        raise NotUTC(f"naive datetime {value!r}; all timestamps must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise NotUTC(f"{value!r} is not UTC (offset {value.utcoffset()})")
    return value


class TzBasisKind(StrEnum):
    """Which source decided the subject's timezone.

    The three disagree, so which one was used is itself a recorded decision;
    picking silently produces out-of-hours contact with a clean audit log.
    """

    DECLARED = "declared"
    BILLING_ADDRESS = "billing_address"
    TELECOM_CIRCLE = "telecom_circle"


@dataclass(frozen=True)
class TimezoneBasis:
    kind: TzBasisKind
    zone: str  # IANA name, e.g. "Asia/Kolkata"

    def __post_init__(self) -> None:
        ZoneInfo(self.zone)  # raises on an unknown zone rather than defaulting


@dataclass(frozen=True)
class Window:
    """A half-open interval `[start, end)`. `end` is never a member."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        ensure_utc(self.start)
        ensure_utc(self.end)
        if self.start > self.end:
            raise ValueError(f"window start {self.start} is after end {self.end}")

    def contains(self, moment: datetime) -> bool:
        ensure_utc(moment)
        return self.start <= moment < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start


def rolling_window(end: datetime, duration: timedelta) -> Window:
    """The half-open lookback `[end - duration, end)`.

    Used by every frequency and cooldown rule. `end` itself is excluded, so an
    event at exactly `t` belongs to the next window, not this one.
    """
    ensure_utc(end)
    if duration <= timedelta(0):
        raise ValueError(f"window duration must be positive, got {duration}")
    return Window(end - duration, end)


def to_local(utc: datetime, tz_basis: TimezoneBasis) -> datetime:
    """Pure timezone conversion. No clock, no instance, safe inside the Gate."""
    ensure_utc(utc)
    return utc.astimezone(ZoneInfo(tz_basis.zone))


def next_contact_window(after: datetime, tz_basis: TimezoneBasis) -> datetime:
    """Earliest UTC instant at or after `after` inside the contact window.

    Pure, so the Gate can compute a DEFER target without holding a clock.
    Calendar rules (Sundays, gazetted holidays, festivals) are policy choices
    with a different basis and compose on top of this in the registry.
    """
    ensure_utc(after)
    zone = ZoneInfo(tz_basis.zone)
    local = after.astimezone(zone)

    if CONTACT_WINDOW_OPEN <= local.time() < CONTACT_WINDOW_CLOSE:
        return after

    before_open = local.time() < CONTACT_WINDOW_OPEN
    day = local.date() if before_open else local.date() + timedelta(days=1)
    return datetime.combine(day, CONTACT_WINDOW_OPEN, tzinfo=zone).astimezone(UTC)


class TimeAuthority:
    """Wall-clock access, timezone conversion, and the legal contact window.

    Bank holidays are injected, never fetched: a hidden network lookup would
    make the Gate's temporal rules non-reproducible under replay.
    """

    def __init__(self, holidays: Mapping[str, frozenset[date]] | None = None) -> None:
        self._holidays: Mapping[str, frozenset[date]] = MappingProxyType(
            {region: frozenset(days) for region, days in (holidays or {}).items()}
        )

    def now(self) -> datetime:
        """THE ONLY WALL-CLOCK READ IN THE REPOSITORY."""
        return datetime.now(UTC)

    def local(self, utc: datetime, tz_basis: TimezoneBasis) -> datetime:
        """Convert a UTC instant into the subject's local wall time."""
        return to_local(utc, tz_basis)

    def is_bank_holiday(self, d: date, region: str) -> bool:
        return d in self._holidays.get(region, frozenset())

    def next_legal_window(self, after: datetime, tz_basis: TimezoneBasis) -> datetime:
        """Earliest UTC instant at or after `after` inside the contact window.

        Returns `after` unchanged when it is already inside the window, so a
        DEFER always carries a timestamp a sleep can be scheduled against.
        """
        return next_contact_window(after, tz_basis)


class FrozenTimeAuthority(TimeAuthority):
    """A clock that does not move. Used by replay mode and by tests.

    WHY it lives in `core` and not in `tests`: replay is a product feature, not
    a test fixture. "Run it again and get the same number" needs a real clock
    substitute on the production path.
    """

    def __init__(
        self,
        at: datetime,
        holidays: Mapping[str, frozenset[date]] | None = None,
    ) -> None:
        super().__init__(holidays)
        self._at = ensure_utc(at)

    def now(self) -> datetime:
        return self._at

    def advance(self, delta: timedelta) -> None:
        self._at = self._at + delta
