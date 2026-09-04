"""Per-source circuit breaker.

One misbehaving gateway must not stall the others. Repeated parse failure trips
that source and leaves the rest serving, which is the difference between a
degraded ingest and a stopped one.

Never a silent drop. A tripped source raises on admission and the refusal is
counted, so the dashboard at M14 can show which gateway stopped speaking a
dialect we understand. The full breaker family arrives at M13; this is the one
L0 cannot run without.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from arc.core.time_authority import ensure_utc

FAILURE_THRESHOLD = 20
COOLDOWN = timedelta(minutes=5)


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class SourceTripped(RuntimeError):
    """This source is refusing traffic until its cooldown elapses."""


@dataclass
class _Source:
    consecutive_failures: int = 0
    opened_at: datetime | None = None


@dataclass
class SourceBreakers:
    """Consecutive failures, per source, with a time-based cooldown."""

    threshold: int = FAILURE_THRESHOLD
    cooldown: timedelta = COOLDOWN
    _sources: dict[str, _Source] = field(default_factory=dict)

    def state(self, source: str, at: datetime) -> BreakerState:
        ensure_utc(at)
        entry = self._sources.get(source)
        if entry is None or entry.opened_at is None:
            return BreakerState.CLOSED
        if at - entry.opened_at >= self.cooldown:
            return BreakerState.HALF_OPEN
        return BreakerState.OPEN

    def admit(self, source: str, at: datetime) -> None:
        """Raise if this source is tripped. Called before anything is spent."""
        if self.state(source, at) is BreakerState.OPEN:
            raise SourceTripped(
                f"source {source!r} is tripped after "
                f"{self._sources[source].consecutive_failures} consecutive failures"
            )

    def record_success(self, source: str) -> None:
        self._sources[source] = _Source()

    def record_failure(self, source: str, at: datetime) -> None:
        ensure_utc(at)
        entry = self._sources.setdefault(source, _Source())
        entry.consecutive_failures += 1
        if entry.consecutive_failures >= self.threshold and entry.opened_at is None:
            entry.opened_at = at

    def failures(self, source: str) -> int:
        entry = self._sources.get(source)
        return 0 if entry is None else entry.consecutive_failures
