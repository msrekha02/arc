"""Verdict algebra. Knows nothing about rules, time zones, or claims.

    BLOCK_PERMANENT > BLOCK > DEFER > ALLOW

Most restrictive always wins, so two rules can never conflict into ambiguity.
The ordering is total, which is what lets thirty-three independent verdicts
collapse into one decision without an adjudication step that could be wrong.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Verdict(StrEnum):
    ALLOW = "allow"
    DEFER = "defer"
    BLOCK = "block"
    BLOCK_PERMANENT = "block_permanent"


_RANK: dict[Verdict, int] = {
    Verdict.ALLOW: 0,
    Verdict.DEFER: 1,
    Verdict.BLOCK: 2,
    Verdict.BLOCK_PERMANENT: 3,
}


class DeferWithoutTimestamp(ValueError):
    """A DEFER that cannot say when. That is a BLOCK wearing the wrong label.

    Raised rather than tolerated because `step.sleepUntil()` consumes the
    timestamp directly: a DEFER with nothing to sleep until silently becomes an
    unbounded stall.
    """


def rank(verdict: Verdict) -> int:
    return _RANK[verdict]


def most_restrictive(verdicts: Iterable[Verdict]) -> Verdict:
    return max(verdicts, key=rank, default=Verdict.ALLOW)


@dataclass(frozen=True)
class Resolution:
    decision: Verdict
    defer_until: datetime | None


def resolve(items: Sequence[tuple[Verdict, datetime | None]]) -> Resolution:
    """Collapse per-rule verdicts into one decision.

    When the outcome is DEFER, the wait is the LATEST of every deferring rule's
    next-eligible time. Waiting for the earliest would wake into a rule that is
    still violated.
    """
    decision = most_restrictive(verdict for verdict, _ in items)

    if decision is not Verdict.DEFER:
        return Resolution(decision=decision, defer_until=None)

    untils = [until for verdict, until in items if verdict is Verdict.DEFER]
    if any(until is None for until in untils):
        raise DeferWithoutTimestamp("a DEFER verdict carried no next-eligible timestamp")
    return Resolution(decision=Verdict.DEFER, defer_until=max(u for u in untils if u is not None))
