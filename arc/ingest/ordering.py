"""Order by event time, never by arrival.

`payment.captured` can arrive before the `payment.failed` it supersedes. Three
percent of the fake's deliveries arrive late on purpose, so this is exercised
on every run rather than reasoned about.

Processing in arrival order creates a claim for money that was already
collected, and a claim that should not exist is worse than a missing one: it
gets diagnosed, allocated budget, and messaged to somebody who already paid.

The resolution has two halves. Deliveries are SORTED by event time, and events
for one account are FOLDED so that the last word wins. Two failed attempts on
one debit are one claim with two attempts, not two claims - an obligation is a
claim, and a presentation is an attempt at it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from arc.ingest.events import RawEvent


@dataclass(frozen=True)
class AccountTimeline:
    """Everything that happened to one account, in event-time order."""

    account_ref: str
    events: tuple[RawEvent, ...]

    @property
    def latest(self) -> RawEvent:
        return self.events[-1]

    @property
    def failed_attempts(self) -> int:
        return sum(1 for event in self.events if event.is_failure)

    @property
    def resolved(self) -> bool:
        """The last word was a capture, so there is nothing to recover."""
        return self.latest.succeeded

    @property
    def superseded(self) -> bool:
        """A failure was overtaken by a later capture.

        This is the count that proves ordering did something. If it is zero on
        a run of adversarial traffic, ordering is not being exercised.
        """
        return self.resolved and any(event.is_failure for event in self.events)


@dataclass
class OrderingResult:
    timelines: tuple[AccountTimeline, ...] = ()
    superseded: int = 0
    out_of_order_arrivals: int = 0
    reordered: list[str] = field(default_factory=list)


def sort_by_event_time(events: Iterable[RawEvent]) -> list[RawEvent]:
    """Deterministic order: event time, then event id to break ties.

    The tiebreak matters for replay. Two events at the same instant must sort
    the same way on every run or the batch digest moves.
    """
    return sorted(events, key=lambda event: (event.event_timestamp, event.source, event.event_id))


def count_out_of_order(arrival_order: Sequence[RawEvent]) -> int:
    """Deliveries that arrived after something which happened later than them.

    Measured on arrival order as received, which is why the pipeline counts it
    before it sorts. It is the observable evidence that late delivery is real
    traffic rather than a documented intention.
    """
    highest = None
    inversions = 0
    for event in arrival_order:
        if highest is not None and event.event_timestamp < highest:
            inversions += 1
        if highest is None or event.event_timestamp > highest:
            highest = event.event_timestamp
    return inversions


def fold_by_account(events: Iterable[RawEvent]) -> OrderingResult:
    """Group into per-account timelines, ordered, with the last word winning."""
    ordered = sort_by_event_time(events)
    grouped: dict[str, list[RawEvent]] = {}
    for event in ordered:
        grouped.setdefault(event.account_ref, []).append(event)

    timelines = tuple(
        AccountTimeline(account_ref=account_ref, events=tuple(rows))
        for account_ref, rows in grouped.items()
    )
    return OrderingResult(
        timelines=timelines,
        superseded=sum(1 for timeline in timelines if timeline.superseded),
        reordered=[t.account_ref for t in timelines if t.superseded],
    )
