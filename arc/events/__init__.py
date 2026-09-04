"""Shared event infrastructure: the vocabulary, the log, and run lifecycle.

    names.py  the closed event set and the six behavioural stops
    bus.py    publish an event, ask what has happened since
    runs.py   the durable run row, its subscriptions, and how it ends

A LEAF PACKAGE, ON PURPOSE. It imports `arc.core` and nothing else in the
system, so anything may depend on it and it depends on nothing that could
depend back. That is what removed the `conductor -> inngest_fns -> conductor`
cycle: both packages now point at this one instead of at each other.
"""

from arc.events.bus import ObservedEvent, Subscription, emit, events_since
from arc.events.names import (
    CANCEL_ON,
    MATCH_FIELD,
    EventName,
    IncompleteCancellation,
    MatchOn,
    assert_cancels_on_every_stop,
    match_value,
)
from arc.events.runs import (
    DurableRun,
    cancel_runs_for_subject,
    finish_run,
    run_status,
    start_run,
)

__all__ = [
    "CANCEL_ON",
    "MATCH_FIELD",
    "DurableRun",
    "EventName",
    "IncompleteCancellation",
    "MatchOn",
    "ObservedEvent",
    "Subscription",
    "assert_cancels_on_every_stop",
    "cancel_runs_for_subject",
    "emit",
    "events_since",
    "finish_run",
    "match_value",
    "run_status",
    "start_run",
]
