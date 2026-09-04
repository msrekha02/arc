"""M12 - durable functions. Inngest owns time; Postgres owns state.

    runtime.py         memoised steps, sleep, waitForEvent, cancelOn
                       (the vocabulary and the event log live in `arc.events`,
                        so nothing has to import this package to publish one)
    gated_enqueue.py   THE ONLY PATH TO AN EFFECT
    salary_retry.py    sleep to payday, re-gate, enqueue
    ptp_tracker.py     wait for payment, record, hand back to the Allocator

TWO STRUCTURAL RULES, BOTH ENFORCED RATHER THAN DOCUMENTED:

    1. Nothing here imports from `arc.channels`. A durable step may not touch
       the world; it writes an outbox row inside a Postgres transaction and
       the outbox workers dispatch. CI fails the build on the import.

    2. Every wake re-enters the Gate. `gated_enqueue` re-fetches claim state
       and calls `certify` at the moment of the wake, with no fast path and no
       "nothing changed" branch.

The split is what keeps exactly-once state transition in exactly one place. A
total Inngest outage degrades to "delayed actions stop firing", not "duplicate
charges".
"""

from arc.events.bus import ObservedEvent, emit
from arc.events.names import (
    CANCEL_ON,
    EventName,
    IncompleteCancellation,
    MatchOn,
    assert_cancels_on_every_stop,
)
from arc.events.runs import (
    DurableRun,
    cancel_runs_for_subject,
    finish_run,
    run_status,
    start_run,
)
from arc.inngest_fns.gated_enqueue import (
    MAX_DEFER_HOPS,
    TERMINAL_STATES,
    EnqueueResult,
    Outcome,
    gated_enqueue,
)
from arc.inngest_fns.ptp_tracker import (
    DEFAULT_GRACE,
    Promise,
    PromiseOutcome,
    PromiseResult,
    classify,
    promise_to_pay_tracker,
)
from arc.inngest_fns.runtime import Clock, ManualClock, RunCancelled, Step, StepFailed
from arc.inngest_fns.salary_retry import RetryOutcome, RetryRequest, salary_aligned_retry

__all__ = [
    "CANCEL_ON",
    "DEFAULT_GRACE",
    "MAX_DEFER_HOPS",
    "TERMINAL_STATES",
    "Clock",
    "DurableRun",
    "EnqueueResult",
    "EventName",
    "IncompleteCancellation",
    "ManualClock",
    "MatchOn",
    "ObservedEvent",
    "Outcome",
    "Promise",
    "PromiseOutcome",
    "PromiseResult",
    "RetryOutcome",
    "RetryRequest",
    "RunCancelled",
    "Step",
    "StepFailed",
    "assert_cancels_on_every_stop",
    "cancel_runs_for_subject",
    "classify",
    "emit",
    "finish_run",
    "gated_enqueue",
    "promise_to_pay_tracker",
    "run_status",
    "salary_aligned_retry",
    "start_run",
]
