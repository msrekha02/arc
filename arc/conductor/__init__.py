"""L6 - the Conductor. Make the decision happen exactly once, or not at all.

    Postgres owns state. Inngest owns time. Neither owns the other's job.

    commit.py        the one transaction: FSM, reserve, ledger, outbox
    outbox.py        the idempotency key, enqueue, SKIP LOCKED claim, reaper
    worker.py        Gate touchpoint 3, the channel call, the outcome
    reservations.py  two-tier budget holds, and the network attempt counter
    fsm.py           claim transitions, applied from M1's table
    kill_switch.py   four modes, the freeze, and a resume that cannot stampede
    breakers.py      the ten circuit breakers, three of which watch the watcher
    erasure.py       the erasure sweep, across every store that holds anything

WHAT IS GUARANTEED:

    exactly-once STATE TRANSITION   by the Postgres transaction
    at-least-once DISPATCH          by lease and retry
    effectively-once EFFECT         by a stable idempotency key the provider
                                    honours
    exactly-once DELIVERY           impossible, and not claimed

The last line is the one that matters. Claiming exactly-once delivery to an
external system is the tell that the designer has not thought about it.
"""

from arc.conductor.breakers import (
    SPECS,
    BreakerId,
    Reading,
    evaluate_all,
    evaluate_and_apply,
)
from arc.conductor.commit import (
    CommitRequest,
    CommitResult,
    UncertifiedAction,
    commit_decision,
)
from arc.conductor.fsm import (
    ClaimNotFound,
    ConcurrentTransition,
    current_state,
    transition,
    transition_from_current,
)
from arc.conductor.kill_switch import (
    RAMP,
    ControlState,
    HeldWorkExecuted,
    Mode,
    advance_ramp,
    current_mode,
    freeze,
    resume,
    set_mode,
)
from arc.conductor.outbox import (
    DEFAULT_LEASE,
    MAX_ATTEMPTS,
    OutboxRow,
    OutboxStatus,
    backoff,
    claim_batch,
    counts_by_status,
    enqueue,
    fetch_row,
    idempotency_key,
    mark,
    reap_expired_leases,
    reschedule,
)
from arc.conductor.reservations import (
    HARD_RESERVE_HORIZON,
    BudgetExhausted,
    Reservation,
    ReservationStatus,
    consume,
    declare_caps,
    expire_stale,
    harden,
    network_attempts_in_window,
    pipeline_demand,
    record_network_attempt,
    release,
    remaining,
    reserve,
    tier_for,
)
from arc.conductor.worker import (
    ChannelLike,
    DispatchOutcome,
    DispatchResult,
    PermanentError,
    RetryableError,
    dispatch,
    release_on_terminal,
    requeue_for_allocation,
    run_worker,
)

__all__ = [
    "BreakerId",
    "BudgetExhausted",
    "ChannelLike",
    "ClaimNotFound",
    "CommitRequest",
    "CommitResult",
    "ConcurrentTransition",
    "ControlState",
    "DEFAULT_LEASE",
    "DispatchOutcome",
    "DispatchResult",
    "HARD_RESERVE_HORIZON",
    "HeldWorkExecuted",
    "MAX_ATTEMPTS",
    "Mode",
    "OutboxRow",
    "OutboxStatus",
    "PermanentError",
    "RAMP",
    "Reading",
    "Reservation",
    "ReservationStatus",
    "RetryableError",
    "SPECS",
    "UncertifiedAction",
    "advance_ramp",
    "backoff",
    "claim_batch",
    "commit_decision",
    "consume",
    "counts_by_status",
    "current_mode",
    "current_state",
    "declare_caps",
    "dispatch",
    "enqueue",
    "evaluate_all",
    "evaluate_and_apply",
    "expire_stale",
    "fetch_row",
    "freeze",
    "harden",
    "idempotency_key",
    "mark",
    "network_attempts_in_window",
    "pipeline_demand",
    "reap_expired_leases",
    "record_network_attempt",
    "release",
    "release_on_terminal",
    "remaining",
    "requeue_for_allocation",
    "reschedule",
    "reserve",
    "resume",
    "run_worker",
    "set_mode",
    "tier_for",
    "transition",
    "transition_from_current",
]
