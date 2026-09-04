"""The only path from a durable function to an effect.

    An Inngest step may never touch a channel. The only path to the world is
    gatedEnqueue(). Enforced by CI: files under inngest_fns/ may not import
    from channels/.

WHAT IT DOES, IN ORDER, AND WHY THE ORDER IS THE DESIGN.

    1. Re-fetch the claim's state. FRESH, from the database, not from the
       event that started the run three days ago. A claim that recovered,
       forbore or was written off in the meantime is terminal and the run
       stops here.
    2. GATE TOUCHPOINT 4 - full re-certification against the state and the
       moment of the WAKE. Not the moment of the decision. Every rule
       evaluates again; nothing is carried forward.
    3. On ALLOW, harden the soft reservation and insert the outbox row in ONE
       transaction.

WHY EVERY WAKE RE-CERTIFIES, WITH NO FAST PATH. The certificate that
authorised the plan expired while the function slept - that is what
certificate windows are FOR. Reusing it would be executing on stale
authorisation, which is the single thing the whole conductor design exists to
prevent. There is no `if nothing changed, skip the gate` branch, because
"nothing changed" is a claim about the world that only the Gate is entitled to
make.

BLOCK IS NOT RETRYABLE. DEFER IS, THREE TIMES.

    This is the consequence of M3's lattice that most needs stating. A DEFER
    carries a computable next-eligible timestamp and `sleepUntil` consumes it
    directly. A BLOCK does not and cannot: a freeze with no known end - an
    issuer outage whose resolution time nobody knows - resolves to BLOCK
    precisely BECAUSE there is no timestamp to sleep until. Treating it as
    retryable would mean sleeping on a duration invented here, which is a
    policy decision the Gate declined to make being made anyway by the
    scheduler.

    So BLOCK terminates the treatment path and the claim returns to the
    Allocator on the next cycle, which will re-score it against the world as
    it then is. And DEFER is bounded to three hops: an unbounded defer loop is
    a retry loop in disguise, reusing a decision and a propensity that were
    computed for a different moment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import UUID

from arc.conductor import reservations
from arc.conductor.fsm import current_state
from arc.conductor.outbox import enqueue, idempotency_key
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType, ClaimState
from arc.gate.context import ACTION_CHANNEL, GateContext
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict
from arc.inngest_fns.runtime import Step
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType

# States from which no further treatment is attempted. FORBORNE is absorbing by
# construction and RECOVERED, WRITTEN_OFF and REVERSED have all resolved the
# claim one way or another.
TERMINAL_STATES: frozenset[ClaimState] = frozenset(
    {
        ClaimState.RECOVERED,
        ClaimState.WRITTEN_OFF,
        ClaimState.FORBORNE,
    }
)

# An unbounded defer loop is a retry loop reusing stale authorisation. After
# three hops the claim goes back to the Allocator for a genuinely fresh
# decision, with a fresh certificate and a fresh propensity.
MAX_DEFER_HOPS = 3


class Outcome(StrEnum):
    ENQUEUED = "ENQUEUED"
    DEFER = "DEFER"
    BLOCKED = "BLOCKED"
    TERMINATED = "TERMINATED"
    REQUEUED = "REQUEUED"
    BUDGET_GONE = "BUDGET_GONE"


@dataclass(frozen=True)
class EnqueueResult:
    outcome: Outcome
    until: datetime | None = None
    blocking_rule_ids: tuple[str, ...] = ()
    outbox_id: int | None = None
    certificate_id: UUID | None = None
    state: ClaimState | None = None
    hardened: int = 0

    @property
    def is_retryable(self) -> bool:
        """Only a DEFER may be slept on, and only because it says until when.

        A BLOCK carries no timestamp. Anything that treated it as retryable
        would be inventing one.
        """
        return self.outcome is Outcome.DEFER and self.until is not None


class CertifyingGate(Protocol):
    def certify(self, ctx: GateContext, action: ActionType, at: datetime) -> Certificate: ...


class ContextSource(Protocol):
    """How a run rebuilds the Gate's view at wake, from current state.

    A Protocol rather than a concrete loader because the context is assembled
    from several stores and the durable function has no business knowing which.
    What matters here is that it is fetched at the wake and never cached across
    a sleep.
    """

    async def context_for(
        self, conn: Any, claim_id: UUID, *, at: datetime
    ) -> GateContext | None: ...


async def gated_enqueue(
    step: Step,
    conn: Any,
    *,
    claim_id: UUID,
    subject_token: str,
    cycle_id: UUID,
    action: ActionType,
    gate: CertifyingGate,
    contexts: ContextSource,
    at: datetime,
    payload: Mapping[str, Any] | None = None,
    ledger: DecisionLedger | None = None,
    reservation_key: str | None = None,
    priority: int = 0,
) -> EnqueueResult:
    """Re-fetch, re-certify, then enqueue. Wrapped as one memoised step."""
    ensure_utc(at)
    ledger = ledger or DecisionLedger()

    async def _body() -> dict[str, Any]:
        result = await _gate_and_enqueue(
            conn,
            claim_id=claim_id,
            subject_token=subject_token,
            cycle_id=cycle_id,
            action=action,
            gate=gate,
            contexts=contexts,
            at=at,
            payload=dict(payload or {}),
            ledger=ledger,
            reservation_key=reservation_key,
            priority=priority,
        )
        return {
            "outcome": result.outcome.value,
            "until": result.until.isoformat() if result.until else None,
            "blocking_rule_ids": list(result.blocking_rule_ids),
            "outbox_id": result.outbox_id,
            "certificate_id": str(result.certificate_id) if result.certificate_id else None,
            "state": result.state.value if result.state else None,
            "hardened": result.hardened,
        }

    recorded = await step.run_step(f"gate-and-enqueue:{action.value}:{at.isoformat()}", _body)
    return EnqueueResult(
        outcome=Outcome(recorded["outcome"]),
        until=datetime.fromisoformat(recorded["until"]) if recorded.get("until") else None,
        blocking_rule_ids=tuple(recorded.get("blocking_rule_ids") or ()),
        outbox_id=recorded.get("outbox_id"),
        certificate_id=(
            UUID(recorded["certificate_id"]) if recorded.get("certificate_id") else None
        ),
        state=ClaimState(recorded["state"]) if recorded.get("state") else None,
        hardened=int(recorded.get("hardened") or 0),
    )


async def _gate_and_enqueue(
    conn: Any,
    *,
    claim_id: UUID,
    subject_token: str,
    cycle_id: UUID,
    action: ActionType,
    gate: CertifyingGate,
    contexts: ContextSource,
    at: datetime,
    payload: dict[str, Any],
    ledger: DecisionLedger,
    reservation_key: str | None,
    priority: int,
) -> EnqueueResult:
    # 1. FRESH STATE. Not the state the event carried.
    state = await current_state(conn, claim_id)
    if state in TERMINAL_STATES:
        return EnqueueResult(outcome=Outcome.TERMINATED, state=state)

    context = await contexts.context_for(conn, claim_id, at=at)
    if context is None:
        # No context is not permission to proceed. GI-5: unknown fails closed.
        return EnqueueResult(
            outcome=Outcome.BLOCKED,
            blocking_rule_ids=("CONTEXT-UNAVAILABLE",),
            state=state,
        )

    # 2. GATE TOUCHPOINT 4 - full re-certification, at the wake moment.
    certificate = gate.certify(context, action, at)

    if certificate.decision is Verdict.DEFER:
        # DEFER always carries a next-eligible timestamp; M3's lattice raises
        # rather than emitting one without. Belt and braces here, because a
        # DEFER with no timestamp reaching `sleepUntil` is an unbounded stall.
        if certificate.defer_until is None:
            return EnqueueResult(
                outcome=Outcome.BLOCKED,
                blocking_rule_ids=tuple(certificate.blocking_rule_ids),
                certificate_id=certificate.certificate_id,
                state=state,
            )
        return EnqueueResult(
            outcome=Outcome.DEFER,
            until=certificate.defer_until,
            blocking_rule_ids=tuple(certificate.blocking_rule_ids),
            certificate_id=certificate.certificate_id,
            state=state,
        )

    if certificate.decision is not Verdict.ALLOW:
        # BLOCK or BLOCK_PERMANENT. Terminates the treatment path. There is no
        # timestamp to sleep until and inventing one here would be the
        # scheduler making a policy decision the Gate declined to make.
        await ledger.append(
            conn,
            LedgerEntry(
                event_type=LedgerEventType.GATE_VETO,
                occurred_at=at,
                claim_id=claim_id,
                subject_token=subject_token,
                payload={
                    "decision": certificate.decision.value,
                    "blocking_rule_ids": list(certificate.blocking_rule_ids),
                    "touchpoint": 4,
                    "retryable": False,
                    "returned_to_allocator": True,
                },
            ),
        )
        return EnqueueResult(
            outcome=Outcome.BLOCKED,
            blocking_rule_ids=tuple(certificate.blocking_rule_ids),
            certificate_id=certificate.certificate_id,
            state=state,
        )

    # 3. ALLOW. Harden the intent and enqueue, in ONE transaction.
    key = reservation_key or idempotency_key(claim_id, action, cycle_id, certificate.certificate_id)
    async with conn.transaction():
        hardened = await reservations.harden(conn, key, at)
        outbox_id = await enqueue(
            conn,
            claim_id=claim_id,
            subject_token=subject_token,
            cycle_id=cycle_id,
            action_type=action,
            channel=ACTION_CHANNEL[action].value,
            payload=payload,
            certificate_id=certificate.certificate_id,
            cert_valid_from=certificate.valid_from,
            cert_valid_until=certificate.valid_until,
            not_before=at,
            priority=priority,
        )
        await ledger.append(
            conn,
            LedgerEntry(
                event_type=LedgerEventType.CERTIFICATE_ISSUED,
                occurred_at=at,
                claim_id=claim_id,
                subject_token=subject_token,
                payload={
                    "certificate_id": str(certificate.certificate_id),
                    "action": action.value,
                    "touchpoint": 4,
                    "valid_from": certificate.valid_from,
                    "valid_until": certificate.valid_until,
                    "hardened_reservations": len(hardened),
                },
            ),
        )

    return EnqueueResult(
        outcome=Outcome.ENQUEUED,
        outbox_id=outbox_id,
        certificate_id=certificate.certificate_id,
        state=state,
        hardened=len(hardened),
    )
