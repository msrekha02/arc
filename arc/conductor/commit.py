"""`commit_decision` - one transaction, four writes, all or nothing.

    FSM transition
    hard budget reservation
    ledger append
    outbox insert ON CONFLICT (idempotency_key) DO NOTHING

THIS IS THE GUARANTEE. Not the outbox table, not the workers, not the
idempotency key - those make dispatch safe. What makes the SYSTEM safe is that
a claim cannot be marked in-treatment without the budget being held, the
decision being recorded, and the intent to act existing, because all four are
one commit. A crash anywhere in here leaves the claim exactly where it was,
holding nothing, with nothing queued and nothing in the ledger.

WHAT COULD GO WRONG IF THEY WERE SEPARATE, concretely: the FSM moves, then the
process dies. The claim now reads IN_TREATMENT forever, no outbox row exists so
nothing will ever dispatch it, and no reaper looks at claims in that state
because nothing is leased. It is a silent, permanent leak of one customer's
recovery, and it does not show up in any counter.

ORDERING INSIDE THE TRANSACTION IS DELIBERATE. The reservation goes before the
ledger append because a refused budget should not leave a decision recorded as
taken. The outbox insert goes last because it is the only write that is
idempotent on its own, so it is the safest one to repeat.

THE LEDGER'S ADVISORY LOCK IS HELD TO THE OUTERMOST COMMIT. That serialises
concurrent `commit_decision` calls at the chain append, which is correct: a
hash chain has one head, and two writers extending it concurrently is exactly
what the lock exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from arc.allocator.budgets import CostVector, cost_of
from arc.conductor import reservations
from arc.conductor.fsm import transition
from arc.conductor.outbox import enqueue, idempotency_key
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType, ClaimState
from arc.gate.context import ACTION_CHANNEL
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType


class UncertifiedAction(PermissionError):
    """An action reached the executor boundary without a valid ALLOW (GI-1)."""


@dataclass(frozen=True)
class CommitRequest:
    """Everything one decision needs to become durable state."""

    claim_id: UUID
    subject_token: str
    cycle_id: UUID
    action: ActionType
    certificate: Certificate
    decision_time: datetime
    planned_execution_time: datetime
    pi_intended: float
    shadow_prices: dict[str, float]
    payload: dict[str, Any]
    from_state: ClaimState = ClaimState.PLANNED
    to_state: ClaimState = ClaimState.IN_TREATMENT
    model_versions: dict[str, str] | None = None
    feature_hash: str | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        ensure_utc(self.decision_time)
        ensure_utc(self.planned_execution_time)

    @property
    def cost(self) -> CostVector:
        return cost_of(self.action)

    @property
    def horizon(self) -> timedelta:
        return self.planned_execution_time - self.decision_time

    @property
    def key(self) -> str:
        return idempotency_key(
            self.claim_id, self.action, self.cycle_id, self.certificate.certificate_id
        )


@dataclass(frozen=True)
class CommitResult:
    idempotency_key: str
    outbox_id: int | None
    ledger_seq: int
    reservations: int
    reservation_tier: reservations.ReservationStatus
    already_enqueued: bool


async def commit_decision(
    conn: Any,
    request: CommitRequest,
    *,
    ledger: DecisionLedger | None = None,
) -> CommitResult:
    """The atomic commit. Raises rather than partially applying.

    GI-1 IS ASSERTED HERE, at the boundary where an action first becomes real.
    A certificate that is not an ALLOW, or one whose window does not cover the
    planned execution time, never reaches the outbox. There is no override.
    """
    ledger = ledger or DecisionLedger()
    certificate = request.certificate

    if certificate.decision is not Verdict.ALLOW:
        raise UncertifiedAction(
            f"claim {request.claim_id}: certificate decision is {certificate.decision}, "
            f"not ALLOW (blocking rules: {list(certificate.blocking_rule_ids)})"
        )
    if certificate.action is not request.action:
        raise UncertifiedAction(
            f"certificate authorises {certificate.action}, not {request.action}"
        )
    if not certificate.is_valid_at(request.planned_execution_time):
        raise UncertifiedAction(
            f"certificate window [{certificate.valid_from.isoformat()}, "
            f"{certificate.valid_until.isoformat()}] does not cover the planned "
            f"execution time {request.planned_execution_time.isoformat()}; a decision "
            "cannot be committed against authorisation that will already have expired"
        )

    key = request.key
    tier = reservations.tier_for(request.horizon)

    async with conn.transaction():
        # 1. FSM. Conditional on the state the caller believes it is in, so a
        #    concurrent mover is detected rather than overwritten.
        await transition(
            conn,
            request.claim_id,
            frm=request.from_state,
            to=request.to_state,
            at=request.decision_time,
        )

        # 2. Budget. Reserved, never checked - and before the ledger, so a
        #    refusal leaves no record of a decision that did not happen.
        held = await reservations.reserve(
            conn,
            cycle_id=request.cycle_id,
            claim_id=request.claim_id,
            subject_token=request.subject_token,
            cost=request.cost,
            idempotency_key=key,
            at=request.decision_time,
            horizon=request.horizon,
        )

        # 3. Ledger. Structured fields only; the guard refuses anything else.
        ref = await ledger.append(
            conn,
            LedgerEntry(
                event_type=LedgerEventType.DECISION,
                occurred_at=request.decision_time,
                claim_id=request.claim_id,
                subject_token=request.subject_token,
                payload={
                    "cycle_id": str(request.cycle_id),
                    "intended_action": request.action.value,
                    "pi_intended": request.pi_intended,
                    "planned_execution_time": request.planned_execution_time.isoformat(),
                    "certificate_id": str(certificate.certificate_id),
                    "rule_registry_version": certificate.rule_registry_version,
                    "cert_valid_from": certificate.valid_from.isoformat(),
                    "cert_valid_until": certificate.valid_until.isoformat(),
                    "shadow_prices": request.shadow_prices,
                    "reservation_tier": tier.value,
                    "idempotency_key": key,
                    "model_versions": request.model_versions or {},
                    "feature_hash": request.feature_hash or "",
                },
            ),
        )

        # 4. Outbox, last and idempotent.
        outbox_id = await enqueue(
            conn,
            claim_id=request.claim_id,
            subject_token=request.subject_token,
            cycle_id=request.cycle_id,
            action_type=request.action,
            channel=ACTION_CHANNEL[request.action].value,
            payload=request.payload,
            certificate_id=certificate.certificate_id,
            cert_valid_from=certificate.valid_from,
            cert_valid_until=certificate.valid_until,
            not_before=request.planned_execution_time,
            priority=request.priority,
        )

    return CommitResult(
        idempotency_key=key,
        outbox_id=outbox_id,
        ledger_seq=ref.seq,
        reservations=len(held),
        reservation_tier=tier,
        already_enqueued=outbox_id is None,
    )
