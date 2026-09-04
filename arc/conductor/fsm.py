"""Claim state transitions, against the database.

M1 owns the transition table. This module does not restate it, it applies it -
there is exactly one `LEGAL_TRANSITIONS` in the repo and a second copy here
would be a second answer to "may this claim move", which is the drift that
makes a state machine decorative.

TWO CHECKS, NOT ONE. The legality check is in Python against M1's table, and
the UPDATE is conditional on the row still being in the state we read. The
second one is not redundant: between reading a claim and writing it, another
worker may have moved it, and an unconditional UPDATE would silently overwrite
that. A conditional UPDATE that matches nothing tells us the world changed, and
we raise instead of clobbering.

FORBORNE and WRITTEN_OFF are absorbing, and the table has no outgoing edges
from either. No expected-value argument reopens hardship.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from arc.core.time_authority import ensure_utc
from arc.core.types import (
    ABSORBING_STATES,
    LEGAL_TRANSITIONS,
    ClaimState,
    IllegalTransition,
)


class ClaimNotFound(LookupError):
    """A transition was attempted against a claim that is not there."""


class ConcurrentTransition(RuntimeError):
    """The claim moved underneath us between read and write."""


@dataclass(frozen=True)
class TransitionRecord:
    claim_id: UUID
    frm: ClaimState
    to: ClaimState
    at: datetime


async def current_state(conn: Any, claim_id: UUID) -> ClaimState:
    row = await conn.fetchval("SELECT state FROM claims WHERE claim_id = $1", claim_id)
    if row is None:
        raise ClaimNotFound(f"claim {claim_id} does not exist")
    return ClaimState(row)


async def transition(
    conn: Any,
    claim_id: UUID,
    *,
    frm: ClaimState,
    to: ClaimState,
    at: datetime,
) -> TransitionRecord:
    """Move a claim along a legal edge, or raise.

    `frm` is passed in rather than read, so the caller states the state it
    believes the claim is in and gets told when it is wrong. A transition that
    silently adapts to whatever it finds is not a transition, it is an
    assignment.
    """
    ensure_utc(at)

    if not isinstance(to, ClaimState) or not isinstance(frm, ClaimState):
        raise IllegalTransition(f"{frm!r} -> {to!r} is not a pair of ClaimStates")

    onward = LEGAL_TRANSITIONS[frm]
    if to not in onward:
        if frm in ABSORBING_STATES:
            raise IllegalTransition(
                f"{frm} is absorbing; no transition out is legal (attempted -> {to})"
            )
        legal = ", ".join(sorted(onward)) or "nothing"
        raise IllegalTransition(f"{frm} -> {to} is not legal; legal: {legal}")

    updated = await conn.fetchval(
        """
        UPDATE claims
           SET state = $3, updated_at = $4
         WHERE claim_id = $1
           AND state = $2
        RETURNING claim_id
        """,
        claim_id,
        frm.value,
        to.value,
        at,
    )
    if updated is None:
        actual = await conn.fetchval("SELECT state FROM claims WHERE claim_id = $1", claim_id)
        if actual is None:
            raise ClaimNotFound(f"claim {claim_id} does not exist")
        raise ConcurrentTransition(
            f"claim {claim_id} was {actual} rather than the expected {frm}; "
            "another worker moved it and this transition would have overwritten that"
        )

    return TransitionRecord(claim_id=claim_id, frm=frm, to=to, at=at)


async def transition_from_current(
    conn: Any, claim_id: UUID, *, to: ClaimState, at: datetime
) -> TransitionRecord:
    """Read the state under a row lock, then move. For callers that do not
    already know where the claim is.

    The `FOR UPDATE` matters: without it two workers read the same state, both
    compute a legal edge, and the second one's conditional UPDATE fails - which
    is safe but noisy. Locking first makes the second one wait and then see the
    truth.
    """
    frm = await conn.fetchval("SELECT state FROM claims WHERE claim_id = $1 FOR UPDATE", claim_id)
    if frm is None:
        raise ClaimNotFound(f"claim {claim_id} does not exist")
    return await transition(conn, claim_id, frm=ClaimState(frm), to=to, at=at)
