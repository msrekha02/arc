"""The immutable, hash-chained, pseudonymous audit trail.

    h_n = SHA256(h_{n-1} || canonical_json(entry_n))

Two properties make this coexist with a right to erasure (GI-4):

* the chain only ever covers pseudonymous tokens and structured fields, so
  destroying a subject's plaintext leaves every hash intact;
* nothing reaches it without passing the PII write-guard first.

Every decision is therefore reconstructible in structure, rationale and rule
basis from the ledger alone. Personal content is reconstructible only while the
Subject Store still holds it, which is the honest version of the claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any
from uuid import UUID

from arc.core.time_authority import ensure_utc
from arc.ledger.pii_guard import PIIGuard

GENESIS_HASH = b"\x00" * 32
HASH_BYTES = 32

# Serialises appenders so the chain has one unambiguous order. Postgres
# advisory locks are transaction scoped, so a crashed appender releases it.
CHAIN_LOCK_KEY = 0x0A2C1E01


class LedgerEventType(StrEnum):
    """What happened. Extended additively by later milestones."""

    CLAIM_DETECTED = "claim_detected"
    CLAIM_DIAGNOSED = "claim_diagnosed"
    STATE_TRANSITION = "state_transition"
    DECISION = "decision"
    CERTIFICATE_ISSUED = "certificate_issued"
    GATE_VETO = "gate_veto"
    ABANDONED_UNEXECUTED = "abandoned_unexecuted"
    CHANNEL_DISPATCHED = "channel_dispatched"
    MONEY_TRANSITION = "money_transition"
    PROMISE_OUTCOME = "promise_outcome"
    TOMBSTONE = "tombstone"


def _encode(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return ensure_utc(value).isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(_encode(item) for item in value)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{type(value).__name__} is not ledgerable")


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Byte-stable JSON: sorted keys, no whitespace, ASCII-escaped.

    The bytes this returns are the bytes that get hashed and the bytes that get
    stored, so verification never has to reproduce a serialisation decision.
    """
    return json.dumps(
        payload,
        default=_encode,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class LedgerEntry:
    """One append. `payload` is structured fields only, never free text."""

    event_type: LedgerEventType
    occurred_at: datetime
    claim_id: UUID | None = None
    subject_token: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ensure_utc(self.occurred_at)
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")

    def scannable(self) -> dict[str, Any]:
        """Everything the write-guard inspects, before a sequence is assigned."""
        return {
            "event_type": self.event_type,
            "claim_id": self.claim_id,
            "subject_token": self.subject_token,
            "payload": dict(self.payload),
        }

    def body(self, seq: int) -> dict[str, Any]:
        """The hashed body. `seq` is inside it, so reordering breaks the hash."""
        return {
            "seq": seq,
            "event_type": self.event_type,
            "occurred_at": self.occurred_at,
            "claim_id": self.claim_id,
            "subject_token": self.subject_token,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class LedgerRef:
    seq: int
    entry_hash: bytes
    prev_hash: bytes


@dataclass(frozen=True)
class ChainBreak:
    """Where verification failed and why. Reported, never swallowed."""

    seq: int
    reason: str


def entry_hash(prev_hash: bytes, body_canonical: bytes) -> bytes:
    if len(prev_hash) != HASH_BYTES:
        raise ValueError(f"prev_hash must be {HASH_BYTES} bytes")
    return hashlib.sha256(prev_hash + body_canonical).digest()


class DecisionLedger:
    """Append-only. There is no update method and no delete method."""

    def __init__(self, guard: PIIGuard | None = None) -> None:
        self._guard = guard or PIIGuard()

    async def append(self, conn: Any, entry: LedgerEntry) -> LedgerRef:
        """Scan, then chain, then write. The scan raises before anything else.

        The guard runs outside the transaction on purpose: a refused write must
        not leave a rolled-back transaction behind that a caller could mistake
        for a transient failure and retry.
        """
        self._guard.scan(entry.scannable())

        async with conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock($1)", CHAIN_LOCK_KEY)
            tail = await conn.fetchrow(
                "SELECT seq, entry_hash FROM decision_ledger ORDER BY seq DESC LIMIT 1"
            )
            seq = (tail["seq"] + 1) if tail else 1
            prev_hash = tail["entry_hash"] if tail else GENESIS_HASH

            canonical = canonical_json(entry.body(seq))
            digest = entry_hash(prev_hash, canonical)

            await conn.execute(
                """
                INSERT INTO decision_ledger
                    (seq, claim_id, subject_token, event_type, occurred_at,
                     body_canonical, prev_hash, entry_hash)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                seq,
                entry.claim_id,
                entry.subject_token,
                entry.event_type.value,
                entry.occurred_at,
                canonical.decode("utf-8"),
                prev_hash,
                digest,
            )

        return LedgerRef(seq=seq, entry_hash=digest, prev_hash=prev_hash)

    async def verify_chain(self, conn: Any, frm: int, to: int) -> bool:
        return not await self.find_breaks(conn, frm, to)

    async def find_breaks(self, conn: Any, frm: int, to: int) -> list[ChainBreak]:
        """Every inconsistency in `[frm, to]`, for a report rather than a bool.

        Three things are checked independently, because an attacker who fixes
        one still trips another: the recomputed hash of each stored body, the
        backward link to the previous entry, and sequence contiguity.
        """
        if frm < 1 or to < frm:
            raise ValueError(f"bad range [{frm}, {to}]")

        if frm == 1:
            expected_prev = GENESIS_HASH
        else:
            anchor = await conn.fetchrow(
                "SELECT entry_hash FROM decision_ledger WHERE seq = $1", frm - 1
            )
            if anchor is None:
                return [ChainBreak(frm - 1, "anchor entry is missing")]
            expected_prev = anchor["entry_hash"]

        rows = await conn.fetch(
            """
            SELECT seq, body_canonical, prev_hash, entry_hash
              FROM decision_ledger
             WHERE seq BETWEEN $1 AND $2
             ORDER BY seq
            """,
            frm,
            to,
        )

        breaks: list[ChainBreak] = []
        expected_seq = frm

        for row in rows:
            if row["seq"] != expected_seq:
                breaks.append(ChainBreak(row["seq"], f"sequence gap, expected {expected_seq}"))
            expected_seq = row["seq"] + 1

            if row["prev_hash"] != expected_prev:
                breaks.append(ChainBreak(row["seq"], "prev_hash does not match the entry before"))

            recomputed = entry_hash(row["prev_hash"], row["body_canonical"].encode("utf-8"))
            if recomputed != row["entry_hash"]:
                breaks.append(ChainBreak(row["seq"], "body does not hash to the stored entry_hash"))

            body = json.loads(row["body_canonical"])
            if body.get("seq") != row["seq"]:
                breaks.append(ChainBreak(row["seq"], "sequence inside the body was altered"))

            expected_prev = row["entry_hash"]

        return breaks

    async def replay(self, conn: Any, claim_id: UUID) -> list[LedgerEntry]:
        """Every entry for one claim, in the order it was written."""
        rows = await conn.fetch(
            """
            SELECT body_canonical FROM decision_ledger
             WHERE claim_id = $1 ORDER BY seq
            """,
            claim_id,
        )
        return [_entry_from_body(json.loads(row["body_canonical"])) for row in rows]

    async def head(self, conn: Any) -> LedgerRef | None:
        row = await conn.fetchrow(
            "SELECT seq, entry_hash, prev_hash FROM decision_ledger ORDER BY seq DESC LIMIT 1"
        )
        if row is None:
            return None
        return LedgerRef(row["seq"], row["entry_hash"], row["prev_hash"])

    async def entries_of_type(self, conn: Any, event_type: LedgerEventType) -> list[LedgerEntry]:
        rows = await conn.fetch(
            "SELECT body_canonical FROM decision_ledger WHERE event_type = $1 ORDER BY seq",
            event_type.value,
        )
        return [_entry_from_body(json.loads(row["body_canonical"])) for row in rows]


def _entry_from_body(body: Mapping[str, Any]) -> LedgerEntry:
    claim = body.get("claim_id")
    return LedgerEntry(
        event_type=LedgerEventType(body["event_type"]),
        occurred_at=datetime.fromisoformat(body["occurred_at"]),
        claim_id=UUID(claim) if claim else None,
        subject_token=body.get("subject_token"),
        payload=body.get("payload", {}),
    )


def chain_over(bodies: Sequence[Mapping[str, Any]]) -> bytes:
    """Fold a sequence of bodies into a head hash. Used by offline checks."""
    digest = GENESIS_HASH
    for body in bodies:
        digest = entry_hash(digest, canonical_json(body))
    return digest
