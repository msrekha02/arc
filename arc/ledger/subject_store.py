"""The mutable, erasable side of the boundary. Everything personal lives here.

An append-only hash chain and a right to erasure are directly contradictory, so
they are not the same store. Names, phone numbers, transcripts and raw bank
narrations live here under a per-subject data key. Erasure destroys that key.
The row stays, unreadable, and the Decision Ledger's chain is untouched because
it never covered the plaintext.

The erasure itself is an audit obligation, so shredding appends a TOMBSTONE to
the Decision Ledger recording that it happened, when, at whose request, and
which refs were destroyed.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from arc.core.ids import is_subject_token
from arc.core.time_authority import ensure_utc
from arc.ledger.decision_ledger import (
    DecisionLedger,
    LedgerEntry,
    LedgerEventType,
    LedgerRef,
    canonical_json,
)

KEY_BYTES = 32
NONCE_BYTES = 12

# The one random source in the repo that is NOT seeded and injected. A
# deterministic data key would make the ciphertext reproducible from the seed,
# which is precisely what crypto-shredding must prevent. The determinism
# convention exists so simulation and policy sampling replay identically; it
# was never about key material. Tests inject a key source rather than a seed.
KeySource = Callable[[int], bytes]


class SubjectErased(LookupError):
    """The key for this subject was destroyed. The row cannot be read again."""


class SubjectUnknown(LookupError):
    """No record under that ref."""


@dataclass(frozen=True)
class TombstoneRef:
    """Proof that an erasure happened, and what it covered."""

    subject_token: str
    refs_destroyed: tuple[str, ...]
    shredded_at: datetime
    ledger_ref: LedgerRef


class SubjectStore:
    """Per-subject envelope encryption. One key, many records, one shred."""

    def __init__(
        self,
        ledger: DecisionLedger,
        key_source: KeySource = secrets.token_bytes,
    ) -> None:
        self._ledger = ledger
        self._key_source = key_source

    async def put(self, conn: Any, token: str, data: Mapping[str, Any]) -> str:
        """Store personal data and return the ref that the Claim carries.

        The ref is all that travels downstream. It is pseudonymous, so it is
        safe in the ledger, and it resolves to nothing once the key is gone.
        """
        if not is_subject_token(token):
            raise ValueError(f"{token!r} is not a derived subject token")

        async with conn.transaction():
            key = await self._ensure_key(conn, token)
            nonce = self._key_source(NONCE_BYTES)
            ciphertext = AESGCM(key).encrypt(nonce, canonical_json(data), token.encode("utf-8"))

            seq = await conn.fetchval(
                """
                SELECT coalesce(max(record_seq), 0) + 1
                  FROM subject_records WHERE subject_token = $1
                """,
                token,
            )
            ref = f"subject://{token}/{seq}"
            await conn.execute(
                """
                INSERT INTO subject_records
                    (ref, subject_token, record_seq, nonce, ciphertext)
                VALUES ($1, $2, $3, $4, $5)
                """,
                ref,
                token,
                seq,
                nonce,
                ciphertext,
            )
        return ref

    async def get(self, conn: Any, ref: str) -> Mapping[str, Any]:
        """Read personal data back, or fail because it was erased."""
        row = await conn.fetchrow(
            """
            SELECT r.subject_token, r.nonce, r.ciphertext, k.data_key, k.shredded_at
              FROM subject_records r
              JOIN subject_keys k ON k.subject_token = r.subject_token
             WHERE r.ref = $1
            """,
            ref,
        )
        if row is None:
            raise SubjectUnknown(f"no subject record at {ref}")
        if row["data_key"] is None:
            raise SubjectErased(
                f"{ref} was crypto-shredded at {row['shredded_at']}; the key no longer exists"
            )

        try:
            plaintext = AESGCM(row["data_key"]).decrypt(
                row["nonce"], row["ciphertext"], row["subject_token"].encode("utf-8")
            )
        except InvalidTag as exc:  # pragma: no cover - tamper path
            raise SubjectUnknown(f"{ref} failed authentication; ciphertext was altered") from exc

        return json.loads(plaintext)

    async def crypto_shred(
        self, conn: Any, token: str, *, at: datetime, requested_by: str
    ) -> TombstoneRef:
        """Destroy the key, then record that it happened.

        The record is appended after the key is gone, in the same transaction,
        so there is no window in which the ledger claims an erasure that did
        not complete, nor one that completed without being recorded.
        """
        ensure_utc(at)
        if not is_subject_token(token):
            raise ValueError(f"{token!r} is not a derived subject token")

        async with conn.transaction():
            refs = [
                row["ref"]
                for row in await conn.fetch(
                    "SELECT ref FROM subject_records WHERE subject_token = $1 ORDER BY record_seq",
                    token,
                )
            ]
            shredded = await conn.fetchval(
                """
                UPDATE subject_keys
                   SET data_key = NULL, shredded_at = $2
                 WHERE subject_token = $1 AND data_key IS NOT NULL
             RETURNING subject_token
                """,
                token,
                at,
            )
            already_shredded = shredded is None
            if already_shredded:
                existing = await conn.fetchval(
                    "SELECT 1 FROM subject_keys WHERE subject_token = $1", token
                )
                if existing is None:
                    raise SubjectUnknown(f"no subject keyed {token}")

            # Counts and refs only. The refs contain the pseudonymous token and
            # nothing else, which is why this is ledgerable at all.
            ref = await self._ledger.append(
                conn,
                LedgerEntry(
                    event_type=LedgerEventType.TOMBSTONE,
                    occurred_at=at,
                    subject_token=token,
                    payload={
                        "reason": "erasure_request",
                        "requested_by": requested_by,
                        "refs_destroyed": refs,
                        "refs_destroyed_count": len(refs),
                        "key_destroyed": not already_shredded,
                    },
                ),
            )

        return TombstoneRef(
            subject_token=token,
            refs_destroyed=tuple(refs),
            shredded_at=at,
            ledger_ref=ref,
        )

    async def is_shredded(self, conn: Any, token: str) -> bool:
        return bool(
            await conn.fetchval(
                "SELECT shredded_at IS NOT NULL FROM subject_keys WHERE subject_token = $1",
                token,
            )
        )

    async def _ensure_key(self, conn: Any, token: str) -> bytes:
        key = await conn.fetchval(
            "SELECT data_key FROM subject_keys WHERE subject_token = $1", token
        )
        if key is not None:
            return key

        shredded = await conn.fetchval(
            "SELECT shredded_at FROM subject_keys WHERE subject_token = $1", token
        )
        if shredded is not None:
            raise SubjectErased(
                f"{token} was erased at {shredded}; a new key would defeat the erasure"
            )

        key = self._key_source(KEY_BYTES)
        await conn.execute(
            "INSERT INTO subject_keys (subject_token, data_key) VALUES ($1, $2)", token, key
        )
        return key
