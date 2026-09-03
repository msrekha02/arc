"""Identifier derivation.

Two rules govern every identifier in the system:

1. A subject is referred to downstream ONLY by `subject_token`, a one-way
   derivation of the raw identifier. The raw identifier must not travel past
   the normaliser, because everything downstream can reach the immutable
   Decision Ledger, and a raw identifier written into a hash chain cannot
   later be erased (GI-4).

2. Identifiers that must survive a retry are derived, not generated. A random
   UUID per attempt would defeat deduplication at the adapter boundary.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from uuid import NAMESPACE_DNS, UUID, uuid5

SUBJECT_TOKEN_PREFIX = "sub_"
SUBJECT_TOKEN_DIGITS = 32
SUBJECT_TOKEN_RE = re.compile(rf"^{SUBJECT_TOKEN_PREFIX}[0-9a-f]{{{SUBJECT_TOKEN_DIGITS}}}$")

# Stable forever. Derived rather than pasted so their provenance is readable.
ARC_NAMESPACE = uuid5(NAMESPACE_DNS, "arc.revenue-continuity")
CLAIM_NAMESPACE = uuid5(ARC_NAMESPACE, "claim")


def subject_token(raw_identifier: str, *, pepper: bytes) -> str:
    """Derive a subject's pseudonymous token. Deterministic and one-way.

    The pepper is injected rather than read from the environment inside this
    function, so a test and a production run differ only by an argument, and
    nothing derives a token without a caller having chosen a key.
    """
    if not isinstance(raw_identifier, str) or not raw_identifier.strip():
        raise ValueError("raw_identifier must be a non-empty string")
    if not isinstance(pepper, bytes) or len(pepper) < 16:
        raise ValueError("pepper must be at least 16 bytes of key material")

    normalised = raw_identifier.strip().casefold().encode("utf-8")
    digest = hmac.new(pepper, normalised, hashlib.sha256).hexdigest()
    return f"{SUBJECT_TOKEN_PREFIX}{digest[:SUBJECT_TOKEN_DIGITS]}"


def is_subject_token(value: object) -> bool:
    """True only for a well-formed derived token, never for a raw identifier."""
    return isinstance(value, str) and SUBJECT_TOKEN_RE.match(value) is not None


def deterministic_uuid(namespace: UUID, *parts: str) -> UUID:
    """UUIDv5 over a delimited join, so ('a','bc') and ('ab','c') differ."""
    if not parts:
        raise ValueError("at least one part is required")
    return uuid5(namespace, "\x1f".join(parts))


def claim_id(source: str, event_id: str) -> UUID:
    """Derive a claim's id from the event that produced it.

    A redelivered webhook therefore produces the same claim id, which is what
    makes adapter-level dedupe idempotent rather than merely likely.
    """
    return deterministic_uuid(CLAIM_NAMESPACE, source, event_id)
