"""Signature verification and the adapter contract.

VERIFY BEFORE PARSE, always, and on the RAW BYTES. Deserialising first and
verifying the reconstructed object is the classic mistake: it hands untrusted
input to a parser and it compares a re-serialisation that may not be
byte-identical to what was signed.

Adapters translate and nothing else. No adapter branches on claim state, on a
cause, or on an amount - `tests/test_ingest.py` walks their ASTs and fails the
build if one starts to. WHY: gateway-specific shape is a fact about a vendor,
and policy is a fact about the system. Once one leaks into the other, a
compliance rule becomes untestable without a webhook fixture.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from arc.core.money import Paise, paise
from arc.ingest.events import MalformedPayload, RawEvent

# `t=<unix seconds>,v1=<hex digest>` - the shape PSPs use, and what the
# wire-level fake emits. The timestamp is inside the signed material, so a
# payload cannot be replayed against a different one.
SIGNATURE_RE = re.compile(r"^t=(?P<t>\d+),v1=(?P<v1>[0-9a-f]{64})$")

_RUPEES_RE = re.compile(r"^\d{1,15}(\.\d{1,2})?$")


class SignatureInvalid(ValueError):
    """The bytes did not verify. They are not parsed and never will be."""


def signed_material(timestamp: int, raw: bytes) -> bytes:
    return f"{timestamp}.".encode() + raw


def expected_signature(secret: bytes, timestamp: int, raw: bytes) -> str:
    digest = hmac.new(secret, signed_material(timestamp, raw), hashlib.sha256).hexdigest()
    return f"t={timestamp},v1={digest}"


def verify_signature(secret: bytes, raw: bytes, signature: str) -> bool:
    """Constant-time verification over the exact delivered bytes.

    Returns a bool rather than raising, because the caller has to record the
    rejection - a burst of failures is a signal, and an exception thrown from
    inside a parser loop tends to become a swallowed one.
    """
    if not isinstance(raw, bytes) or not isinstance(signature, str):
        return False
    match = SIGNATURE_RE.match(signature)
    if match is None:
        return False
    timestamp = int(match.group("t"))
    return hmac.compare_digest(expected_signature(secret, timestamp, raw), signature)


def signature_timestamp(signature: str) -> datetime | None:
    """The instant the signature claims, for the pipeline's freshness check.

    The check itself lives in the pipeline, which is handed `at` as a
    parameter. An adapter that decided freshness would need a clock, and
    nothing in this repo reads one except the Time Authority.
    """
    match = SIGNATURE_RE.match(signature or "")
    if match is None:
        return None
    return datetime.fromtimestamp(int(match.group("t")), tz=UTC)


def payload_hash(raw: bytes) -> bytes:
    return hashlib.sha256(raw).digest()


def parse_json(raw: bytes) -> Mapping[str, Any]:
    try:
        loaded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedPayload(f"payload is not JSON: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise MalformedPayload("payload is not a JSON object")
    return loaded


def paise_from_rupee_string(text: object) -> Paise:
    """Parse "2449.47" into 244947 paise, by integers only.

    Never through a float. `float("2449.47") * 100` is 244946.99999999997, and
    a rounding rule that is right almost always is exactly the silent,
    compounding error GI-2 exists to prevent.
    """
    if isinstance(text, int) and not isinstance(text, bool):
        return paise(text)
    if not isinstance(text, str) or not _RUPEES_RE.match(text.strip()):
        raise MalformedPayload(f"amount {text!r} is not a rupee decimal")
    whole, _, fraction = text.strip().partition(".")
    return paise(int(whole) * 100 + int((fraction + "00")[:2]))


def epoch_to_utc(value: object) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MalformedPayload(f"timestamp {value!r} is not unix seconds")
    return datetime.fromtimestamp(value, tz=UTC)


def iso_to_utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise MalformedPayload(f"timestamp {value!r} is not an ISO string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MalformedPayload(f"timestamp {value!r} is not ISO 8601") from exc
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@runtime_checkable
class Adapter(Protocol):
    """One gateway dialect. Verification, then translation. Nothing else."""

    source: str

    def verify(self, raw: bytes, sig: str) -> bool: ...

    def parse(self, raw: bytes) -> RawEvent: ...


class SignedAdapter:
    """Shared HMAC verification, so no dialect reimplements it slightly wrong."""

    source: str = "unknown"

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("adapter secret must be at least 16 bytes")
        self._secret = secret

    def verify(self, raw: bytes, sig: str) -> bool:
        return verify_signature(self._secret, raw, sig)

    def parse(self, raw: bytes) -> RawEvent:  # pragma: no cover - overridden
        raise NotImplementedError
