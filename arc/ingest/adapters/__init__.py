"""The dialects, and the registry that maps a source name onto one.

A source with no adapter is refused, not guessed at. GI-5: unknown fails
closed, and a webhook from an unrecognised sender is the most obviously
untrusted input there is.
"""

from __future__ import annotations

from collections.abc import Mapping

from arc.ingest.adapters.base import (
    Adapter,
    SignatureInvalid,
    SignedAdapter,
    expected_signature,
    payload_hash,
    signature_timestamp,
    verify_signature,
)
from arc.ingest.adapters.billing import BillingAdapter
from arc.ingest.adapters.nach import NachAdapter
from arc.ingest.adapters.pgw import PaymentGatewayAdapter
from arc.ingest.adapters.upi import UpiAutopayAdapter

__all__ = [
    "Adapter",
    "BillingAdapter",
    "NachAdapter",
    "PaymentGatewayAdapter",
    "SignatureInvalid",
    "SignedAdapter",
    "UnknownSource",
    "UpiAutopayAdapter",
    "build_registry",
    "expected_signature",
    "payload_hash",
    "signature_timestamp",
    "verify_signature",
]

ADAPTER_TYPES = (PaymentGatewayAdapter, NachAdapter, UpiAutopayAdapter, BillingAdapter)


class UnknownSource(KeyError):
    """No adapter claims this source. The delivery is refused."""


def build_registry(secrets: Mapping[str, bytes]) -> dict[str, Adapter]:
    """One adapter per source, each with its own signing secret.

    Per-source secrets rather than one shared key: a compromise at one gateway
    must not let an attacker sign traffic that appears to come from another.
    """
    registry: dict[str, Adapter] = {}
    for adapter_type in ADAPTER_TYPES:
        source = adapter_type.source
        if source not in secrets:
            continue
        registry[source] = adapter_type(secrets[source])
    missing = set(secrets) - set(registry)
    if missing:
        raise UnknownSource(f"no adapter for source(s): {', '.join(sorted(missing))}")
    return registry
