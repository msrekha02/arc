"""NPCI NACH dialect. Bank mandate debits, returned or settled.

The file reports a settlement DATE and no instant, which is faithful to how
NACH actually works. The event is flagged `date_only` and the pipeline takes
its instant from the signed delivery rather than inventing midnight.
"""

from __future__ import annotations

from datetime import UTC, datetime

from arc.core.types import Rail
from arc.ingest.adapters.base import (
    SignedAdapter,
    paise_from_rupee_string,
    parse_json,
    payload_hash,
)
from arc.ingest.events import (
    Initiator,
    MalformedPayload,
    PersonalData,
    RawEvent,
    WireKind,
    optional,
    require,
)

SOURCE = "npci_nach"

_KINDS = {
    "ACH_DR_SETTLED": WireKind.DEBIT_CAPTURED,
    "ACH_DR_RETURN": WireKind.DEBIT_FAILED,
}


class NachAdapter(SignedAdapter):
    source = SOURCE

    def parse(self, raw: bytes) -> RawEvent:
        body = parse_json(raw)
        message = str(require(body, "message_type"))
        kind = _KINDS.get(message)
        if kind is None:
            raise MalformedPayload(f"unknown {SOURCE} message type {message!r}")

        customer = require(body, "customer")
        reference = require(body, "reference")

        return RawEvent(
            source=SOURCE,
            event_id=str(require(body, "transaction_id")),
            event_timestamp=_settlement_date(require(body, "settlement_date")),
            date_only=True,
            kind=kind,
            rail=Rail.ENACH,
            account_ref=str(require(reference, "account_id")),
            customer_ref=None,
            amount_paise=paise_from_rupee_string(require(body, "amount")),
            raw=raw,
            raw_hash=payload_hash(raw),
            succeeded=kind is WireKind.DEBIT_CAPTURED,
            attempt=int(optional(reference, "attempt") or 1),
            initiated_by=Initiator(str(optional(reference, "initiated_by") or Initiator.MERCHANT)),
            decline_code=_text(optional(body, "return_code")),
            issuer_ref=_text(optional(body, "destination_bank")),
            personal=PersonalData(
                name=_text(optional(customer, "name")),
                phone=_text(optional(customer, "mobile")),
                # The bank narration. This single field is why the redaction
                # boundary is at L1 and not only in front of the LLM.
                narration=_text(optional(body, "narration")),
                identifiers=_identifiers(body, customer),
            ),
        )


def _settlement_date(value: object) -> datetime:
    if not isinstance(value, str):
        raise MalformedPayload(f"settlement_date {value!r} is not a date string")
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError as exc:
        raise MalformedPayload(f"settlement_date {value!r} is not ISO 8601") from exc


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _identifiers(body: object, customer: object) -> dict[str, str]:
    found = {
        "mandate": optional(body, "umrn"),
        "ifsc": optional(customer, "ifsc"),
        "account_last4": optional(customer, "account_last4"),
    }
    return {key: str(value) for key, value in found.items() if value is not None}
