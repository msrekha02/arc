"""UPI Autopay dialect. Mandate debits against a virtual payment address."""

from __future__ import annotations

from arc.core.money import paise
from arc.core.types import Rail
from arc.ingest.adapters.base import (
    SignedAdapter,
    iso_to_utc,
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

SOURCE = "upi_autopay"

_KINDS = {
    "autopay.debit.success": WireKind.DEBIT_CAPTURED,
    "autopay.debit.failed": WireKind.DEBIT_FAILED,
}


class UpiAutopayAdapter(SignedAdapter):
    source = SOURCE

    def parse(self, raw: bytes) -> RawEvent:
        body = parse_json(raw)
        name = str(require(body, "event"))
        kind = _KINDS.get(name)
        if kind is None:
            raise MalformedPayload(f"unknown {SOURCE} event {name!r}")

        payer = require(body, "payer")
        reference = require(body, "ref")

        return RawEvent(
            source=SOURCE,
            event_id=str(require(body, "txn_id")),
            event_timestamp=iso_to_utc(require(body, "timestamp")),
            kind=kind,
            rail=Rail.UPI_AUTOPAY,
            account_ref=str(require(reference, "account_id")),
            customer_ref=None,
            amount_paise=paise(int(require(body, "amount_paise"))),
            raw=raw,
            raw_hash=payload_hash(raw),
            succeeded=kind is WireKind.DEBIT_CAPTURED,
            attempt=int(optional(reference, "attempt") or 1),
            initiated_by=Initiator(str(optional(reference, "initiated_by") or Initiator.MERCHANT)),
            decline_code=_text(optional(body, "npci_error_code")),
            issuer_ref=_text(optional(body, "issuer")),
            personal=PersonalData(
                name=_text(optional(payer, "name")),
                phone=_text(optional(payer, "mobile")),
                narration=_text(optional(body, "npci_error_desc")),
                identifiers=_identifiers(body, payer),
            ),
        )


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _identifiers(body: object, payer: object) -> dict[str, str]:
    found = {"mandate": optional(body, "mandate_id"), "vpa": optional(payer, "vpa")}
    return {key: str(value) for key, value in found.items() if value is not None}
