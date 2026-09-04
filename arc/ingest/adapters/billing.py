"""Billing dialect. B2B invoices that aged past their due date.

An invoice does not decline, so there is no response code here - what it
carries instead is an ageing bucket, which is the same fact in a different
vocabulary and normalises onto the same claim.
"""

from __future__ import annotations

from arc.core.types import Rail
from arc.ingest.adapters.base import (
    SignedAdapter,
    epoch_to_utc,
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

SOURCE = "billing"

_KINDS = {"invoice.overdue": WireKind.INVOICE_OVERDUE}


class BillingAdapter(SignedAdapter):
    source = SOURCE

    def parse(self, raw: bytes) -> RawEvent:
        body = parse_json(raw)
        name = str(require(body, "type"))
        kind = _KINDS.get(name)
        if kind is None:
            raise MalformedPayload(f"unknown {SOURCE} event {name!r}")

        invoice = require(body, "invoice")
        buyer = require(invoice, "buyer")
        reference = require(body, "ref")
        overdue = optional(invoice, "days_overdue")

        return RawEvent(
            source=SOURCE,
            event_id=str(require(invoice, "id")),
            event_timestamp=epoch_to_utc(require(body, "created_at")),
            kind=kind,
            rail=Rail.INVOICE,
            account_ref=str(require(reference, "account_id")),
            customer_ref=None,
            amount_paise=paise_from_rupee_string(require(invoice, "amount_due")),
            raw=raw,
            raw_hash=payload_hash(raw),
            succeeded=False,
            attempt=int(optional(reference, "attempt") or 1),
            initiated_by=Initiator.MERCHANT,
            ageing_bucket=_text(optional(invoice, "ageing_bucket")),
            days_overdue=None if overdue is None else int(overdue),
            personal=PersonalData(
                name=_text(optional(buyer, "name")),
                email=_text(optional(buyer, "email")),
                phone=_text(optional(buyer, "contact")),
                identifiers=_identifiers(buyer),
            ),
        )


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _identifiers(buyer: object) -> dict[str, str]:
    gstin = optional(buyer, "gstin")
    return {} if gstin is None else {"gstin": str(gstin)}
