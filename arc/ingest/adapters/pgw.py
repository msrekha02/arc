"""Card gateway dialect. Payments and abandoned checkouts.

Translation only. This module does not know what a claim is.
"""

from __future__ import annotations

from arc.core.money import paise
from arc.core.types import Rail
from arc.ingest.adapters.base import (
    SignedAdapter,
    epoch_to_utc,
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

SOURCE = "pgw"

_KINDS = {
    "payment.captured": WireKind.DEBIT_CAPTURED,
    "payment.failed": WireKind.DEBIT_FAILED,
    "checkout.abandoned": WireKind.CHECKOUT_ABANDONED,
}


class PaymentGatewayAdapter(SignedAdapter):
    source = SOURCE

    def parse(self, raw: bytes) -> RawEvent:
        body = parse_json(raw)
        name = require(body, "event")
        kind = _KINDS.get(str(name))
        if kind is None:
            raise MalformedPayload(f"unknown {SOURCE} event {name!r}")

        # An abandoned checkout is an order, a debit is a payment. Two shapes,
        # one envelope, and picking the right key is translation.
        holder = "order" if kind is WireKind.CHECKOUT_ABANDONED else "payment"
        entity = require(body, "payload", holder)
        customer = require(entity, "customer")

        return RawEvent(
            source=SOURCE,
            event_id=str(require(entity, "id")),
            event_timestamp=epoch_to_utc(require(body, "created_at")),
            kind=kind,
            rail=Rail.CARD,
            account_ref=str(require(entity, "notes", "account_id")),
            customer_ref=str(require(customer, "id")),
            amount_paise=paise(int(require(entity, "amount"))),
            raw=raw,
            raw_hash=payload_hash(raw),
            succeeded=str(optional(entity, "status")) == "captured",
            attempt=int(optional(entity, "notes", "attempt") or optional(entity, "attempts") or 1),
            initiated_by=Initiator(
                str(optional(entity, "notes", "initiated_by") or Initiator.MERCHANT)
            ),
            decline_code=_text(optional(entity, "error_reason")),
            advice_code=_text(optional(entity, "merchant_advice_code")),
            issuer_ref=_text(optional(entity, "card", "issuer")),
            personal=PersonalData(
                name=_text(optional(customer, "name")),
                email=_text(optional(customer, "email")),
                phone=_text(optional(customer, "contact")),
                narration=_text(optional(entity, "error_description")),
                identifiers=_identifiers(optional(entity, "card", "last4")),
            ),
        )


def _text(value: object) -> str | None:
    return None if value is None else str(value)


def _identifiers(last4: object) -> dict[str, str]:
    return {} if last4 is None else {"card_last4": str(last4)}
