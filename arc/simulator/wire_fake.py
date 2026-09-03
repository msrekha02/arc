"""The wire-level fake: gateway-shaped webhooks, signed, duplicated, late.

WHY this is a separate component from the world, and not a method on it: they
fail differently. The world is wrong when its behavioural constants are wrong.
The fake is wrong when its payload shape drifts from the gateway's. Merging
them couples the L0 adapter to simulator internals, and the claim that ARC was
built against production-shaped input quietly stops being true.

What the adapter at M5 has to survive here is exactly what it has to survive in
production:

  * a signature it must verify BEFORE it parses anything
  * 2% duplicate delivery, because gateways redeliver on ack timeout
  * 3% late delivery, which arrives out of order with respect to event time
  * three different dialects for the same underlying fact
  * free text with a real name in it, which must never reach the ledger

Two modes. `replay` is anchored to the world's epoch and is byte-identical for
a given seed, three runs out of three - that is what lets a judge say "run it
again" and get the same number. `live` is anchored to an injected `now` and
jitters around it, for the "watch it react" beat. Live requires the caller to
pass the time, because nothing in this package reads a clock.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from arc.core.types import Rail
from arc.simulator import codes as code_book
from arc.simulator.seeds import Stream, rng, stable_hash
from arc.simulator.world import BatchEvent, EventKind, World

Mode = Literal["replay", "live"]

# source: gateway webhook redelivery on acknowledgement timeout. Two percent
# is the order of magnitude PSPs document for at-least-once delivery.
DUPLICATE_RATE = 0.02

# source: webhook delivery-lag distributions, where a small tail arrives long
# after the event and therefore out of order relative to what followed it.
OUT_OF_ORDER_RATE = 0.03

# Normal delivery lag, and the lag applied to the late tail. The late lag
# comfortably exceeds the spacing between events, so a late delivery really
# does arrive after events that happened later than it.
DELIVERY_LAG = timedelta(seconds=4)
DELIVERY_JITTER_SECONDS = 6
LATE_DELIVERY_LAG = timedelta(hours=9)

# An adapter can tell a late delivery from a prompt one by comparing arrival
# against the event timestamp in the body. Anything beyond this is late.
LATE_THRESHOLD = timedelta(minutes=30)

# Redelivery of an acknowledged event comes back within a few minutes.
REDELIVERY_LAG = timedelta(minutes=3)

SOURCE_BY_RAIL = {
    Rail.CARD: "pgw",
    Rail.ENACH: "npci_nach",
    Rail.UPI_AUTOPAY: "upi_autopay",
    Rail.INVOICE: "billing",
}


@dataclass(frozen=True, slots=True)
class WireEvent:
    """One HTTP delivery, as the adapter receives it.

    `body` is bytes, not a dict, because the signature covers the exact bytes
    and an adapter that re-serialises before verifying has a bug the fake
    should expose rather than hide.
    """

    source: str
    event_id: str
    event_timestamp: datetime
    received_at: datetime
    signature: str
    body: bytes
    delivery: int  # 1 on first delivery, 2 on a redelivery

    def payload(self) -> dict:
        """Convenience for tests. The adapter parses the bytes itself."""
        return json.loads(self.body.decode("utf-8"))


def signature_payload(timestamp: datetime, body: bytes) -> bytes:
    """Exactly what the HMAC covers: the timestamp, a dot, then the raw body.

    Including the timestamp is what stops a captured payload being replayed
    against the endpoint a week later.
    """
    return f"{int(timestamp.timestamp())}.".encode() + body


def sign(secret: bytes, timestamp: datetime, body: bytes) -> str:
    digest = hmac.new(secret, signature_payload(timestamp, body), hashlib.sha256).hexdigest()
    return f"t={int(timestamp.timestamp())},v1={digest}"


def verify(secret: bytes, event: WireEvent) -> bool:
    """Constant-time verification, offered so tests check the real thing.

    The adapter at M5 implements its own; this exists so the fake can be
    proven to sign correctly without trusting the code under test.
    """
    expected = sign(secret, event.event_timestamp, event.body)
    return hmac.compare_digest(expected, event.signature)


def _amount_string(amount_paise: int) -> str:
    """Rupees with two decimals, built from integer paise by division and
    remainder. Never through a float - GI-2 holds on the wire too."""
    rupees, paise_part = divmod(amount_paise, 100)
    return f"{rupees}.{paise_part:02d}"


def _canonical(payload: dict) -> bytes:
    """Deterministic JSON. Key order is fixed so the same event produces the
    same bytes, which is what makes a redelivery byte-identical."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


class WireFake:
    """Emits gateway-shaped webhooks for a world's batch.

    L0 should not be able to tell these from production traffic: same envelope,
    same signature scheme, same three dialects, same delivery pathologies.
    """

    def __init__(self, world: World, secret: bytes) -> None:
        if not isinstance(secret, bytes) or len(secret) < 16:
            raise ValueError("secret must be at least 16 bytes of key material")
        self._world = world
        self._secret = secret

    def emit(
        self,
        mode: Mode = "replay",
        seed: int | None = None,
        *,
        now: datetime | None = None,
    ) -> tuple[WireEvent, ...]:
        """Deliver the batch.

        `replay` is anchored to the world's own event times and is byte-
        identical for a given seed. `live` re-anchors the whole batch to
        `now` and jitters delivery around it; it requires `now` to be passed
        in, because reading a clock here would put a second clock in a repo
        that has exactly one.
        """
        if mode not in ("replay", "live"):
            raise ValueError(f"unknown mode {mode!r}; expected 'replay' or 'live'")
        if mode == "live" and now is None:
            raise ValueError("live mode requires `now` - the simulator never reads a clock")

        effective_seed = self._world.seed if seed is None else seed
        generator = rng(effective_seed, Stream.WIRE)

        batch = self._world.batch_events()
        offset = timedelta(0)
        if mode == "live" and now is not None:
            offset = now - batch[-1].at if batch else timedelta(0)

        late = self._select(generator, len(batch), OUT_OF_ORDER_RATE)
        duplicated = self._select(generator, len(batch), DUPLICATE_RATE)

        deliveries: list[WireEvent] = []
        for index, event in enumerate(batch):
            occurred = event.at + offset
            lag = LATE_DELIVERY_LAG if index in late else DELIVERY_LAG
            if mode == "live":
                lag += timedelta(seconds=int(generator.integers(0, DELIVERY_JITTER_SECONDS)))
            first = self._deliver(event, occurred, occurred + lag, delivery=1)
            deliveries.append(first)
            if index in duplicated:
                deliveries.append(
                    WireEvent(
                        source=first.source,
                        event_id=first.event_id,
                        event_timestamp=first.event_timestamp,
                        received_at=first.received_at + REDELIVERY_LAG,
                        signature=first.signature,
                        body=first.body,
                        delivery=2,
                    )
                )

        deliveries.sort(key=lambda d: (d.received_at, d.event_id, d.delivery))
        return tuple(deliveries)

    @staticmethod
    def _select(generator: np.random.Generator, total: int, rate: float) -> frozenset[int]:
        """Exactly `round(rate * total)` indices, not a per-event coin flip.

        An exact count means the injected pathology rate is a property of the
        batch rather than something that happened to come out near it, so a
        test can assert on it without a tolerance band hiding a broken fake.
        """
        count = int(round(rate * total))
        if count <= 0 or total == 0:
            return frozenset()
        return frozenset(int(i) for i in generator.choice(total, size=count, replace=False))

    def _deliver(
        self, event: BatchEvent, occurred: datetime, arrived: datetime, *, delivery: int
    ) -> WireEvent:
        body = _canonical(self._body(event, occurred))
        return WireEvent(
            source=SOURCE_BY_RAIL[event.rail],
            event_id=event.event_id,
            event_timestamp=occurred,
            received_at=arrived,
            signature=sign(self._secret, occurred, body),
            body=body,
            delivery=delivery,
        )

    # -- the three dialects ------------------------------------------------
    #
    # The same underlying fact - a debit did not settle - looks completely
    # different on each rail. That is the whole reason L1 exists, and the
    # reason the adapters are forbidden from deciding anything: the shape of
    # the payload is a gateway concern and must not reach policy.

    def _body(self, event: BatchEvent, occurred: datetime) -> dict:
        account = self._world._account(event.account_id)
        if event.kind is EventKind.INVOICE_OVERDUE:
            return self._invoice_body(event, account, occurred)
        if event.kind is EventKind.CHECKOUT_ABANDON:
            return self._checkout_body(event, account, occurred)
        if event.rail is Rail.ENACH:
            return self._nach_body(event, account, occurred)
        if event.rail is Rail.UPI_AUTOPAY:
            return self._upi_body(event, account, occurred)
        return self._card_body(event, account, occurred)

    def _card_body(self, event: BatchEvent, account, occurred: datetime) -> dict:
        digits = stable_hash(account.account_id, "card")
        return {
            "entity": "event",
            "event": "payment.captured" if event.succeeded else "payment.failed",
            "created_at": int(occurred.timestamp()),
            "payload": {
                "payment": {
                    "entity": "payment",
                    "id": f"pay_{stable_hash(event.event_id):014x}",
                    "amount": int(event.amount_paise),
                    "currency": "INR",
                    "status": "captured" if event.succeeded else "failed",
                    "method": "card",
                    "card": {
                        "last4": f"{digits % 10000:04d}",
                        "network": ("Visa", "MasterCard", "RuPay")[digits % 3],
                        "issuer": account.issuer_id,
                    },
                    "error_code": None if event.succeeded else "BAD_REQUEST_ERROR",
                    "error_reason": event.decline_code,
                    "error_description": code_book.describe(event.rail, event.decline_code),
                    "merchant_advice_code": event.advice_code,
                    "customer": {
                        "id": account.customer_ref,
                        "name": account.person.name,
                        "email": account.person.email,
                        "contact": account.person.phone,
                    },
                    "notes": {
                        "account_id": account.account_id,
                        "attempt": event.attempt,
                        "initiated_by": str(event.initiated_by),
                    },
                }
            },
        }

    def _nach_body(self, event: BatchEvent, account, occurred: datetime) -> dict:
        digits = stable_hash(account.account_id, "nach")
        settlement = occurred.date().isoformat()
        # The narration is free text with a real name in it. It flows
        # downstream with the claim, which is why the redaction boundary is at
        # L1 and not only in front of the LLM.
        narration = (
            f"ACH DR RTN CHG {occurred.strftime('%d%m%y')} "
            f"{account.person.name.upper()} UMRN {digits % 10**12:012d}"
        )
        return {
            "message_type": "ACH_DR_SETTLED" if event.succeeded else "ACH_DR_RETURN",
            "umrn": f"HDFC{digits % 10**12:012d}",
            "transaction_id": f"NACH{stable_hash(event.event_id) % 10**14:014d}",
            "settlement_date": settlement,
            "amount": _amount_string(int(event.amount_paise)),
            "return_code": event.decline_code,
            "return_reason": code_book.describe(event.rail, event.decline_code),
            "destination_bank": account.issuer_id,
            "customer": {
                "name": account.person.name,
                "account_last4": f"{digits % 10000:04d}",
                # IFSC shape: four letters, a zero, then six characters.
                "ifsc": f"{_bank_letters(account.issuer_id)}0{digits % 10**6:06d}",
                "mobile": account.person.phone,
            },
            "narration": narration,
            "reference": {
                "account_id": account.account_id,
                "attempt": event.attempt,
                "initiated_by": str(event.initiated_by),
            },
        }

    def _upi_body(self, event: BatchEvent, account, occurred: datetime) -> dict:
        handle = account.person.email.split("@")[0].replace(".", "")
        return {
            "event": "autopay.debit.success" if event.succeeded else "autopay.debit.failed",
            "mandate_id": f"UPIM{stable_hash(account.account_id, 'upi') % 10**12:012d}",
            "txn_id": f"UPI{stable_hash(event.event_id) % 10**15:015d}",
            "timestamp": occurred.isoformat(),
            "amount_paise": int(event.amount_paise),
            "npci_error_code": event.decline_code,
            "npci_error_desc": code_book.describe(event.rail, event.decline_code),
            "issuer": account.issuer_id,
            "payer": {
                "vpa": f"{handle}@examplebank",
                "name": account.person.name,
                "mobile": account.person.phone,
            },
            "ref": {
                "account_id": account.account_id,
                "attempt": event.attempt,
                "initiated_by": str(event.initiated_by),
            },
        }

    def _invoice_body(self, event: BatchEvent, account, occurred: datetime) -> dict:
        digits = stable_hash(account.account_id, "gstin")
        due = occurred - timedelta(days=_AGEING_DAYS[account.invoice_ageing_bucket])
        return {
            "type": "invoice.overdue",
            "created_at": int(occurred.timestamp()),
            "invoice": {
                "id": f"inv_{stable_hash(event.event_id):012x}",
                "amount_due": _amount_string(int(event.amount_paise)),
                "currency": "INR",
                "due_date": due.date().isoformat(),
                "days_overdue": (occurred.date() - due.date()).days,
                "ageing_bucket": account.invoice_ageing_bucket,
                "buyer": {
                    "name": account.person.name,
                    "email": account.person.email,
                    "gstin": f"{digits % 100:02d}AAACX{digits % 10000:04d}A1Z5",
                    "contact": account.person.phone,
                },
            },
            "ref": {"account_id": account.account_id, "attempt": event.attempt},
        }

    def _checkout_body(self, event: BatchEvent, account, occurred: datetime) -> dict:
        return {
            "entity": "event",
            "event": "checkout.abandoned",
            "created_at": int(occurred.timestamp()),
            "payload": {
                "order": {
                    "id": f"order_{stable_hash(event.event_id):012x}",
                    "amount": int(event.amount_paise),
                    "currency": "INR",
                    "status": "created",
                    "attempts": event.attempt,
                    "customer": {
                        "id": account.customer_ref,
                        "name": account.person.name,
                        "email": account.person.email,
                        "contact": account.person.phone,
                    },
                    "notes": {"account_id": account.account_id},
                }
            },
        }


def _bank_letters(issuer_id: str) -> str:
    """Four letters from an issuer id, so the IFSC matches the real shape and
    the PII detector at M2 recognises it as one."""
    letters = "".join(character for character in issuer_id if character.isalpha()).upper()
    return (letters + "XXXX")[:4]


_AGEING_DAYS = {
    "current": 3,
    "1_30": 18,
    "31_60": 45,
    "61_90": 75,
    "90_plus": 120,
}


def late_deliveries(events: tuple[WireEvent, ...]) -> tuple[WireEvent, ...]:
    """Deliveries that arrived long after the event they describe.

    Computed from the payload the way an adapter would - arrival minus the
    event timestamp - rather than from a flag the fake set. A fake that
    labelled its own noise would prove nothing about the adapter.
    """
    return tuple(
        event for event in events if event.received_at - event.event_timestamp > LATE_THRESHOLD
    )


def arrival_inversions(events: tuple[WireEvent, ...]) -> int:
    """Deliveries that arrived after something which happened later than them.

    This is the property that makes `order by event_timestamp, not arrival`
    load-bearing at M5: process in arrival order and a captured payment can
    overwrite the failure it supersedes.
    """
    inversions = 0
    highest: datetime | None = None
    for event in events:
        if highest is not None and event.event_timestamp < highest:
            inversions += 1
        if highest is None or event.event_timestamp > highest:
            highest = event.event_timestamp
    return inversions
