"""M5 acceptance gate: L0 adapters and the L1 redaction boundary.

The eight named tests are:

    test_unsigned_webhook_rejected_before_parse
    test_duplicate_event_id_deduped
    test_out_of_order_resolved_by_event_time
    test_bank_narration_does_not_reach_claim_evidence_structured
    test_arm_assigned_at_subject_not_claim
    test_all_claims_of_subject_share_arm
    test_arm_assignment_deterministic
    test_strata_balanced_within_tolerance

`test_all_claims_of_subject_share_arm` is the milestone. Claim-level
randomisation violates SUTVA under a shared contact budget, and it does so
silently: every number M11 reports would still have a confidence interval, and
the interval would be around the wrong quantity.

Every database test runs inside a transaction that is rolled back, so the
global ledger sequence is not shared between them.
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import asyncpg
import pytest
from arc.core.ids import subject_token
from arc.core.money import paise
from arc.core.types import ClaimState, ClaimType, Rail
from arc.ingest.adapters import build_registry, expected_signature
from arc.ingest.adapters.base import (
    paise_from_rupee_string,
    verify_signature,
)
from arc.ingest.adapters.nach import NachAdapter
from arc.ingest.archive import RawArchive
from arc.ingest.breaker import BreakerState, SourceBreakers, SourceTripped
from arc.ingest.dedupe import Dedupe
from arc.ingest.events import MalformedPayload, PersonalData, RawEvent, WireKind
from arc.ingest.normaliser import (
    ClaimContext,
    MissingValueEstimate,
    Normaliser,
    UnresolvableIdentity,
    pseudonymous_mandate_ref,
)
from arc.ingest.ordering import count_out_of_order, fold_by_account, sort_by_event_time
from arc.ingest.pipeline import Delivery, IngestPipeline
from arc.ledger.decision_ledger import DecisionLedger
from arc.ledger.pii_guard import PIIDetected, PIIGuard
from arc.ledger.subject_store import SubjectStore
from arc.proving_ground.arms import (
    ARMS,
    Arm,
    ArmRegistry,
    Strata,
    assign_arm,
    claim_count_bucket,
    decile_cutoffs,
    value_decile,
)
from arc.simulator.seeds import DEVELOP_SEED
from arc.simulator.wire_fake import DUPLICATE_RATE, WireFake
from arc.simulator.world import World

DSN = os.environ.get("DATABASE_URL", "postgresql://arc:arc@localhost:5432/arc")

SECRET = b"m5-acceptance-gate-secret-000000"
PEPPER = b"m5-acceptance-gate-pepper-000000"
EXPERIMENT = "m5-acceptance"
SOURCES = ("pgw", "npci_nach", "upi_autopay", "billing")

T0 = datetime(2025, 11, 3, 9, 0, tzinfo=UTC)

# The narration the boundary exists to stop. It carries a name and an account
# number, and at 49 characters it passes M1's shape cap comfortably - which is
# the point: the cap is not the defence.
NARRATION = "ACH DR RTN CHG 201025 DIVYA KHAN UMRN 091019459048"


def deterministic_keys() -> Callable[[int], bytes]:
    counter = itertools.count(1)
    return lambda n: hashlib.sha256(f"m5-key-{next(counter)}".encode()).digest()[:n]


@pytest.fixture
async def conn() -> AsyncIterator[asyncpg.Connection]:
    try:
        connection = await asyncpg.connect(DSN)
    except (OSError, asyncpg.PostgresError) as exc:  # pragma: no cover
        pytest.fail(f"Postgres is not reachable at {DSN}; run `make up && make migrate`: {exc}")

    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


def build_pipeline(*, ltv_multiple: int = 6) -> IngestPipeline:
    ledger = DecisionLedger()
    return IngestPipeline(
        adapters=build_registry(dict.fromkeys(SOURCES, SECRET)),
        normaliser=Normaliser(
            pepper=PEPPER,
            ltv_source=lambda event: paise(int(event.amount_paise) * ltv_multiple),
        ),
        arms=ArmRegistry(EXPERIMENT),
        subject_store=SubjectStore(ledger, key_source=deterministic_keys()),
        ledger=ledger,
        archive=RawArchive(),
        dedupe=Dedupe(),
        breakers=SourceBreakers(),
    )


@pytest.fixture
def pipeline() -> IngestPipeline:
    return build_pipeline()


# ---------------------------------------------------------------------------
# Payload builders. Real gateway shapes, minted here so a test can control the
# customer, the account and the event time exactly.
# ---------------------------------------------------------------------------
def sign(body: bytes, at: datetime) -> str:
    return expected_signature(SECRET, int(at.timestamp()), body)


def canonical(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def pgw_body(
    *,
    event_id: str,
    at: datetime,
    amount_paise: int,
    customer_ref: str,
    account_ref: str,
    captured: bool = False,
    name: str = "Priya Sharma",
    attempt: int = 1,
) -> bytes:
    return canonical(
        {
            "entity": "event",
            "event": "payment.captured" if captured else "payment.failed",
            "created_at": int(at.timestamp()),
            "payload": {
                "payment": {
                    "entity": "payment",
                    "id": event_id,
                    "amount": amount_paise,
                    "currency": "INR",
                    "status": "captured" if captured else "failed",
                    "method": "card",
                    "card": {"last4": "4242", "network": "Visa", "issuer": "ISS_LP01"},
                    "error_code": None if captured else "BAD_REQUEST_ERROR",
                    "error_reason": None if captured else "51",
                    "error_description": None if captured else "Insufficient funds",
                    "merchant_advice_code": None,
                    "customer": {
                        "id": customer_ref,
                        "name": name,
                        "email": "priya.sharma@example.com",
                        "contact": "+919876543210",
                    },
                    "notes": {
                        "account_id": account_ref,
                        "attempt": attempt,
                        "initiated_by": "merchant",
                    },
                }
            },
        }
    )


def nach_body(*, event_id: str, on: str, amount: str, account_ref: str) -> bytes:
    return canonical(
        {
            "message_type": "ACH_DR_RETURN",
            "umrn": "HDFC091019459048",
            "transaction_id": event_id,
            "settlement_date": on,
            "amount": amount,
            "return_code": "AM04",
            "return_reason": "Insufficient funds",
            "destination_bank": "ISS_CO01",
            "customer": {
                "name": "Divya Khan",
                "account_last4": "9048",
                "ifsc": "ISSC0459048",
                "mobile": "+916127375241",
            },
            "narration": NARRATION,
            "reference": {"account_id": account_ref, "attempt": 1, "initiated_by": "merchant"},
        }
    )


def delivery(
    source: str,
    body: bytes,
    at: datetime,
    *,
    received_at: datetime | None = None,
    signature: str | None = None,
) -> Delivery:
    return Delivery(
        source=source,
        raw=body,
        signature=sign(body, at) if signature is None else signature,
        received_at=received_at or at,
    )


def card_delivery(
    *,
    event_id: str,
    at: datetime,
    amount_paise: int,
    customer_ref: str,
    account_ref: str,
    captured: bool = False,
    received_at: datetime | None = None,
) -> Delivery:
    body = pgw_body(
        event_id=event_id,
        at=at,
        amount_paise=amount_paise,
        customer_ref=customer_ref,
        account_ref=account_ref,
        captured=captured,
    )
    return delivery("pgw", body, at, received_at=received_at)


def subject_deliveries(customer_ref: str, claims: int, *, base: int = 100_000) -> list[Delivery]:
    """One subject, `claims` failed card debits on distinct accounts."""
    return [
        card_delivery(
            event_id=f"pay_{customer_ref}_{index}",
            at=T0 + timedelta(minutes=index),
            amount_paise=base + index * 1_000,
            customer_ref=customer_ref,
            account_ref=f"acct_{customer_ref}_{index}",
        )
        for index in range(claims)
    ]


# ---------------------------------------------------------------------------
# Gate test 1 - verify before parse
# ---------------------------------------------------------------------------
async def test_unsigned_webhook_rejected_before_parse(conn, pipeline) -> None:
    """An unverified delivery is never handed to a parser.

    Not merely "not turned into a claim". The bytes must not reach
    deserialisation at all, because a parser is the first thing an attacker
    gets to run. Proven with a spy on the adapter: `parse` is not called.
    """
    body = pgw_body(
        event_id="pay_unsigned",
        at=T0,
        amount_paise=129900,
        customer_ref="cust_1",
        account_ref="acct_1",
    )

    parsed: list[bytes] = []
    adapter = pipeline._adapters["pgw"]
    real_parse = adapter.parse

    def spy(raw: bytes):
        parsed.append(raw)
        return real_parse(raw)

    adapter.parse = spy  # type: ignore[method-assign]

    report = await pipeline.ingest(
        conn,
        [
            Delivery(source="pgw", raw=body, signature="t=1,v1=" + "0" * 64, received_at=T0),
            Delivery(source="pgw", raw=body, signature="not-a-signature", received_at=T0),
            Delivery(source="pgw", raw=body, signature="", received_at=T0),
        ],
        T0,
    )

    assert parsed == [], "an unverified payload reached the parser"
    assert report.rejected_signature == 3
    assert report.claims_created == 0

    # Archived anyway, and marked as having failed verification. Discarding it
    # would throw away the evidence of the attempt.
    rows = await conn.fetch(
        "SELECT signature_valid, parse_error FROM raw_events WHERE source = 'pgw'"
    )
    assert len(rows) == 3
    assert all(row["signature_valid"] is False for row in rows)
    assert all(row["parse_error"] == "signature invalid" for row in rows)


async def test_a_tampered_body_does_not_verify(pipeline) -> None:
    """The signature covers the exact bytes, so changing the amount breaks it."""
    body = pgw_body(
        event_id="pay_tamper",
        at=T0,
        amount_paise=129900,
        customer_ref="cust_1",
        account_ref="acct_1",
    )
    signature = sign(body, T0)
    adapter = pipeline._adapters["pgw"]

    assert adapter.verify(body, signature)
    assert not adapter.verify(body.replace(b"129900", b"999900"), signature)
    assert not adapter.verify(body, signature.replace("t=", "t=1"))
    assert not verify_signature(b"a-different-secret-key-0123456789", body, signature)


async def test_unknown_source_is_refused_not_guessed(conn, pipeline) -> None:
    """GI-5: a webhook from an unrecognised sender fails closed."""
    body = pgw_body(event_id="pay_x", at=T0, amount_paise=1000, customer_ref="c", account_ref="a")
    report = await pipeline.ingest(
        conn, [Delivery(source="mystery", raw=body, signature=sign(body, T0), received_at=T0)], T0
    )
    assert report.unknown_source == 1
    assert report.claims_created == 0
    assert await conn.fetchval("SELECT count(*) FROM raw_events WHERE source = 'mystery'") == 1


async def test_stale_signature_refused(conn, pipeline) -> None:
    """A captured payload replayed days later has a valid signature and is
    still refused, because the timestamp it signed is inside the material."""
    body = pgw_body(event_id="pay_old", at=T0, amount_paise=1000, customer_ref="c", account_ref="a")
    report = await pipeline.ingest(
        conn,
        [delivery("pgw", body, T0, received_at=T0 + timedelta(days=5))],
        T0 + timedelta(days=5),
    )
    assert report.stale_signature == 1
    assert report.claims_created == 0


async def test_archive_is_written_before_the_parse(conn, pipeline) -> None:
    """A payload that verifies but does not parse is still archived.

    That is the case the ordering exists for: the deliveries most worth
    replaying through a fixed parser are exactly the ones a parse-then-archive
    order would never keep.
    """
    broken = canonical({"entity": "event", "event": "payment.failed", "created_at": 1})
    report = await pipeline.ingest(conn, [delivery("pgw", broken, T0)], T0)

    assert report.parse_failures == 1
    row = await conn.fetchrow("SELECT body, parse_error, signature_valid FROM raw_events")
    assert row["signature_valid"] is True
    assert row["parse_error"] is not None
    assert bytes(row["body"]) == broken, "the archive must hold the original bytes"


# ---------------------------------------------------------------------------
# Gate tests 2 and 3 - dedupe and ordering
# ---------------------------------------------------------------------------
async def test_duplicate_event_id_deduped(conn, pipeline) -> None:
    """Redelivery produces one claim, not two.

    Gateways redeliver on acknowledgement timeout. Without this, one failure
    becomes two claims, is diagnosed twice, and spends the contact budget twice
    on one person.
    """
    signed = card_delivery(
        event_id="pay_dupe",
        at=T0,
        amount_paise=129900,
        customer_ref="cust_dupe",
        account_ref="acct_dupe",
    )
    redelivery = replace(signed, received_at=T0 + timedelta(minutes=3))

    report = await pipeline.ingest(conn, [signed, redelivery, redelivery], T0)

    assert report.delivered == 3
    assert report.deduplicated == 2
    assert report.claims_created == 1
    assert await conn.fetchval("SELECT count(*) FROM claims") == 1

    # Three deliveries, three archive rows, one dedupe row. The archive counts
    # deliveries and the dedupe table counts events, and keeping them apart is
    # what makes redelivery rate measurable at all.
    assert await conn.fetchval("SELECT count(*) FROM raw_events") == 3
    assert await conn.fetchval("SELECT count(*) FROM ingest_dedupe") == 1


async def test_dedupe_window_expires_after_thirty_days(conn) -> None:
    """Beyond the window the record ages out and the event counts as new.

    Deliberate: unbounded dedupe state grows forever, and a redelivery a month
    later is a different operational event rather than a duplicate.
    """
    dedupe = Dedupe()
    assert (await dedupe.claim(conn, "pgw", "evt_1", T0)).is_new
    assert not (await dedupe.claim(conn, "pgw", "evt_1", T0 + timedelta(days=29))).is_new
    assert (await dedupe.claim(conn, "pgw", "evt_1", T0 + timedelta(days=31))).is_new

    assert await dedupe.purge(conn, T0 + timedelta(days=90)) == 1
    assert await conn.fetchval("SELECT count(*) FROM ingest_dedupe") == 0


async def test_dedupe_is_per_source(conn) -> None:
    """Two gateways may use the same id space without colliding."""
    dedupe = Dedupe()
    assert (await dedupe.claim(conn, "pgw", "shared_id", T0)).is_new
    assert (await dedupe.claim(conn, "npci_nach", "shared_id", T0)).is_new


async def test_out_of_order_resolved_by_event_time(conn, pipeline) -> None:
    """A capture that arrives first but happened later still supersedes.

    Processed by arrival, the failure would create a claim for money already
    collected, and that claim would go on to be diagnosed, funded and messaged
    to somebody who has paid. It is the worst customer experience the system
    can produce, and it is one sort order away.
    """
    failure = card_delivery(
        event_id="pay_late_failure",
        at=T0,
        amount_paise=250000,
        customer_ref="cust_race",
        account_ref="acct_race",
        received_at=T0 + timedelta(hours=9),
    )
    capture = card_delivery(
        event_id="pay_early_capture",
        at=T0 + timedelta(minutes=30),
        amount_paise=250000,
        customer_ref="cust_race",
        account_ref="acct_race",
        captured=True,
        received_at=T0 + timedelta(minutes=31),
    )

    # Arrival order: the capture first, though it happened second.
    report = await pipeline.ingest(conn, [capture, failure], T0 + timedelta(hours=10))

    assert report.out_of_order_arrivals == 1
    assert report.superseded_by_capture == 1
    assert report.claims_created == 0
    assert await conn.fetchval("SELECT count(*) FROM claims") == 0


async def test_a_lone_failure_still_becomes_a_claim(conn, pipeline) -> None:
    """The control for the test above: without the capture, there is a claim."""
    failure = card_delivery(
        event_id="pay_lonely",
        at=T0,
        amount_paise=250000,
        customer_ref="cust_lonely",
        account_ref="acct_lonely",
    )
    report = await pipeline.ingest(conn, [failure], T0 + timedelta(hours=1))
    assert report.claims_created == 1
    assert report.superseded_by_capture == 0


async def test_repeat_presentations_are_one_claim_with_two_attempts(conn, pipeline) -> None:
    """An obligation is a claim; a presentation is an attempt at it.

    The gateway re-presents on its own schedule. Counting those as separate
    claims would double the money at risk and double the contact budget spent
    recovering it.
    """
    first = card_delivery(
        event_id="pay_attempt_1",
        at=T0,
        amount_paise=129900,
        customer_ref="cust_two",
        account_ref="acct_two",
    )
    second = card_delivery(
        event_id="pay_attempt_2",
        at=T0 + timedelta(days=1),
        amount_paise=129900,
        customer_ref="cust_two",
        account_ref="acct_two",
    )
    report = await pipeline.ingest(conn, [first, second], T0 + timedelta(days=2))

    assert report.claims_created == 1
    claim = report.claims[0]
    assert claim.evidence_structured["failed_attempts"] == 2
    assert claim.detected_at == T0 + timedelta(days=1), "the latest event dates the claim"


def test_sorting_is_deterministic_on_a_tie() -> None:
    """Two events at the same instant sort the same way on every replay."""

    def event(event_id: str) -> RawEvent:
        return RawEvent(
            source="pgw",
            event_id=event_id,
            event_timestamp=T0,
            kind=WireKind.DEBIT_FAILED,
            rail=Rail.CARD,
            account_ref="acct_tie",
            customer_ref="cust_tie",
            amount_paise=paise(1000),
            raw=b"{}",
            raw_hash=hashlib.sha256(b"{}").digest(),
        )

    events = [event("b"), event("a"), event("c")]
    assert [e.event_id for e in sort_by_event_time(events)] == ["a", "b", "c"]
    assert [e.event_id for e in sort_by_event_time(list(reversed(events)))] == ["a", "b", "c"]


def test_out_of_order_counter_measures_arrival_against_event_time() -> None:
    ordered = sort_by_event_time(
        [
            RawEvent(
                source="pgw",
                event_id=f"e{index}",
                event_timestamp=T0 + timedelta(minutes=index),
                kind=WireKind.DEBIT_FAILED,
                rail=Rail.CARD,
                account_ref=f"a{index}",
                customer_ref="c",
                amount_paise=paise(1),
                raw=b"{}",
                raw_hash=hashlib.sha256(b"{}").digest(),
            )
            for index in range(5)
        ]
    )
    assert count_out_of_order(ordered) == 0
    assert count_out_of_order(list(reversed(ordered))) == 4


# ---------------------------------------------------------------------------
# Gate test 4 - the redaction boundary
# ---------------------------------------------------------------------------
def normaliser() -> Normaliser:
    return Normaliser(pepper=PEPPER, ltv_source=lambda event: paise(int(event.amount_paise) * 6))


def nach_event() -> RawEvent:
    body = nach_body(
        event_id="NACH59593970044975",
        on="2025-10-20",
        amount="842.31",
        account_ref="acct_0000059",
    )
    return NachAdapter(SECRET).parse(body)


def test_bank_narration_does_not_reach_claim_evidence_structured() -> None:
    """The narration goes to the subject store. It does not go to the claim.

    Free text is not a formatting problem. Once a name is hash-chained into an
    append-only ledger, erasure is impossible without breaking the chain, and
    the chain is the whole reason the audit trail is worth having.
    """
    event = nach_event()
    assert event.personal.narration == NARRATION

    claim, record = normaliser().normalise(event)

    rendered = json.dumps(dict(claim.evidence_structured), default=str)
    for leak in (
        "DIVYA",
        "KHAN",
        "Divya",
        "091019459048",
        "HDFC091019459048",
        "ISSC0459048",
        "+916127375241",
        "9048",
        NARRATION,
    ):
        assert leak not in rendered, f"{leak!r} crossed the boundary into the claim"

    # And it really is on the other side, not merely dropped. A boundary that
    # discarded the narration would break diagnosis at M6 instead of protecting
    # anybody.
    assert record.payload["narration"] == NARRATION
    assert record.payload["name"] == "Divya Khan"
    assert record.payload["identifiers"]["mandate"] == "HDFC091019459048"

    # What crosses is a pointer, a digest, and a one-way mandate pseudonym.
    assert claim.evidence_hash == hashlib.sha256(event.raw).digest()
    assert claim.evidence_structured["mandate_ref"] == pseudonymous_mandate_ref("HDFC091019459048")
    assert claim.subject_token.startswith("sub_")


def test_the_claim_survives_the_pii_guard_and_the_record_would_not() -> None:
    """The two halves are on opposite sides of the guard, by construction.

    The claim passes. The subject record, offered to the same guard, is
    refused - which is the proof that the split is real and not cosmetic.
    """
    claim, record = normaliser().normalise(nach_event())
    guard = PIIGuard()

    assert guard.scan(dict(claim.evidence_structured)) == []

    with pytest.raises(PIIDetected):
        guard.scan(dict(record.payload))


def test_normaliser_refuses_to_build_a_claim_carrying_free_text() -> None:
    """Defence in depth: the boundary itself fails the write.

    The ledger guard would catch it later, but later means after the claim has
    been passed around, and a claim built wrong is one somebody eventually
    writes.
    """
    event = nach_event()
    smuggled = ClaimContext(extra_evidence={"note": "Contact RAJESH KUMAR on 9876543210"})
    with pytest.raises(PIIDetected):
        normaliser().normalise(event, smuggled)


async def test_narration_reaches_the_subject_store_and_never_the_claim(conn, pipeline) -> None:
    """The full path, on a payload the frozen wire fake actually produces.

    End to end: signed bytes in, subject store and claims table out. The
    narration must be readable through the subject store and absent from every
    ledgerable surface - the claim object, the claims row, and the hash-chained
    ledger entry.
    """
    body = nach_body(
        event_id="NACH_FULLPATH_001",
        on="2025-11-03",
        amount="1299.00",
        account_ref="acct_fullpath",
    )
    report = await pipeline.ingest(conn, [delivery("npci_nach", body, T0)], T0)
    assert report.claims_created == 1

    claim = report.claims[0]
    assert claim.evidence_ref is not None

    # Readable on the erasable side.
    store = SubjectStore(DecisionLedger(), key_source=deterministic_keys())
    stored = await store.get(conn, claim.evidence_ref)
    assert stored["narration"] == NARRATION
    assert stored["name"] == "Divya Khan"

    # Absent from every ledgerable surface.
    row = await conn.fetchrow(
        "SELECT evidence_structured::text AS evidence, evidence_ref FROM claims "
        "WHERE claim_id = $1",
        claim.claim_id,
    )
    assert "DIVYA" not in row["evidence"] and "Divya" not in row["evidence"]
    assert row["evidence_ref"] == claim.evidence_ref

    entries = await conn.fetch(
        "SELECT body_canonical FROM decision_ledger WHERE claim_id = $1", claim.claim_id
    )
    assert entries
    for entry in entries:
        text = entry["body_canonical"]
        assert "DIVYA" not in text.upper()
        assert "091019459048" not in text
        assert "916127375241" not in text

    # The whole raw payload is on the erasable side too, so erasure has one
    # authoritative place to destroy.
    assert NARRATION in stored["raw_payload"]


async def test_erasure_can_sweep_the_raw_archive(conn, pipeline) -> None:
    """The archive holds untrusted bytes including personal data.

    Unlike the ledger it is deletable, and it carries the subject token
    precisely so an erasure request can find every delivery belonging to one
    person. Without that column the archive would be an unerasable copy of
    everything the subject store shreds.
    """
    body = nach_body(
        event_id="NACH_ERASE_001", on="2025-11-03", amount="500.00", account_ref="acct_erase"
    )
    report = await pipeline.ingest(conn, [delivery("npci_nach", body, T0)], T0)
    token = report.claims[0].subject_token

    assert (
        await conn.fetchval("SELECT count(*) FROM raw_events WHERE subject_token = $1", token) == 1
    )

    await conn.execute("DELETE FROM raw_events WHERE subject_token = $1", token)
    assert (
        await conn.fetchval("SELECT count(*) FROM raw_events WHERE subject_token = $1", token) == 0
    )


def test_identity_that_cannot_be_resolved_is_refused() -> None:
    """No customer id, no phone, no email - so no claim.

    A made-up token would be randomised into an arm of its own and quietly
    contaminate the subject-level design it was supposed to belong to.
    """
    anonymous = replace(nach_event(), customer_ref=None, personal=PersonalData())
    with pytest.raises(UnresolvableIdentity):
        normaliser().normalise(anonymous)


def test_a_claim_without_a_value_estimate_is_rejected() -> None:
    """M1's rule, enforced here: a claim nobody valued cannot be prioritised."""
    valueless = Normaliser(pepper=PEPPER, ltv_source=lambda event: None)
    with pytest.raises(MissingValueEstimate):
        valueless.normalise(nach_event())


# ---------------------------------------------------------------------------
# Gate tests 5 to 8 - subject-level randomisation (GI-8)
# ---------------------------------------------------------------------------
async def test_arm_assigned_at_subject_not_claim(conn, pipeline) -> None:
    """One row per subject, and nowhere to put a claim id.

    The schema is the cheapest place to make claim-level randomisation
    impossible rather than merely discouraged.
    """
    report = await pipeline.ingest(conn, subject_deliveries("cust_solo", 3), T0)
    assert report.claims_created == 3
    assert report.subjects == 1

    assert await conn.fetchval("SELECT count(*) FROM subject_arms") == 1

    columns = {
        row["column_name"]
        for row in await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'subject_arms'"
        )
    }
    assert "claim_id" not in columns
    assert "subject_token" in columns


async def test_all_claims_of_subject_share_arm(conn, pipeline) -> None:
    """THE MILESTONE TEST. Every claim inherits its subject's arm, no exceptions.

    Subjects holding one, two and five claims, ingested together. A subject
    with one claim in control and another in treatment breaks SUTVA twice: the
    portfolio allocation can starve the control claim of budget the treated
    claim consumed, and one message reminds them about every claim they hold.

    Neither effect is small and neither is visible afterwards. The estimate is
    simply wrong, with a confidence interval around the wrong quantity.
    """
    plan = {"cust_one": 1, "cust_two": 2, "cust_five": 5}
    deliveries = [
        item for customer, count in plan.items() for item in subject_deliveries(customer, count)
    ]
    report = await pipeline.ingest(conn, deliveries, T0)

    assert report.claims_created == sum(plan.values()) == 8
    assert report.subjects == 3

    # Asserted on what the PIPELINE stamped each claim with, not on what the
    # database holds. `subject_arms` is keyed by subject_token, so a database
    # read could never show one subject in two arms - the assertion would be
    # true by construction and would prove nothing about the assignment code.
    by_subject: dict[str, set[Arm]] = {}
    for claim in report.claims:
        stamped = report.claim_arms.get(claim.claim_id)
        assert stamped is not None, "a claim reached the batch with no arm assigned"
        by_subject.setdefault(claim.subject_token, set()).add(stamped)

    assert len(by_subject) == 3
    for token, arms in by_subject.items():
        assert len(arms) == 1, (
            f"{token} holds claims in {len(arms)} different arms "
            f"({', '.join(sorted(str(arm) for arm in arms))}) - randomisation is "
            "happening below the subject"
        )

    # And the arm the pipeline stamped is the one that was persisted.
    registry = ArmRegistry(EXPERIMENT)
    for claim in report.claims:
        assert report.claim_arms[claim.claim_id] is await registry.arm_of(conn, claim.subject_token)

    # And the claim counts really were 1, 2 and 5. A test where every subject
    # held one claim would pass without proving anything.
    counts = sorted(
        len([claim for claim in report.claims if claim.subject_token == token])
        for token in by_subject
    )
    assert counts == [1, 2, 5]

    # The stratum recorded is the one the subject was actually assigned in.
    buckets = {
        row["subject_token"]: row["claim_count_bucket"]
        for row in await conn.fetch("SELECT subject_token, claim_count_bucket FROM subject_arms")
    }
    assert sorted(buckets.values()) == ["1", "2_3", "4_plus"]


async def test_a_later_claim_does_not_move_a_subject_between_arms(conn, pipeline) -> None:
    """First assignment wins, even when the stratum would now be different.

    A subject's claim count grows as claims arrive. Recomputing the assignment
    would move them mid-experiment, and a subject who was control in week one
    and treated in week two is in neither arm.
    """
    first = await pipeline.ingest(conn, subject_deliveries("cust_growing", 1), T0)
    token = first.claims[0].subject_token
    original = await ArmRegistry(EXPERIMENT).arm_of(conn, token)

    later = await pipeline.ingest(
        conn,
        [
            card_delivery(
                event_id=f"pay_later_{index}",
                at=T0 + timedelta(days=1, minutes=index),
                amount_paise=900_000 + index,
                customer_ref="cust_growing",
                account_ref=f"acct_later_{index}",
            )
            for index in range(4)
        ],
        T0 + timedelta(days=1),
    )

    assert later.claims_created == 4
    assert all(claim.subject_token == token for claim in later.claims)
    assert await ArmRegistry(EXPERIMENT).arm_of(conn, token) is original

    # One row still, holding the stratum from the first assignment.
    row = await conn.fetchrow(
        "SELECT claim_count_bucket, assigned_at FROM subject_arms WHERE subject_token = $1",
        token,
    )
    assert row["claim_count_bucket"] == "1"
    assert row["assigned_at"] == T0


def test_arm_assignment_deterministic() -> None:
    """Same subject, same experiment, same stratum, same arm. Every time.

    Replay depends on it: a run that reassigned arms would produce a different
    treated set and therefore a different headline number.
    """
    token = subject_token("+919876543210", pepper=PEPPER)
    strata = Strata("2_3", 4, Rail.CARD)

    assert len({assign_arm(token, EXPERIMENT, strata) for _ in range(500)}) == 1

    # It really does depend on all three inputs.
    variants = {
        assign_arm(token, EXPERIMENT, strata),
        assign_arm(token, "other-experiment", strata),
        assign_arm(token, EXPERIMENT, Strata("1", 4, Rail.CARD)),
        assign_arm(subject_token("+919000000001", pepper=PEPPER), EXPERIMENT, strata),
    }
    assert len(variants) > 1, "assignment ignores one of its inputs"


def test_arm_assignment_refuses_a_raw_identifier() -> None:
    """A phone number must never be what gets randomised.

    It would mean the raw identifier travelled far enough to reach the
    experiment, which is exactly what the redaction boundary prevents.
    """
    with pytest.raises(ValueError, match="derived subject token"):
        assign_arm("+919876543210", EXPERIMENT, Strata("1", 0, Rail.CARD))
    with pytest.raises(ValueError):
        assign_arm(subject_token("x", pepper=PEPPER), "", Strata("1", 0, Rail.CARD))


def test_strata_balanced_within_tolerance() -> None:
    """Arms balance inside every stratum, not merely across the batch.

    Without stratification a stratum holding a few hundred subjects can land a
    third of them in one arm by chance, and every interval computed from it
    widens. The tolerance is the honest one for this sample size rather than a
    number picked to make the assertion pass.
    """
    strata = [
        Strata(bucket, decile, rail)
        for bucket in ("1", "2_3", "4_plus")
        for decile in (0, 5, 9)
        for rail in (Rail.CARD, Rail.ENACH)
    ]
    expected = 1.0 / len(ARMS)
    subjects = 3_000

    for stratum in strata:
        counts = dict.fromkeys(ARMS, 0)
        for index in range(subjects):
            counts[
                assign_arm(subject_token(f"+9198{index:08d}", pepper=PEPPER), EXPERIMENT, stratum)
            ] += 1

        for arm, count in counts.items():
            share = count / subjects
            assert abs(share - expected) < 0.03, (
                f"{stratum.key} put {share:.3f} of subjects in {arm}, expected {expected:.3f}"
            )


def test_claim_count_buckets_and_deciles() -> None:
    assert claim_count_bucket(1) == "1"
    assert claim_count_bucket(2) == claim_count_bucket(3) == "2_3"
    assert claim_count_bucket(4) == claim_count_bucket(50) == "4_plus"
    with pytest.raises(ValueError):
        claim_count_bucket(0)

    cutoffs = decile_cutoffs([paise(value) for value in range(1, 1001)])
    assert value_decile(paise(1), cutoffs) == 0
    assert value_decile(paise(1000), cutoffs) == 9
    assert value_decile(paise(500), cutoffs) in range(10)


def test_control_arm_is_named_and_singular() -> None:
    """M8 excludes control subjects from the allocation pool entirely.

    Leaving them in as merely untreated would let them consume or be starved
    by a shared budget, which is itself a treatment effect.
    """
    from arc.proving_ground.arms import CONTROL_ARMS

    assert set(CONTROL_ARMS) == {Arm.NULL}
    assert [str(arm) for arm in ARMS] == [
        "null",
        "naive_dunning",
        "gateway_default",
        "greedy_unconstrained",
        "arc",
    ]


# ---------------------------------------------------------------------------
# Adapters translate. They do not decide.
# ---------------------------------------------------------------------------
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "arc" / "ingest" / "adapters"

# Domain names an adapter must never branch on. Carrying an amount across is
# translation; changing behaviour because of one is policy, and policy in an
# effector cannot be audited or replayed.
FORBIDDEN_BRANCH_NAMES = frozenset(
    {
        "state",
        "claim_state",
        "ClaimState",
        "cause",
        "Cause",
        "CauseLayer",
        "CauseLabel",
        "claim_type",
        "ClaimType",
        "amount",
        "amount_paise",
        "ltv",
        "ltv_remaining_paise",
        "arm",
        "Arm",
        "confidence",
    }
)

FORBIDDEN_IMPORTS = frozenset(
    {"ClaimState", "ClaimType", "Cause", "CauseLabel", "CauseLayer", "Claim", "Arm"}
)


def _adapter_modules() -> list[Path]:
    return sorted(path for path in ADAPTER_DIR.glob("*.py") if path.name != "__init__.py")


def _names_in(node: ast.AST) -> set[str]:
    """Names, attributes and string literals anywhere in an expression.

    String literals count because `payload["amount_paise"] > 100000` reaches
    the same domain concept through a subscript, and a scan that only looked
    at identifiers would wave it through.
    """
    found: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.add(child.value)
    return found


def branch_violations(path: Path) -> list[str]:
    """Every branch in one adapter whose condition mentions a domain name."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences: list[str] = []

    for node in ast.walk(tree):
        tests: list[ast.AST] = []
        if isinstance(node, (ast.If, ast.While, ast.IfExp)):
            tests.append(node.test)
        elif isinstance(node, ast.Match):
            tests.append(node.subject)
        elif isinstance(node, ast.comprehension):
            tests.extend(node.ifs)

        for test in tests:
            leaked = _names_in(test) & FORBIDDEN_BRANCH_NAMES
            if leaked:
                offences.append(
                    f"{path.name}:{getattr(node, 'lineno', 0)} branches on "
                    f"{', '.join(sorted(leaked))}"
                )

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name in FORBIDDEN_IMPORTS:
                    offences.append(f"{path.name}:{node.lineno} imports {alias.name}")
    return offences


def test_adapters_contain_no_decision_logic() -> None:
    """An adapter translates a dialect. It does not decide anything.

    Gateway shape is a fact about a vendor and policy is a fact about the
    system. Once one leaks into the other, a compliance rule cannot be tested
    without a webhook fixture, and a vendor changing a field name becomes a
    policy change nobody reviewed.
    """
    modules = _adapter_modules()
    assert len(modules) >= 4, "the adapter scan found almost nothing; the walk is broken"

    offences = [offence for path in modules for offence in branch_violations(path)]
    assert not offences, "adapters must not decide:\n" + "\n".join(f"  {o}" for o in offences)


def test_the_adapter_scan_catches_a_planted_decision(tmp_path: Path) -> None:
    """The scan is worth nothing unless a real violation trips it."""
    planted = tmp_path / "sneaky.py"
    planted.write_text(
        "def parse(payload):\n"
        "    if payload['amount_paise'] > 100000:\n"
        "        return 'escalate'\n"
        "    return 'normal'\n",
        encoding="utf-8",
    )
    assert branch_violations(planted)

    ternary = tmp_path / "ternary.py"
    ternary.write_text(
        "def parse(claim):\n    return 'x' if claim.state == 'detected' else 'y'\n",
        encoding="utf-8",
    )
    assert branch_violations(ternary)


def test_the_adapter_scan_allows_translation(tmp_path: Path) -> None:
    """Reading a payload key and carrying a value across is not a decision."""
    honest = tmp_path / "honest.py"
    honest.write_text(
        "def parse(payload):\n"
        "    if 'error_reason' in payload:\n"
        "        code = payload['error_reason']\n"
        "    else:\n"
        "        code = None\n"
        "    return {'decline_code': code, 'value': payload['amount']}\n",
        encoding="utf-8",
    )
    assert branch_violations(honest) == []


def test_adapters_never_touch_the_database() -> None:
    """Verification and translation are pure. No I/O lives in a dialect."""
    for path in _adapter_modules():
        source = path.read_text(encoding="utf-8")
        for forbidden in ("asyncpg", "conn.", "await ", "SELECT ", "INSERT "):
            assert forbidden not in source, f"{path.name} performs I/O ({forbidden!r})"


# ---------------------------------------------------------------------------
# The adversarial traffic has to be exercised, not merely tolerated
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def adversarial_batch() -> list[Delivery]:
    """A real run of the frozen wire fake: signed, duplicated, late.

    Ingested through the production path with no fixtures in the way, so the
    2% redelivery and 3% late arrival are traffic rather than a description.
    """
    world = World(seed=DEVELOP_SEED, size=260)
    return [
        Delivery(
            source=wire.source,
            raw=wire.body,
            signature=wire.signature,
            received_at=wire.received_at,
        )
        for wire in WireFake(world, SECRET).emit("replay", DEVELOP_SEED)
    ]


async def test_dedupe_and_reordering_are_provably_invoked(conn, adversarial_batch) -> None:
    """Non-zero counts, so a change that dropped the adversarial behaviour fails.

    A build where the fake quietly stopped redelivering, or stopped delivering
    late, would still be green on every other test in this file. It would also
    mean the adapter's two hardest requirements were never being tried.
    """
    pipeline = build_pipeline()
    at = max(item.received_at for item in adversarial_batch) + timedelta(minutes=1)
    report = await pipeline.ingest(conn, adversarial_batch, at)

    assert report.delivered == len(adversarial_batch)
    assert report.rejected_signature == 0, "the fake's own signatures must verify"
    assert report.parse_failures == 0, "every dialect the fake speaks must parse"

    # Redelivery: exactly the rate the fake injects, and every duplicate caught.
    distinct = len({(item.source, item.raw) for item in adversarial_batch})
    assert report.deduplicated == len(adversarial_batch) - distinct
    assert report.deduplicated > 0, "no redelivery in the batch - dedupe was never exercised"
    assert report.deduplicated == pytest.approx(DUPLICATE_RATE * distinct, rel=0.35), (
        "the injected redelivery rate has drifted"
    )

    # Late arrival: deliveries that landed after something which happened later.
    assert report.out_of_order_arrivals > 0, (
        "nothing arrived out of order - event-time ordering was never exercised"
    )

    # And ordering did real work: at least one failure was overtaken by a
    # later capture and produced no claim.
    assert report.superseded_by_capture > 0, "no capture superseded a failure"

    assert report.claims_created > 0
    assert report.claims_created == await conn.fetchval("SELECT count(*) FROM claims")


async def test_every_dialect_survives_the_full_path(conn, adversarial_batch) -> None:
    """All four leak surfaces normalise onto one claim type.

    One object means one decision loop, one budget, one gate and one ledger.
    Four would mean four systems that cannot share a contact budget, which is
    the largest single source of both harassment risk and wasted spend.
    """
    pipeline = build_pipeline()
    at = max(item.received_at for item in adversarial_batch) + timedelta(minutes=1)
    report = await pipeline.ingest(conn, adversarial_batch, at)

    sources = {claim.evidence_structured["source"] for claim in report.claims}
    assert sources == set(SOURCES)

    produced = {claim.claim_type for claim in report.claims}
    assert produced == set(ClaimType)

    rails = {claim.rail for claim in report.claims}
    assert rails == set(Rail)

    assert all(claim.state is ClaimState.DETECTED for claim in report.claims)
    assert all(claim.evidence_ref is not None for claim in report.claims)
    assert all(int(claim.ltv_remaining_paise) > 0 for claim in report.claims)


async def test_the_whole_batch_leaves_no_free_text_in_the_ledger(conn, adversarial_batch) -> None:
    """The boundary holds across hundreds of real payloads, not one fixture.

    The narrations in this batch carry names the simulator generated. If any
    of them reached the hash chain, erasure would already be impossible.
    """
    pipeline = build_pipeline()
    at = max(item.received_at for item in adversarial_batch) + timedelta(minutes=1)
    report = await pipeline.ingest(conn, adversarial_batch, at)
    assert report.claims_created > 50

    rows = await conn.fetch("SELECT body_canonical FROM decision_ledger")
    assert rows
    blob = "\n".join(row["body_canonical"] for row in rows)

    for marker in ("narration", "ACH DR RTN", "@example.com", "+91", "UMRN"):
        assert marker not in blob, f"{marker!r} reached the decision ledger"

    # The claims table is equally clean.
    evidence = await conn.fetch("SELECT evidence_structured::text AS body FROM claims")
    joined = "\n".join(row["body"] for row in evidence)
    for marker in ("narration", "@example.com", "+91", "UMRN", "ifsc"):
        assert marker not in joined, f"{marker!r} reached the claims table"


async def test_ingest_is_idempotent_across_reruns(conn, adversarial_batch) -> None:
    """Replaying the same batch creates nothing new.

    Dedupe is what makes a retried ingest safe, and a retried ingest is the
    normal case after any operational incident.
    """
    pipeline = build_pipeline()
    at = max(item.received_at for item in adversarial_batch) + timedelta(minutes=1)

    first = await pipeline.ingest(conn, adversarial_batch, at)
    claims_after_first = await conn.fetchval("SELECT count(*) FROM claims")

    second = await pipeline.ingest(conn, adversarial_batch, at)
    assert second.claims_created == 0
    assert second.deduplicated == len(adversarial_batch)
    assert await conn.fetchval("SELECT count(*) FROM claims") == claims_after_first
    assert first.claims_created > 0


# ---------------------------------------------------------------------------
# Money on the wire, and the per-source breaker
# ---------------------------------------------------------------------------
def test_rupee_strings_parse_to_paise_without_a_float() -> None:
    """GI-2 holds at the boundary too, which is the easiest place to lose it.

    `float("2449.47") * 100` is 244946.99999999997. A rounding rule that is
    right almost always is exactly the silent, compounding error integer paise
    exist to prevent, and the headline number is a sum of these.
    """
    assert paise_from_rupee_string("2449.47") == 244947
    assert paise_from_rupee_string("0.01") == 1
    assert paise_from_rupee_string("1299.00") == 129900
    assert paise_from_rupee_string("7") == 700
    assert paise_from_rupee_string(129900) == 129900

    for bad in ("12.345", "1,299.00", "abc", "-5.00", "", None, 12.5):
        with pytest.raises(MalformedPayload):
            paise_from_rupee_string(bad)

    # And it really is exact across the range, not merely on the examples.
    for rupees in range(0, 400):
        for pennies in range(0, 100, 7):
            text = f"{rupees}.{pennies:02d}"
            assert paise_from_rupee_string(text) == rupees * 100 + pennies


def test_amounts_survive_the_full_path_exactly() -> None:
    """A decimal rupee string on the wire becomes the same integer paise."""
    event = NachAdapter(SECRET).parse(
        nach_body(event_id="NACH_MONEY", on="2025-11-03", amount="2449.47", account_ref="acct_m")
    )
    assert event.amount_paise == 244947
    assert isinstance(event.amount_paise, int)

    claim, _ = normaliser().normalise(event)
    assert claim.amount_paise == 244947
    assert claim.ltv_remaining_paise == 244947 * 6


async def test_one_bad_source_does_not_stall_the_others(conn) -> None:
    """A misbehaving gateway trips itself and leaves the rest serving.

    Never a silent drop: the refusal is counted, so the console can show which
    gateway stopped speaking a dialect we understand.
    """
    breakers = SourceBreakers(threshold=3)
    pipeline = build_pipeline()
    pipeline._breakers = breakers

    garbage = canonical({"entity": "event", "event": "payment.failed", "created_at": 1})
    good = card_delivery(
        event_id="pay_healthy",
        at=T0,
        amount_paise=129900,
        customer_ref="cust_healthy",
        account_ref="acct_healthy",
    )
    nach_ok = delivery(
        "npci_nach",
        nach_body(event_id="NACH_OK", on="2025-11-03", amount="100.00", account_ref="acct_ok"),
        T0,
    )

    report = await pipeline.ingest(conn, [delivery("pgw", garbage, T0)] * 4 + [good, nach_ok], T0)

    assert report.parse_failures == 3
    assert breakers.state("pgw", T0) is BreakerState.OPEN
    assert breakers.state("npci_nach", T0) is BreakerState.CLOSED
    assert report.tripped == ["pgw"]

    # The healthy source got through while the broken one was refused.
    assert report.claims_created == 1
    assert report.claims[0].rail is Rail.ENACH


def test_breaker_recovers_after_its_cooldown() -> None:
    breakers = SourceBreakers(threshold=2, cooldown=timedelta(minutes=5))
    breakers.record_failure("pgw", T0)
    breakers.record_failure("pgw", T0)

    assert breakers.state("pgw", T0) is BreakerState.OPEN
    with pytest.raises(SourceTripped):
        breakers.admit("pgw", T0)

    assert breakers.state("pgw", T0 + timedelta(minutes=6)) is BreakerState.HALF_OPEN
    breakers.admit("pgw", T0 + timedelta(minutes=6))

    breakers.record_success("pgw")
    assert breakers.state("pgw", T0 + timedelta(minutes=6)) is BreakerState.CLOSED


def test_folding_reports_what_ordering_actually_did() -> None:
    """The counters the acceptance gate asserts on come from real folding."""

    def event(event_id: str, minutes: int, captured: bool) -> RawEvent:
        return RawEvent(
            source="pgw",
            event_id=event_id,
            event_timestamp=T0 + timedelta(minutes=minutes),
            kind=WireKind.DEBIT_CAPTURED if captured else WireKind.DEBIT_FAILED,
            rail=Rail.CARD,
            account_ref="acct_fold",
            customer_ref="cust_fold",
            amount_paise=paise(1000),
            raw=b"{}",
            raw_hash=hashlib.sha256(b"{}").digest(),
            succeeded=captured,
        )

    folded = fold_by_account([event("b", 30, True), event("a", 0, False)])
    assert folded.superseded == 1
    assert folded.timelines[0].resolved
    assert folded.timelines[0].failed_attempts == 1

    unresolved = fold_by_account([event("a", 0, False), event("b", 30, False)])
    assert unresolved.superseded == 0
    assert unresolved.timelines[0].failed_attempts == 2
    assert unresolved.timelines[0].latest.event_id == "b"
