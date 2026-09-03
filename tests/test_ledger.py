"""M2 acceptance gate: the three stores, the hash chain, and the write-guard.

The six named tests in the build doc are:

    test_chain_verifies_over_1000_entries
    test_tampered_entry_breaks_verification
    test_pii_in_bank_narration_fails_the_write
    test_crypto_shred_leaves_chain_valid
    test_tombstone_recorded_on_erasure
    test_recovery_reversed_moves_total_down

Every test runs inside a transaction that is rolled back, so the global ledger
sequence is not shared between them.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import asyncpg
import pytest
from arc.core.ids import subject_token
from arc.core.money import Paise, paise
from arc.ledger.decision_ledger import (
    GENESIS_HASH,
    DecisionLedger,
    LedgerEntry,
    LedgerEventType,
    canonical_json,
    entry_hash,
)
from arc.ledger.money_ledger import (
    IllegalMoneyTransition,
    InsufficientBalance,
    MoneyAccount,
    MoneyLedger,
)
from arc.ledger.pii_guard import PIIDetected, PIIGuard, PIIKind, luhn_valid
from arc.ledger.subject_store import SubjectErased, SubjectStore, SubjectUnknown

DSN = os.environ.get("DATABASE_URL", "postgresql://arc:arc@localhost:5432/arc")

PEPPER = b"m2-acceptance-gate-pepper-000000"
TOKEN = subject_token("+919876543210", pepper=PEPPER)
OTHER_TOKEN = subject_token("priya.sharma@example.com", pepper=PEPPER)

T0 = datetime(2026, 3, 17, 6, 30, tzinfo=UTC)

# The narration the build doc names. 43 characters, so M1's length cap passes
# it; it has to be refused on content.
BANK_NARRATION = "RAJESH KUMAR AC 50100234567890 INSUF FUNDS"


def deterministic_keys() -> Callable[[int], bytes]:
    """Distinct-but-reproducible key material, so nonces are never reused."""
    counter = itertools.count(1)
    return lambda n: hashlib.sha256(f"m2-key-{next(counter)}".encode()).digest()[:n]


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


@pytest.fixture
def ledger() -> DecisionLedger:
    return DecisionLedger()


@pytest.fixture
def subjects(ledger: DecisionLedger) -> SubjectStore:
    return SubjectStore(ledger, key_source=deterministic_keys())


@pytest.fixture
def money(ledger: DecisionLedger) -> MoneyLedger:
    return MoneyLedger(ledger)


def decision_entry(index: int, claim: Any = None) -> LedgerEntry:
    return LedgerEntry(
        event_type=LedgerEventType.DECISION,
        occurred_at=T0 + timedelta(seconds=index),
        claim_id=claim or uuid4(),
        subject_token=TOKEN,
        payload={
            "intended_action": "whatsapp_utility",
            "pi_intended": 0.184,
            "realized_action": "whatsapp_utility",
            "veto_occurred": False,
            "blocking_rule_ids": [],
            "amount_paise": 129900 + index,
            "rule_registry_version": "2026.03.1",
        },
    )


# ---------------------------------------------------------------------------
# Gate test 1
# ---------------------------------------------------------------------------
async def test_chain_verifies_over_1000_entries(conn: Any, ledger: DecisionLedger) -> None:
    refs = [await ledger.append(conn, decision_entry(i)) for i in range(1000)]

    assert len(refs) == 1000
    assert refs[0].prev_hash == GENESIS_HASH
    assert refs[0].seq == 1
    assert refs[-1].seq == 1000

    # Each entry names its predecessor, and no two entries share a hash.
    for earlier, later in itertools.pairwise(refs):
        assert later.prev_hash == earlier.entry_hash
    assert len({ref.entry_hash for ref in refs}) == 1000

    assert await ledger.verify_chain(conn, 1, 1000) is True
    assert await ledger.find_breaks(conn, 1, 1000) == []

    # A window that does not start at the genesis still anchors correctly.
    assert await ledger.verify_chain(conn, 400, 600) is True


# ---------------------------------------------------------------------------
# Gate test 2
# ---------------------------------------------------------------------------
async def test_tampered_entry_breaks_verification(conn: Any, ledger: DecisionLedger) -> None:
    for i in range(20):
        await ledger.append(conn, decision_entry(i))
    assert await ledger.verify_chain(conn, 1, 20) is True

    # Simulate an attacker who already holds table ownership and has turned the
    # append-only trigger off. The chain has to catch what the trigger cannot.
    await conn.execute(
        "ALTER TABLE decision_ledger DISABLE TRIGGER trg_decision_ledger_append_only"
    )

    original = await conn.fetchval("SELECT body_canonical FROM decision_ledger WHERE seq = 7")
    body = json.loads(original)
    body["payload"]["realized_action"] = "do_nothing"
    forged = canonical_json(body).decode("utf-8")

    # Naive tamper: rewrite the body and leave the hashes alone.
    await conn.execute("UPDATE decision_ledger SET body_canonical = $1 WHERE seq = 7", forged)

    assert await ledger.verify_chain(conn, 1, 20) is False
    breaks = await ledger.find_breaks(conn, 1, 20)
    assert [b.seq for b in breaks] == [7]
    assert "does not hash" in breaks[0].reason

    # Sophisticated tamper: recompute this entry's own hash so it is internally
    # consistent. The next entry still names the old hash, so it is still caught.
    prev_hash = await conn.fetchval("SELECT prev_hash FROM decision_ledger WHERE seq = 7")
    await conn.execute(
        "UPDATE decision_ledger SET entry_hash = $1 WHERE seq = 7",
        entry_hash(prev_hash, forged.encode("utf-8")),
    )

    assert await ledger.verify_chain(conn, 1, 20) is False
    breaks = await ledger.find_breaks(conn, 1, 20)
    assert [b.seq for b in breaks] == [8]
    assert "prev_hash" in breaks[0].reason

    # Deleting an entry outright leaves a gap that verification reports.
    await conn.execute("DELETE FROM decision_ledger WHERE seq = 15")
    assert await ledger.verify_chain(conn, 1, 20) is False


async def test_ledger_is_append_only_at_the_database(conn: Any, ledger: DecisionLedger) -> None:
    """The trigger refuses first; the chain is the second line, not the only one."""
    await ledger.append(conn, decision_entry(0))

    # Each attempt gets its own savepoint: the first refusal aborts the
    # surrounding transaction, which would otherwise mask the second.
    for statement in (
        "UPDATE decision_ledger SET event_type = 'gate_veto' WHERE seq = 1",
        "DELETE FROM decision_ledger WHERE seq = 1",
    ):
        with pytest.raises(asyncpg.RaiseError, match="append-only"):
            async with conn.transaction():
                await conn.execute(statement)

    assert await conn.fetchval("SELECT count(*) FROM decision_ledger") == 1
    assert await ledger.verify_chain(conn, 1, 1) is True


# ---------------------------------------------------------------------------
# Gate test 3 - the critical one
# ---------------------------------------------------------------------------
async def test_pii_in_bank_narration_fails_the_write(conn: Any, ledger: DecisionLedger) -> None:
    """A narration carrying a name and an account number must not be written."""
    before = await ledger.head(conn)

    entry = LedgerEntry(
        event_type=LedgerEventType.CLAIM_DIAGNOSED,
        occurred_at=T0,
        claim_id=uuid4(),
        subject_token=TOKEN,
        payload={"decline_code": "MAC03", "narration": BANK_NARRATION},
    )

    with pytest.raises(PIIDetected) as caught:
        await ledger.append(conn, entry)

    kinds = {hit.kind for hit in caught.value.hits}
    assert PIIKind.NAME_TOKEN in kinds
    assert PIIKind.BANK_ACCOUNT in kinds
    assert all(hit.path.startswith("payload.narration") for hit in caught.value.hits)

    # The write did not happen. Not partially, not pending, not at all.
    assert await ledger.head(conn) == before
    assert await conn.fetchval("SELECT count(*) FROM decision_ledger") == 0

    # The guard does not leak what it caught into its own message.
    message = str(caught.value)
    assert "RAJESH" not in message
    assert "KUMAR" not in message
    assert "50100234567890" not in message

    # The same claim writes cleanly once the narration is behind a ref.
    clean = LedgerEntry(
        event_type=LedgerEventType.CLAIM_DIAGNOSED,
        occurred_at=T0,
        claim_id=entry.claim_id,
        subject_token=TOKEN,
        payload={"decline_code": "MAC03", "evidence_ref": f"subject://{TOKEN}/1"},
    )
    assert (await ledger.append(conn, clean)).seq == 1


@pytest.mark.parametrize(
    ("label", "payload", "expected"),
    [
        ("email", {"note": "reach priya.sharma@example.com"}, PIIKind.EMAIL),
        ("mobile", {"contact": "+91 9876543210"}, PIIKind.PHONE_IN),
        ("bare mobile", {"contact": "9876543210"}, PIIKind.PHONE_IN),
        ("aadhaar", {"uid": "2345 6789 0123"}, PIIKind.AADHAAR),
        ("pan card", {"tax_id": "ABCDE1234F"}, PIIKind.PAN_CARD),
        ("ifsc", {"branch": "HDFC0001234"}, PIIKind.IFSC),
        ("card pan", {"instrument": "4111 1111 1111 1111"}, PIIKind.CARD_PAN),
        ("bank account", {"beneficiary": "50100234567890"}, PIIKind.BANK_ACCOUNT),
        ("account as int", {"beneficiary": 50100234567890}, PIIKind.BANK_ACCOUNT),
        ("name only", {"caller": "Priya Sharma rang back"}, PIIKind.NAME_TOKEN),
        ("nested", {"prior": [{"narration": "ACH DR RETURN MEENAKSHI IYER"}]}, PIIKind.NAME_TOKEN),
        ("in a key", {"rajesh.kumar@example.com": 1}, PIIKind.EMAIL),
        ("in bytes", {"raw": b"PRIYA SHARMA AC 000123456789"}, PIIKind.NAME_TOKEN),
    ],
)
async def test_guard_refuses_every_pattern_in_the_set(
    conn: Any, ledger: DecisionLedger, label: str, payload: dict[str, Any], expected: PIIKind
) -> None:
    entry = LedgerEntry(
        event_type=LedgerEventType.DECISION, occurred_at=T0, claim_id=uuid4(), payload=payload
    )
    with pytest.raises(PIIDetected) as caught:
        await ledger.append(conn, entry)
    assert expected in {hit.kind for hit in caught.value.hits}, label
    assert await conn.fetchval("SELECT count(*) FROM decision_ledger") == 0


def test_guard_does_not_fire_on_a_real_decision_record() -> None:
    """False positives would push developers towards wanting a bypass."""
    PIIGuard().scan(
        {
            "intended_action": "voice_call",
            "pi_intended": 0.184,
            "gate_verdict": "block_permanent",
            "blocking_rule_ids": ["ABS-FORBORNE", "CD-VOICE", "TIME-WINDOW", "NET-MAC03"],
            "realized_action": "do_nothing",
            "veto_occurred": True,
            "shadow_prices": {"voice": 340.0, "contact": 12.5},
            "model_versions": {"bounce": "v3", "uplift": "v7"},
            "rule_registry_version": "2026.03.1",
            "amount_paise": 470000000,
            "ltv_remaining_paise": 1500000000,
            "feature_hash": "9f2c1a4b" * 8,
            "claim_id": str(uuid4()),
            "subject_token": TOKEN,
        }
    )


def test_guard_has_no_bypass() -> None:
    """The only public method raises. There is no inspect-and-continue seam."""
    public = {name for name in dir(PIIGuard) if not name.startswith("_")}
    assert public == {"scan"}


def test_luhn_keeps_long_numbers_out_of_card_pan() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


# ---------------------------------------------------------------------------
# Gate test 4
# ---------------------------------------------------------------------------
async def test_crypto_shred_leaves_chain_valid(
    conn: Any, ledger: DecisionLedger, subjects: SubjectStore
) -> None:
    ref = await subjects.put(
        conn,
        TOKEN,
        {"name": "Rajesh Kumar", "phone": "+919876543210", "narration": BANK_NARRATION},
    )
    second = await subjects.put(conn, TOKEN, {"transcript": "spoke to Rajesh Kumar on Tuesday"})

    for i in range(5):
        await ledger.append(conn, decision_entry(i))
    head_before = await ledger.head(conn)
    assert head_before is not None
    assert await ledger.verify_chain(conn, 1, head_before.seq) is True

    assert (await subjects.get(conn, ref))["name"] == "Rajesh Kumar"

    tombstone = await subjects.crypto_shred(
        conn, TOKEN, at=T0 + timedelta(days=1), requested_by="dpo:erasure-request-8814"
    )

    # The plaintext is gone, both records at once, because the key was the
    # thing destroyed rather than the rows.
    for gone in (ref, second):
        with pytest.raises(SubjectErased):
            await subjects.get(conn, gone)
    assert await subjects.is_shredded(conn, TOKEN) is True
    assert await conn.fetchval(
        "SELECT data_key IS NULL FROM subject_keys WHERE subject_token = $1", TOKEN
    )

    # The rows survive, unreadable, so the tombstone points at something.
    assert (
        await conn.fetchval("SELECT count(*) FROM subject_records WHERE subject_token = $1", TOKEN)
        == 2
    )

    # The chain never covered the plaintext, so it still verifies, including
    # over the tombstone that was appended by the erasure itself.
    assert await ledger.verify_chain(conn, 1, tombstone.ledger_ref.seq) is True
    assert await ledger.find_breaks(conn, 1, tombstone.ledger_ref.seq) == []
    assert tombstone.ledger_ref.seq == head_before.seq + 1


async def test_subject_data_is_ciphertext_at_rest(conn: Any, subjects: SubjectStore) -> None:
    await subjects.put(conn, TOKEN, {"name": "Rajesh Kumar", "narration": BANK_NARRATION})
    stored = await conn.fetchval("SELECT ciphertext FROM subject_records LIMIT 1")

    assert b"Rajesh" not in stored
    assert b"RAJESH" not in stored
    assert b"50100234567890" not in stored


async def test_erased_subject_cannot_be_rekeyed(conn: Any, subjects: SubjectStore) -> None:
    """Writing again after erasure would quietly undo it."""
    await subjects.put(conn, TOKEN, {"name": "Rajesh Kumar"})
    await subjects.crypto_shred(conn, TOKEN, at=T0, requested_by="dpo:test")

    with pytest.raises(SubjectErased):
        await subjects.put(conn, TOKEN, {"name": "Rajesh Kumar"})


async def test_shredding_one_subject_leaves_another_readable(
    conn: Any, subjects: SubjectStore
) -> None:
    mine = await subjects.put(conn, TOKEN, {"name": "Rajesh Kumar"})
    theirs = await subjects.put(conn, OTHER_TOKEN, {"name": "Priya Sharma"})

    await subjects.crypto_shred(conn, TOKEN, at=T0, requested_by="dpo:test")

    with pytest.raises(SubjectErased):
        await subjects.get(conn, mine)
    assert (await subjects.get(conn, theirs))["name"] == "Priya Sharma"


async def test_unknown_ref_is_distinguished_from_erased(conn: Any, subjects: SubjectStore) -> None:
    with pytest.raises(SubjectUnknown):
        await subjects.get(conn, f"subject://{TOKEN}/99")


# ---------------------------------------------------------------------------
# Gate test 5
# ---------------------------------------------------------------------------
async def test_tombstone_recorded_on_erasure(
    conn: Any, ledger: DecisionLedger, subjects: SubjectStore
) -> None:
    first = await subjects.put(conn, TOKEN, {"name": "Rajesh Kumar"})
    second = await subjects.put(conn, TOKEN, {"phone": "+919876543210"})

    at = T0 + timedelta(days=2)
    tombstone = await subjects.crypto_shred(conn, TOKEN, at=at, requested_by="dpo:req-8814")

    written = await ledger.entries_of_type(conn, LedgerEventType.TOMBSTONE)
    assert len(written) == 1
    recorded = written[0]

    # That erasure occurred, when, under whose request, and which refs died.
    assert recorded.event_type is LedgerEventType.TOMBSTONE
    assert recorded.occurred_at == at
    assert recorded.subject_token == TOKEN
    assert recorded.payload["reason"] == "erasure_request"
    assert recorded.payload["requested_by"] == "dpo:req-8814"
    assert recorded.payload["refs_destroyed"] == [first, second]
    assert recorded.payload["refs_destroyed_count"] == 2
    assert recorded.payload["key_destroyed"] is True

    assert tombstone.refs_destroyed == (first, second)
    assert tombstone.shredded_at == at

    # The tombstone itself carries no personal data, only the pseudonymous
    # token, so recording the erasure does not re-create what was erased.
    PIIGuard().scan(dict(recorded.payload))

    # A second request records that nothing further was destroyed.
    again = await subjects.crypto_shred(conn, TOKEN, at=at, requested_by="dpo:req-8815")
    entries = await ledger.entries_of_type(conn, LedgerEventType.TOMBSTONE)
    assert len(entries) == 2
    assert entries[1].payload["key_destroyed"] is False
    assert await ledger.verify_chain(conn, 1, again.ledger_ref.seq) is True


# ---------------------------------------------------------------------------
# Gate test 6
# ---------------------------------------------------------------------------
async def test_recovery_reversed_moves_total_down(conn: Any, money: MoneyLedger) -> None:
    claim = uuid4()
    amount = paise(129900)

    await money.open_claim(conn, claim, amount, at=T0)
    await money.transition(
        conn, claim, MoneyAccount.AT_RISK, MoneyAccount.IN_TREATMENT, amount, at=T0
    )
    await money.transition(
        conn, claim, MoneyAccount.IN_TREATMENT, MoneyAccount.RECOVERED, amount, at=T0
    )

    assert await money.recovered_total(conn) == 129900
    assert await money.is_balanced(conn) is True

    # The chargeback lands.
    await money.reverse_recovery(conn, claim, paise(50000), at=T0 + timedelta(days=3))

    assert await money.recovered_total(conn) == 79900
    assert await money.claim_balance(conn, claim, MoneyAccount.REVERSED) == 50000
    assert await money.is_balanced(conn) is True
    assert await money.unbalanced_groups(conn) == []

    # And again, taking the headline the whole way down.
    await money.reverse_recovery(conn, claim, paise(79900), at=T0 + timedelta(days=4))
    assert await money.recovered_total(conn) == 0
    assert await money.is_balanced(conn) is True


async def test_every_movement_writes_two_balancing_legs(conn: Any, money: MoneyLedger) -> None:
    claim = uuid4()
    movement = await money.open_claim(conn, claim, paise(500000), at=T0)

    legs = await conn.fetch(
        "SELECT account, delta_paise FROM money_entries WHERE group_id = $1 ORDER BY delta_paise",
        movement.group_id,
    )
    assert len(legs) == 2
    assert sum(leg["delta_paise"] for leg in legs) == 0
    assert {leg["account"] for leg in legs} == {"external", "at_risk"}

    balances = await money.claim_balances(conn, claim)
    assert balances[MoneyAccount.AT_RISK] == 500000
    assert balances[MoneyAccount.EXTERNAL] == -500000


async def test_illegal_money_transition_rejected(conn: Any, money: MoneyLedger) -> None:
    claim = uuid4()
    await money.open_claim(conn, claim, paise(129900), at=T0)

    # Money cannot settle without having been recovered.
    with pytest.raises(IllegalMoneyTransition):
        await money.transition(
            conn, claim, MoneyAccount.AT_RISK, MoneyAccount.SETTLED, paise(129900), at=T0
        )
    # Nor can a reversal be conjured from money at risk.
    with pytest.raises(IllegalMoneyTransition):
        await money.transition(
            conn, claim, MoneyAccount.AT_RISK, MoneyAccount.REVERSED, paise(129900), at=T0
        )


async def test_cannot_move_more_than_the_claim_holds(conn: Any, money: MoneyLedger) -> None:
    claim = uuid4()
    await money.open_claim(conn, claim, paise(129900), at=T0)

    with pytest.raises(InsufficientBalance):
        await money.transition(
            conn, claim, MoneyAccount.AT_RISK, MoneyAccount.RECOVERED, paise(200000), at=T0
        )
    assert await money.is_balanced(conn) is True


async def test_money_amounts_must_be_integer_paise(conn: Any, money: MoneyLedger) -> None:
    claim = uuid4()
    with pytest.raises(TypeError):
        await money.open_claim(conn, claim, 1299.0, at=T0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await money.open_claim(conn, claim, Paise(0), at=T0)


async def test_money_transitions_are_recorded_in_the_ledger(
    conn: Any, ledger: DecisionLedger, money: MoneyLedger
) -> None:
    claim = uuid4()
    await money.open_claim(conn, claim, paise(129900), at=T0)
    await money.transition(
        conn, claim, MoneyAccount.AT_RISK, MoneyAccount.RECOVERED, paise(129900), at=T0
    )
    await money.reverse_recovery(conn, claim, paise(129900), at=T0)

    entries = await ledger.entries_of_type(conn, LedgerEventType.MONEY_TRANSITION)
    assert [e.payload["to_account"] for e in entries] == ["at_risk", "recovered", "reversed"]
    assert [e.payload["is_recovery_reversal"] for e in entries] == [False, False, True]
    assert await ledger.verify_chain(conn, 1, len(entries)) is True


# ---------------------------------------------------------------------------
# Canonical form and replay
# ---------------------------------------------------------------------------
def test_canonical_json_is_stable_under_key_order() -> None:
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})
    assert canonical_json({"a": 1}) == b'{"a":1}'


def test_canonical_json_encodes_domain_types() -> None:
    claim = uuid4()
    encoded = json.loads(canonical_json({"id": claim, "at": T0, "digest": b"\x01\x02"}))
    assert encoded["id"] == str(claim)
    assert encoded["at"] == "2026-03-17T06:30:00+00:00"
    assert encoded["digest"] == "0102"


async def test_replay_returns_one_claims_entries_in_order(
    conn: Any, ledger: DecisionLedger
) -> None:
    mine, theirs = uuid4(), uuid4()
    for i in range(4):
        await ledger.append(conn, decision_entry(i, claim=mine))
        await ledger.append(conn, decision_entry(i, claim=theirs))

    replayed = await ledger.replay(conn, mine)
    assert len(replayed) == 4
    assert all(entry.claim_id == mine for entry in replayed)
    assert [e.occurred_at for e in replayed] == sorted(e.occurred_at for e in replayed)
    assert replayed[0].payload["intended_action"] == "whatsapp_utility"


async def test_entry_requires_utc(conn: Any) -> None:
    from arc.core.time_authority import NotUTC

    with pytest.raises(NotUTC):
        LedgerEntry(
            event_type=LedgerEventType.DECISION,
            occurred_at=datetime(2026, 3, 17, 6, 30),
        )
