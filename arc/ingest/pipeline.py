"""The ingest path, in the order the steps have to happen.

    admit  ->  verify  ->  ARCHIVE  ->  parse  ->  dedupe
           ->  order by event time  ->  fold per account
           ->  normalise (THE REDACTION BOUNDARY)
           ->  assign arms per SUBJECT  ->  persist  ->  ledger

Three orderings in that chain are not stylistic.

VERIFY BEFORE PARSE, on the raw bytes, because an unverified webhook is
attacker-controlled input to a money-moving system.

ARCHIVE BEFORE PARSE, so a parser bug is recoverable by replaying the original
bytes. The deliveries that most need replaying are the ones that failed to
parse, and those are exactly the ones a parse-then-archive order would lose.

ORDER BY EVENT TIME, NOT ARRIVAL, because a capture can arrive before the
failure it supersedes. Processed by arrival that produces a claim for money
already collected - and a claim that should not exist is worse than a missing
one, because it gets diagnosed, funded, and messaged to somebody who paid.

Dedupe runs immediately after the parse rather than before it, because the
event id lives in the body of these dialects rather than in a header. It is
still ahead of every effect: nothing is stored, decided or contacted before an
event id has been claimed exactly once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from arc.core.money import paise
from arc.core.time_authority import ensure_utc
from arc.core.types import Claim
from arc.ingest.adapters import Adapter, signature_timestamp
from arc.ingest.adapters.base import payload_hash
from arc.ingest.archive import RawArchive
from arc.ingest.breaker import SourceBreakers, SourceTripped
from arc.ingest.dedupe import Dedupe
from arc.ingest.events import MalformedPayload, RawEvent
from arc.ingest.normaliser import ClaimContext, Normaliser, with_evidence_ref
from arc.ingest.ordering import AccountTimeline, count_out_of_order, fold_by_account
from arc.ledger.decision_ledger import DecisionLedger, LedgerEntry, LedgerEventType
from arc.ledger.subject_store import SubjectStore
from arc.proving_ground.arms import (
    Arm,
    ArmRegistry,
    Strata,
    claim_count_bucket,
    decile_cutoffs,
    value_decile,
)

# The most a delivery may lag the signature it carries. The timestamp is
# inside the signed material, so an attacker cannot move it, which makes
# replaying yesterday's capture a bounded attack rather than an open one.
# Generous, because a gateway retrying delivery for hours is normal and a
# webhook rejected for lateness is money silently not recovered.
MAX_SIGNATURE_AGE = timedelta(hours=48)


@dataclass(frozen=True)
class Delivery:
    """One HTTP delivery, exactly as it arrived. Untrusted until verified."""

    source: str
    raw: bytes
    signature: str
    received_at: datetime

    def __post_init__(self) -> None:
        ensure_utc(self.received_at)


@dataclass
class IngestReport:
    """What the batch did. Every rejection is counted, never swallowed.

    `deduplicated` and `out_of_order_arrivals` are asserted non-zero by the
    acceptance gate on adversarial traffic: a change that quietly stopped
    exercising redelivery or late arrival would otherwise pass as green.
    """

    delivered: int = 0
    archived: int = 0
    rejected_signature: int = 0
    stale_signature: int = 0
    unknown_source: int = 0
    parse_failures: int = 0
    deduplicated: int = 0
    out_of_order_arrivals: int = 0
    superseded_by_capture: int = 0
    resolved_without_claim: int = 0
    claims_created: int = 0
    subjects: int = 0
    tripped: list[str] = field(default_factory=list)
    arms: dict[str, Arm] = field(default_factory=dict)
    claims: list[Claim] = field(default_factory=list)

    # The arm each individual claim was stamped with, recorded per claim so
    # the SUTVA guard can be asserted on what the pipeline decided rather than
    # only on what the schema would allow. `subject_arms` has subject_token as
    # its primary key, so a database read can never show one subject in two
    # arms - which would make that assertion true by construction and prove
    # nothing about the code that assigns them.
    claim_arms: dict[UUID, Arm] = field(default_factory=dict)


class IngestPipeline:
    """L0 and L1, wired in the only order that is safe."""

    def __init__(
        self,
        *,
        adapters: dict[str, Adapter],
        normaliser: Normaliser,
        arms: ArmRegistry,
        subject_store: SubjectStore,
        ledger: DecisionLedger,
        archive: RawArchive | None = None,
        dedupe: Dedupe | None = None,
        breakers: SourceBreakers | None = None,
        max_signature_age: timedelta = MAX_SIGNATURE_AGE,
    ) -> None:
        self._adapters = adapters
        self._normaliser = normaliser
        self._arms = arms
        self._subject_store = subject_store
        self._ledger = ledger
        self._archive = archive or RawArchive()
        self._dedupe = dedupe or Dedupe()
        self._breakers = breakers or SourceBreakers()
        self._max_signature_age = max_signature_age

    async def accept(
        self, conn: Any, deliveries: Iterable[Delivery], at: datetime
    ) -> tuple[list[RawEvent], IngestReport]:
        """Verify, archive, parse, dedupe. What survives, in arrival order."""
        ensure_utc(at)
        report = IngestReport()
        accepted: list[RawEvent] = []

        for delivery in deliveries:
            report.delivered += 1

            try:
                self._breakers.admit(delivery.source, at)
            except SourceTripped:
                if delivery.source not in report.tripped:
                    report.tripped.append(delivery.source)
                continue

            adapter = self._adapters.get(delivery.source)
            digest = payload_hash(delivery.raw)

            # Verification happens on the bytes, before anything reads them.
            valid = adapter is not None and adapter.verify(delivery.raw, delivery.signature)

            # Archived either way. Refusing to keep an unverified delivery
            # would discard the evidence of an attack.
            archived = await self._archive.store(
                conn,
                source=delivery.source,
                raw=delivery.raw,
                signature=delivery.signature,
                payload_hash=digest,
                received_at=delivery.received_at,
                signature_valid=valid,
            )
            report.archived += 1

            if adapter is None:
                report.unknown_source += 1
                await self._archive.annotate(
                    conn, archived.archive_id, parse_error="unknown source"
                )
                continue

            if not valid:
                report.rejected_signature += 1
                self._breakers.record_failure(delivery.source, at)
                await self._archive.annotate(
                    conn, archived.archive_id, parse_error="signature invalid"
                )
                continue

            # Freshness is measured at ARRIVAL, not against the batch clock.
            # The question is whether this delivery reached us promptly after
            # it was signed, which is what a replayed capture fails. Measuring
            # against `at` instead would refuse a legitimate backfill of two
            # weeks of history purely for being history.
            signed_at = signature_timestamp(delivery.signature)
            age = delivery.received_at - signed_at if signed_at is not None else None
            if age is not None and age > self._max_signature_age:
                report.stale_signature += 1
                await self._archive.annotate(
                    conn, archived.archive_id, parse_error="signature too old"
                )
                continue

            try:
                event = adapter.parse(delivery.raw)
            except MalformedPayload as exc:
                report.parse_failures += 1
                self._breakers.record_failure(delivery.source, at)
                await self._archive.annotate(conn, archived.archive_id, parse_error=str(exc))
                continue

            # A NACH file reports a date and no instant. The signed delivery
            # timestamp is the honest instant, and it cannot be forged because
            # it sits inside the signed material.
            if event.date_only and signed_at is not None:
                event = replace(event, event_timestamp=signed_at, date_only=False)

            self._breakers.record_success(delivery.source)
            await self._archive.annotate(
                conn,
                archived.archive_id,
                event_id=event.event_id,
                event_timestamp=event.event_timestamp,
            )

            verdict = await self._dedupe.claim(conn, event.source, event.event_id, at)
            if not verdict.is_new:
                report.deduplicated += 1
                continue

            accepted.append(event)

        report.out_of_order_arrivals = count_out_of_order(accepted)
        return accepted, report

    async def ingest(self, conn: Any, deliveries: Iterable[Delivery], at: datetime) -> IngestReport:
        """The whole path, ending in persisted claims with arms attached."""
        accepted, report = await self.accept(conn, deliveries, at)
        folded = fold_by_account(accepted)
        report.superseded_by_capture = folded.superseded

        pending: list[tuple[AccountTimeline, RawEvent]] = []
        for timeline in folded.timelines:
            if timeline.resolved:
                report.resolved_without_claim += 1
                continue
            pending.append((timeline, timeline.latest))

        report.arms = await self._assign_arms(conn, pending, at)
        report.subjects = len(report.arms)

        for timeline, event in pending:
            claim = await self._persist(conn, timeline, event, at)
            token = self._normaliser.subject_token_for(event)
            report.claim_arms[claim.claim_id] = report.arms[token]
            report.claims.append(claim)
            report.claims_created += 1

        return report

    async def _assign_arms(
        self, conn: Any, pending: Sequence[tuple[AccountTimeline, RawEvent]], at: datetime
    ) -> dict[str, Arm]:
        """One arm per subject, from the batch's own strata, assigned once.

        The claim-count bucket and the value decile are computed across the
        whole batch before any assignment is made. Assigning as claims stream
        past would put a subject's first claim in a different stratum from
        their third, and the persisted registry would then be papering over an
        instability rather than guarding against a genuine race.
        """
        by_subject: dict[str, list[RawEvent]] = {}
        for _, event in pending:
            token = self._normaliser.subject_token_for(event)
            by_subject.setdefault(token, []).append(event)

        totals = {
            token: paise(sum(int(event.amount_paise) for event in events))
            for token, events in by_subject.items()
        }
        cutoffs = decile_cutoffs(list(totals.values()))

        assigned: dict[str, Arm] = {}
        for token, events in by_subject.items():
            strata = Strata(
                claim_count_bucket=claim_count_bucket(len(events)),
                value_decile=value_decile(totals[token], cutoffs),
                # The subject's largest claim decides the rail they stratify
                # on. A subject is one row in the design, so one rail.
                rail=max(events, key=lambda event: int(event.amount_paise)).rail,
            )
            assignment = await self._arms.assign_once(conn, token, strata, at)
            assigned[token] = assignment.arm
        return assigned

    async def _persist(
        self, conn: Any, timeline: AccountTimeline, event: RawEvent, at: datetime
    ) -> Claim:
        """Store the erasable half, then the ledgerable half, then the trail."""
        claim, record = self._normaliser.normalise(
            event, ClaimContext(failed_attempts=timeline.failed_attempts)
        )

        ref = await self._subject_store.put(conn, record.subject_token, record.payload)
        claim = with_evidence_ref(claim, ref)

        await _insert_claim(conn, claim)
        await conn.execute(
            "UPDATE raw_events SET subject_token = $2 WHERE source = $1 AND event_id = $3",
            event.source,
            claim.subject_token,
            event.event_id,
        )
        await self._ledger.append(
            conn,
            LedgerEntry(
                event_type=LedgerEventType.CLAIM_DETECTED,
                occurred_at=at,
                claim_id=claim.claim_id,
                subject_token=claim.subject_token,
                payload={
                    "detected_at": claim.detected_at,
                    "amount_paise": int(claim.amount_paise),
                    "ltv_remaining_paise": int(claim.ltv_remaining_paise),
                    "claim_type": str(claim.claim_type),
                    "rail": str(claim.rail),
                    "evidence": dict(claim.evidence_structured),
                    "evidence_ref": claim.evidence_ref,
                    "evidence_hash": claim.evidence_hash,
                },
            ),
        )
        return claim


async def _insert_claim(conn: Any, claim: Claim) -> None:
    await conn.execute(
        """
        INSERT INTO claims
            (claim_id, subject_token, amount_paise, ltv_remaining_paise,
             claim_type, rail, detected_at, evidence_structured,
             evidence_ref, evidence_hash, state)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10, $11)
        ON CONFLICT (claim_id) DO NOTHING
        """,
        claim.claim_id,
        claim.subject_token,
        int(claim.amount_paise),
        int(claim.ltv_remaining_paise),
        claim.claim_type.value,
        claim.rail.value,
        claim.detected_at,
        json.dumps(dict(claim.evidence_structured), sort_keys=True, default=str),
        claim.evidence_ref,
        claim.evidence_hash,
        claim.state.value,
    )
