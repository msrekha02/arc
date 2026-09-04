"""L1: THE REDACTION BOUNDARY. Four worlds collapse into one Claim here.

Everything that arrives is split in two and the two halves go to different
stores that have different rules.

    Claim           pseudonymous, structured, closed vocabulary, ledgerable,
                    hash-chained, and therefore never erasable
    SubjectRecord   the name, the number, the bank narration, the raw payload,
                    encrypted under a per-subject key and destroyed on request

A claim carries a pointer and a digest across that line. Nothing else.

WHY the boundary is HERE and not only in front of the LLM: a bank narration
flows downstream attached to the claim. Scrub only the model's input and the
narration still reaches the Decision Ledger, where it is hash-chained and can
never be removed without breaking the chain. There is no later place to put
this - by the time anything else sees the claim, the damage is already done.

The 128-character cap on evidence strings from M1 is a SHAPE rule and makes
smuggling awkward. It is not the defence. The defence is the PII guard, run
here as well as at the ledger boundary, because a claim that has already been
built wrong will be written by somebody eventually.

`normalise` performs no I/O. It is a pure function of the event and its
context, which is what lets the whole boundary be tested without a database.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from arc.core.ids import claim_id as derive_claim_id
from arc.core.ids import is_subject_token, subject_token
from arc.core.money import Paise
from arc.core.types import Claim, ClaimState, ClaimType, Rail
from arc.ingest.events import RawEvent, WireKind
from arc.ledger.pii_guard import PIIGuard

MANDATE_REF_PREFIX = "mnd_"
MANDATE_REF_DIGITS = 32


class UnresolvableIdentity(ValueError):
    """Nothing in the payload identifies a subject. The claim is refused.

    Fails closed rather than inventing a token: a claim under a made-up
    identity would be randomised into an arm of its own and contaminate the
    subject-level design it was supposed to belong to.
    """


class MissingValueEstimate(ValueError):
    """No `ltv_remaining` for this claim, so it cannot be prioritised.

    Rejecting is correct. A default would let a claim through carrying a value
    nobody chose, and the allocator's objective is weighted by exactly that
    number.
    """


@dataclass(frozen=True)
class SubjectRecord:
    """The erasable half. Goes to the subject store and NOWHERE else.

    There is deliberately no method here that returns a ledger-safe projection
    of this object. A convenience like that is how a name reaches the chain.
    """

    subject_token: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not is_subject_token(self.subject_token):
            raise ValueError("a subject record must be keyed by a derived token")


@dataclass(frozen=True)
class ClaimContext:
    """Batch-derived facts the event itself cannot know.

    `failed_attempts` counts presentations of the same obligation, which is a
    property of the timeline rather than of any one delivery.
    """

    failed_attempts: int = 1
    superseded_by_capture: bool = False
    extra_evidence: Mapping[str, Any] = field(default_factory=dict)


def default_identity(event: RawEvent) -> str:
    """The gateway's customer id if it has one, else the phone, else the email.

    Three rails out of four carry no customer id, which is a real problem and
    not a simulator artifact: identity resolution across rails is the merchant's
    job. A merchant with a customer directory injects a better resolver; this
    is the honest fallback, and it fails rather than guessing.
    """
    if event.customer_ref:
        return event.customer_ref
    if event.personal.phone:
        return event.personal.phone
    if event.personal.email:
        return event.personal.email
    raise UnresolvableIdentity(
        f"{event.source}/{event.event_id} carries no customer id, phone or email"
    )


_CLAIM_TYPE_BY_RAIL: Mapping[Rail, ClaimType] = {
    Rail.CARD: ClaimType.CARD_DECLINE,
    Rail.ENACH: ClaimType.MANDATE_FAILURE,
    Rail.UPI_AUTOPAY: ClaimType.MANDATE_FAILURE,
    Rail.INVOICE: ClaimType.INVOICE_OVERDUE,
}


def claim_type_of(event: RawEvent) -> ClaimType:
    """What kind of leak this is. A DECISION, which is why it is not in L0."""
    if event.kind is WireKind.INVOICE_OVERDUE:
        return ClaimType.INVOICE_OVERDUE
    if event.kind is WireKind.CHECKOUT_ABANDONED:
        return ClaimType.CHECKOUT_ABANDON
    return _CLAIM_TYPE_BY_RAIL[event.rail]


def pseudonymous_mandate_ref(raw_mandate: str) -> str:
    """A stable pseudonym for a mandate identifier.

    A UMRN carries a long digit run and the write-guard refuses it, correctly:
    it identifies a bank instrument. The Sentinel still needs to group debits
    by mandate at M6, so what crosses the boundary is a one-way derivation with
    the same shape as a subject token - which the guard masks as a system
    identifier rather than reading as an account number.
    """
    digest = hashlib.sha256(raw_mandate.encode("utf-8")).hexdigest()
    return f"{MANDATE_REF_PREFIX}{digest[:MANDATE_REF_DIGITS]}"


class Normaliser:
    """The boundary. One call in, two objects out, on opposite sides of it."""

    def __init__(
        self,
        *,
        pepper: bytes,
        ltv_source: Callable[[RawEvent], Paise | None],
        identity: Callable[[RawEvent], str] = default_identity,
        guard: PIIGuard | None = None,
    ) -> None:
        if not isinstance(pepper, bytes) or len(pepper) < 16:
            raise ValueError("pepper must be at least 16 bytes of key material")
        self._pepper = pepper
        self._ltv_source = ltv_source
        self._identity = identity
        self._guard = guard or PIIGuard()

    def subject_token_for(self, event: RawEvent) -> str:
        """One-way derivation of the subject's pseudonymous token.

        The raw identifier stops here. Everything downstream sees the token,
        which is what makes the Decision Ledger safe to hash-chain.
        """
        return subject_token(self._identity(event), pepper=self._pepper)

    def normalise(
        self, event: RawEvent, context: ClaimContext | None = None
    ) -> tuple[Claim, SubjectRecord]:
        """Split one delivery across the boundary.

        Returns the ledgerable claim and the erasable record. The claim's
        `evidence_ref` is None until the caller has stored the record and knows
        where it went - the boundary does no I/O, so it cannot know that here.
        """
        context = context or ClaimContext()
        token = self.subject_token_for(event)

        ltv = self._ltv_source(event)
        if ltv is None:
            raise MissingValueEstimate(
                f"{event.source}/{event.event_id} has no ltv_remaining; "
                "a claim without a value estimate cannot be prioritised"
            )

        evidence = self._evidence(event, context)

        # Defence in depth. The ledger guard would catch this later, but later
        # means after the claim has been passed around, and a claim built wrong
        # is one somebody eventually writes.
        self._guard.scan(evidence)

        claim = Claim(
            claim_id=derive_claim_id(event.source, event.event_id),
            subject_token=token,
            amount_paise=event.amount_paise,
            ltv_remaining_paise=ltv,
            claim_type=claim_type_of(event),
            rail=event.rail,
            detected_at=event.event_timestamp,
            evidence_structured=evidence,
            evidence_ref=None,
            evidence_hash=event.raw_hash,
            cause=None,
            state=ClaimState.DETECTED,
        )
        return claim, SubjectRecord(subject_token=token, payload=self._record(event))

    def _evidence(self, event: RawEvent, context: ClaimContext) -> dict[str, Any]:
        """The ledgerable half: closed vocabulary, scalars, no free text.

        Every field here is one a later milestone actually reads. Nothing is
        carried "in case it is useful" - an unused field on the wrong side of
        the boundary is a liability with no offsetting benefit.
        """
        mandate = event.personal.identifiers.get("mandate")
        evidence: dict[str, Any] = {
            "source": event.source,
            "wire_kind": str(event.kind),
            "rail": str(event.rail),
            "account_ref": event.account_ref,
            "attempt": event.attempt,
            "failed_attempts": context.failed_attempts,
            "initiated_by": str(event.initiated_by),
        }
        optional_fields = {
            "decline_code": event.decline_code,
            "advice_code": event.advice_code,
            "issuer_ref": event.issuer_ref,
            "ageing_bucket": event.ageing_bucket,
            "days_overdue": event.days_overdue,
            "mandate_ref": pseudonymous_mandate_ref(mandate) if mandate else None,
        }
        evidence.update({k: v for k, v in optional_fields.items() if v is not None})
        evidence.update(context.extra_evidence)
        return evidence

    def _record(self, event: RawEvent) -> dict[str, Any]:
        """The erasable half: everything personal, plus the raw payload.

        The raw bytes are kept here as well as in the archive so that erasure
        has one authoritative place to destroy. The archive is separately
        swept, because it is deletable - unlike the ledger, which is why the
        ledger never sees any of this.
        """
        personal = event.personal
        return {
            "source": event.source,
            "event_id": event.event_id,
            "account_ref": event.account_ref,
            "name": personal.name,
            "email": personal.email,
            "phone": personal.phone,
            "narration": personal.narration,
            "identifiers": dict(personal.identifiers),
            "raw_payload": event.raw.decode("utf-8", errors="replace"),
        }


def with_evidence_ref(claim: Claim, ref: str) -> Claim:
    """Attach the subject-store pointer once the record has been written."""
    return replace(claim, evidence_ref=ref)
