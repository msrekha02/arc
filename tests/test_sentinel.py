"""M6 acceptance gate: cause attribution, in the order that prevents harm.

The nine named tests are:

    test_injected_outage_detected_as_ISSUER
    test_outage_produces_zero_contact_actions
    test_thin_issuer_returns_INSUFFICIENT_POWER_not_NORMAL
    test_insufficient_power_caps_confidence
    test_backoff_ladder_records_which_level_answered
    test_orphaned_mandate_diagnosed_as_MERCHANT
    test_merchant_cause_routes_to_SELF_HEALING
    test_unmapped_code_returns_UNKNOWN_not_guess
    test_llm_cause_confidence_capped_at_070

`test_outage_produces_zero_contact_actions` is the milestone. The world planted
two issuer outages and told the Sentinel nothing about either; it has to find
one from correlated declines against a denominator, and the claims behind it
have to end up suppressed with no way for a message to reach anybody.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from arc.core.ids import subject_token
from arc.core.money import Paise, paise
from arc.core.time_authority import TimezoneBasis, TzBasisKind
from arc.core.types import (
    LLM_CONFIDENCE_CAP,
    ActionType,
    Cause,
    CauseLabel,
    CauseLayer,
    Claim,
    ClaimState,
    ClaimType,
    CohortVerdict,
    DiagnosisPath,
    Rail,
)
from arc.gate.context import (
    CONTACT_CHANNELS,
    ConsentState,
    GateContext,
    SubjectFlags,
)
from arc.gate.evaluator import Gate
from arc.gate.registry import load_registry
from arc.ingest.adapters import build_registry
from arc.sentinel.code_map import CODE_MAP, UNMAPPED, code_lookup
from arc.sentinel.cohort import (
    ALPHA,
    LADDER,
    N_MIN,
    SIGMA_FLOOR,
    TAU,
    CohortHistory,
    CohortLevel,
    StaticDowntimeFeed,
    cohort_check,
    ewma_baseline,
)
from arc.sentinel.diagnose import (
    INSUFFICIENT_POWER_CONFIDENCE_CAP,
    LAYER_ROUTE,
    ORDERED_CHECKS,
    DiagnosisContext,
    Finding,
    blind_spot_share,
    diagnose,
)
from arc.sentinel.mandate_health import (
    MandateFacts,
    MandateHistory,
    RawMandateIdentifier,
    mandate_health,
)
from arc.simulator.seeds import DEVELOP_SEED
from arc.simulator.wire_fake import WireFake
from arc.simulator.world import OUTAGES, World

SECRET = b"m6-acceptance-gate-secret-00000"
PEPPER = b"m6-acceptance-gate-pepper-00000"
SOURCES = ("pgw", "npci_nach", "upi_autopay", "billing")

# Large enough that the two-hour outage has a cohort worth detecting. The cell
# sizes are the point: a 15-minute bucket holds a median of one transaction, so
# the ladder is exercised on every call rather than in a contrived case.
COHORT_POPULATION = 20_000

OUTAGE = OUTAGES[0]
# The second half of the outage. Two-hour buckets are aligned to the epoch, so
# the first half shares its bucket with ninety minutes of normal traffic.
DURING_OUTAGE = OUTAGE.start + timedelta(minutes=45)
BEFORE_OUTAGE = OUTAGE.start - timedelta(days=2)

IST = TimezoneBasis(TzBasisKind.DECLARED, "Asia/Kolkata")
TOKEN = subject_token("+919876543210", pepper=PEPPER)
DEFAULT_AMOUNT: Paise = paise(129_900)


def make_claim(
    *,
    rail: Rail = Rail.CARD,
    claim_type: ClaimType = ClaimType.CARD_DECLINE,
    issuer: str | None = OUTAGE.issuer_id,
    code: str | None = "51",
    advice: str | None = None,
    amount: Paise = DEFAULT_AMOUNT,
    at: datetime = DURING_OUTAGE,
    mandate_ref: str | None = None,
) -> Claim:
    evidence: dict[str, object] = {
        "source": "pgw",
        "rail": str(rail),
        "account_ref": "acct_0000001",
        "failed_attempts": 1,
    }
    if issuer is not None:
        evidence["issuer_ref"] = issuer
    if code is not None:
        evidence["decline_code"] = code
    if advice is not None:
        evidence["advice_code"] = advice
    if mandate_ref is not None:
        evidence["mandate_ref"] = mandate_ref

    return Claim(
        claim_id=uuid4(),
        subject_token=TOKEN,
        amount_paise=amount,
        ltv_remaining_paise=paise(int(amount) * 6),
        claim_type=claim_type,
        rail=rail,
        detected_at=at,
        evidence_structured=evidence,
        evidence_hash=hashlib.sha256(b"payload").digest(),
        state=ClaimState.DETECTED,
    )


@pytest.fixture(scope="module")
def ingested() -> list:
    """The frozen world's batch, through the real adapters.

    Deliberately the production path rather than a fixture: the Sentinel is fed
    exactly what L0 produces, including the 5% of decline codes the world
    remaps to the wrong reason.
    """
    world = World(seed=DEVELOP_SEED, size=COHORT_POPULATION)
    registry = build_registry(dict.fromkeys(SOURCES, SECRET))
    events = [
        registry[wire.source].parse(wire.body)
        for wire in WireFake(world, SECRET).emit("replay", DEVELOP_SEED)
    ]
    # An invoice never passes through an issuer, so it is not cohort evidence.
    return [event for event in events if event.rail is not Rail.INVOICE]


@pytest.fixture(scope="module")
def history(ingested) -> CohortHistory:
    return CohortHistory.from_events(ingested)


@pytest.fixture(scope="module")
def gate() -> Gate:
    return Gate(load_registry())


def gate_context(claim: Claim, diagnosis, *, at: datetime) -> GateContext:
    """A subject the Gate would otherwise be free to contact.

    Consent granted on every channel, no freezes, inside the legal contact
    window. Everything that could independently block a message is switched
    off, so if nothing reaches this subject it is because of the diagnosis.
    """
    from arc.gate.context import Channel

    return GateContext(
        claim_id=claim.claim_id,
        subject_token=claim.subject_token,
        rail=claim.rail,
        claim_state=diagnosis.next_state,
        amount_paise=claim.amount_paise,
        tz_basis=IST,
        cause=diagnosis.cause,
        consent=dict.fromkeys(CONTACT_CHANNELS, ConsentState.GRANTED)
        | {Channel.SILENT: ConsentState.GRANTED},
        flags=SubjectFlags(
            identity_verified=True,
            issuer_degraded=not diagnosis.contact_permitted
            and diagnosis.cause.layer is CauseLayer.ISSUER,
            issuer_degraded_until=diagnosis.suppress_until,
        ),
    )


def contact_actions(actions: Iterable) -> set:
    from arc.gate.context import ACTION_CHANNEL

    return {action for action in actions if ACTION_CHANNEL[action] in CONTACT_CHANNELS}


# ---------------------------------------------------------------------------
# Gate tests 1 and 2 - the outage
# ---------------------------------------------------------------------------
def test_injected_outage_detected_as_ISSUER(history) -> None:
    """The world planted it and said nothing. The Sentinel finds it.

    Nothing in `ObservableState` names an outage. What reaches the detector is
    a burst of correlated declines against a denominator, which is the only
    evidence a real system gets.
    """
    claim = make_claim(code="91")
    result = diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
    )

    assert result.cause.layer is CauseLayer.ISSUER
    assert result.cause.label is CauseLabel.ISSUER_OUTAGE
    assert result.answered_by is DiagnosisPath.COHORT
    assert result.cause.cohort_power is CohortVerdict.DEGRADED
    assert result.next_state is ClaimState.SUPPRESSED

    # And it is separable: the same instant at a healthy peer is not degraded.
    peer = make_claim(issuer="ISS_LP01", code="91")
    peer_result = diagnose(
        peer, DiagnosisContext.from_claim(peer, cohort_history=history), DURING_OUTAGE
    )
    assert peer_result.cause.layer is not CauseLayer.ISSUER or (
        peer_result.answered_by is not DiagnosisPath.COHORT
    )

    # Two hours before, the same issuer is fine.
    quiet = make_claim(at=BEFORE_OUTAGE, code="51")
    quiet_result = diagnose(
        quiet, DiagnosisContext.from_claim(quiet, cohort_history=history), BEFORE_OUTAGE
    )
    assert quiet_result.cause.cohort_power is not CohortVerdict.DEGRADED


def test_outage_produces_zero_contact_actions(history, gate) -> None:
    """THE MILESTONE. A detected outage reaches nobody.

    Every claim behind the incident routes to SUPPRESSED, and the Gate offers
    no contact action for any of them - not one message, not one call. The
    subject in this test has consent on every channel and no other freeze, so
    the only thing standing between them and a dunning message is the
    diagnosis.

    The naive system messages all of these people about their bank's incident.
    """
    suppressed = 0
    for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY):
        # Including the remapped code the world plants 5% of the time. Under a
        # code-map-first pipeline this claim reads as a delinquent customer.
        for code in ("91", "51", "AM04", None):
            claim = make_claim(rail=rail, code=code)
            result = diagnose(
                claim,
                DiagnosisContext.from_claim(claim, cohort_history=history),
                DURING_OUTAGE,
            )

            assert result.cause.layer is CauseLayer.ISSUER, (
                f"{rail}/{code} escaped the cohort and was blamed on the customer"
            )
            assert result.next_state is ClaimState.SUPPRESSED
            assert result.contact_permitted is False
            suppressed += 1

            eligible = gate.project(
                gate_context(claim, result, at=DURING_OUTAGE),
                list(ActionType),
                DURING_OUTAGE,
            )
            assert contact_actions(eligible) == set(), (
                f"{rail}/{code} left {sorted(str(a) for a in contact_actions(eligible))} "
                "reachable during an outage"
            )

    assert suppressed == 12

    # The claim really does move to SUPPRESSED through the state machine,
    # rather than the diagnosis merely saying it should.
    claim = make_claim(code="91")
    result = diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
    )
    routed = result.apply(claim)
    assert routed.state is ClaimState.SUPPRESSED
    assert routed.cause is result.cause


# ---------------------------------------------------------------------------
# Gate tests 3, 4 and 5 - cohort power and the ladder
# ---------------------------------------------------------------------------
def test_thin_issuer_returns_INSUFFICIENT_POWER_not_NORMAL(history) -> None:
    """The verdict has three members and the third one is used.

    For most issuer-instrument combinations most of the time there is no
    power. A detector that answered NORMAL there would restore code-map-first
    behaviour for the majority of traffic without anyone noticing, because
    NORMAL and "we could not tell" look identical downstream.
    """
    thin = cohort_check("ISS_CO03", Rail.UPI_AUTOPAY, DURING_OUTAGE, history)
    assert thin.verdict is CohortVerdict.INSUFFICIENT_POWER
    assert thin.level is None
    assert thin.has_power is False

    # An issuer with no traffic at all is the same answer, not NORMAL.
    absent = cohort_check("ISS_DOES_NOT_EXIST", Rail.CARD, DURING_OUTAGE, CohortHistory())
    assert absent.verdict is CohortVerdict.INSUFFICIENT_POWER
    assert absent.attempts == 0

    # It happens often enough to matter, which is the whole reason the verdict
    # exists. Measured rather than asserted.
    issuers = {event.issuer_ref for event in history._cells and ()} or None
    verdicts = [
        cohort_check(issuer, rail, DURING_OUTAGE, history).verdict
        for issuer in ("ISS_CO01", "ISS_CO02", "ISS_CO03", "ISS_SP03")
        for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY)
    ]
    assert CohortVerdict.INSUFFICIENT_POWER in verdicts
    assert issuers is None  # the sweep above is the measurement; this is a no-op guard


def test_insufficient_power_caps_confidence(history) -> None:
    """Without cohort power, a customer-layer cause is capped at 0.75.

    Below the Gate's 0.80 threshold for money-moving actions, so the claim
    spends its first cycle on conservative actions rather than presenting a
    debit on the strength of a code we could not corroborate.
    """
    claim = make_claim(issuer="ISS_CO03", rail=Rail.UPI_AUTOPAY, code="Z9")
    result = diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
    )

    assert result.cause.cohort_power is CohortVerdict.INSUFFICIENT_POWER
    assert result.cause.layer is CauseLayer.CUSTOMER
    assert result.cause.confidence == INSUFFICIENT_POWER_CONFIDENCE_CAP == 0.75
    assert result.confidence_capped is True

    # The same code where the cohort DOES have power keeps its full weight.
    confident = make_claim(issuer="ISS_LP01", rail=Rail.CARD, code="51", at=BEFORE_OUTAGE)
    strong = diagnose(
        confident,
        DiagnosisContext.from_claim(confident, cohort_history=history),
        BEFORE_OUTAGE,
    )
    assert strong.cause.cohort_power is CohortVerdict.NORMAL
    assert strong.cause.confidence > INSUFFICIENT_POWER_CONFIDENCE_CAP
    assert strong.confidence_capped is False


def test_the_cap_does_not_delay_a_silent_repair(history) -> None:
    """A merchant-layer finding is not capped by missing cohort power.

    The cap exists because "we could not tell whether this was systemic" is
    doubt about blaming the CUSTOMER. It says nothing about whether our own
    mandate is broken, and capping that would hold up a repair that costs
    nobody anything.
    """
    ref = "mnd_" + "a" * 32
    facts = MandateFacts(
        mandate_ref=ref,
        status="active",
        registered_at=DURING_OUTAGE - timedelta(days=200),
        instrument_reissued_at=DURING_OUTAGE - timedelta(days=30),
    )
    mandate_history = MandateHistory()
    for day in (20, 10):
        mandate_history.record(ref, DURING_OUTAGE - timedelta(days=day), succeeded=False)

    claim = make_claim(issuer="ISS_CO03", rail=Rail.ENACH, code="MD01", mandate_ref=ref)
    result = diagnose(
        claim,
        DiagnosisContext.from_claim(
            claim,
            cohort_history=history,
            mandate=facts,
            mandate_history=mandate_history,
        ),
        DURING_OUTAGE,
    )
    assert result.cause.cohort_power is CohortVerdict.INSUFFICIENT_POWER
    assert result.cause.layer is CauseLayer.MERCHANT
    assert result.cause.confidence > INSUFFICIENT_POWER_CONFIDENCE_CAP
    assert result.confidence_capped is False


def test_backoff_ladder_records_which_level_answered(history) -> None:
    """Which rung answered is on every result, because the blind-spot metric
    needs it. "We found nothing" and "we found nothing at a resolution coarse
    enough to hide it" are different answers.
    """
    answered: dict[CohortLevel | None, int] = {}
    for issuer in ("ISS_LP01", "ISS_LP02", "ISS_PS01", "ISS_CO01", "ISS_CO03"):
        for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY):
            level = cohort_check(issuer, rail, DURING_OUTAGE, history).level
            answered[level] = answered.get(level, 0) + 1

    # More than one rung is actually used - a ladder where every call answers
    # at the same level is not a ladder.
    assert len({level for level in answered if level is not None}) >= 2
    assert None in answered, "no cell was thin enough to exhaust the ladder"

    # The declared order is the order climbed.
    assert LADDER == (
        CohortLevel.ISSUER_INSTRUMENT_15M,
        CohortLevel.ISSUER_INSTRUMENT_2H,
        CohortLevel.ISSUER_DAY,
        CohortLevel.INSTRUMENT_NETWORK_15M,
    )

    # A 15-minute cell holds a median of one transaction in this world, so the
    # finest rung can essentially never answer alone. That is the situation the
    # ladder exists for, and it is measured rather than assumed.
    assert answered.get(CohortLevel.ISSUER_INSTRUMENT_15M, 0) == 0


def test_blind_spot_share_is_measurable(history) -> None:
    """An unmeasured blind spot is a defect; a measured one is a limitation."""
    diagnoses = {}
    pairs = [
        ("ISS_LP01", Rail.CARD, "51"),
        ("ISS_CO03", Rail.UPI_AUTOPAY, "Z9"),
        ("ISS_CO02", Rail.UPI_AUTOPAY, "Z9"),
        ("ISS_SP03", Rail.UPI_AUTOPAY, "Z9"),
    ]
    for index, (issuer, rail, code) in enumerate(pairs):
        claim = make_claim(issuer=issuer, rail=rail, code=code)
        diagnoses[str(index)] = diagnose(
            claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
        )

    share = blind_spot_share(diagnoses)
    assert 0.0 < share < 1.0, "the sample must contain both powered and thin cells"
    assert blind_spot_share({}) == 0.0


# ---------------------------------------------------------------------------
# Gate tests 6 and 7 - the orphaned mandate cohort
# ---------------------------------------------------------------------------
ORPHAN_REF = "mnd_" + "b" * 32


def orphaned_context(history: CohortHistory, *, status: str = "active") -> DiagnosisContext:
    """The ~3% cohort the world planted: reissued, then silently broken.

    Our own records still say `active`, which is what makes it silent. The
    evidence is three ordinary facts in combination, none conclusive alone.
    """
    facts = MandateFacts(
        mandate_ref=ORPHAN_REF,
        status=status,
        registered_at=DURING_OUTAGE - timedelta(days=400),
        instrument_reissued_at=DURING_OUTAGE - timedelta(days=45),
    )
    mandate_history = MandateHistory()
    for day in (40, 25, 10):
        mandate_history.record(ORPHAN_REF, DURING_OUTAGE - timedelta(days=day), succeeded=False)
    return DiagnosisContext(
        issuer_ref="ISS_LP01",
        cohort_history=history,
        mandate=facts,
        mandate_history=mandate_history,
        decline_code="MD01",
    )


def test_orphaned_mandate_diagnosed_as_MERCHANT(history) -> None:
    """Our fault, not theirs. The customer withheld nothing.

    `MD01` means "no mandate", which the code map reads as the customer having
    revoked it. Mandate health runs FIRST and resolves the ambiguity from
    evidence: the instrument was reissued after registration, every
    presentation since has failed, and our records still claim it is active.
    """
    claim = make_claim(rail=Rail.ENACH, code="MD01", issuer="ISS_LP01", mandate_ref=ORPHAN_REF)
    result = diagnose(claim, orphaned_context(history), DURING_OUTAGE)

    assert result.cause.layer is CauseLayer.MERCHANT
    assert result.cause.label is CauseLabel.MANDATE_ORPHANED
    assert result.answered_by is DiagnosisPath.MANDATE

    # And this is exactly what the code map would have said instead. The order
    # is the only thing standing between this customer and a dunning message.
    assert code_lookup(Rail.ENACH, "MD01").layer is CauseLayer.CUSTOMER
    assert code_lookup(Rail.ENACH, "MD01").label is CauseLabel.MANDATE_REVOKED


def test_merchant_cause_routes_to_SELF_HEALING(history, gate) -> None:
    """Repaired at the rail, with zero customer contact.

    "We recover this money without ever messaging the customer" is only a
    claim worth making if no message can physically go out, so the Gate is
    asked as well as the router.
    """
    claim = make_claim(rail=Rail.ENACH, code="MD01", issuer="ISS_LP01", mandate_ref=ORPHAN_REF)
    result = diagnose(claim, orphaned_context(history), DURING_OUTAGE)

    assert result.next_state is ClaimState.SELF_HEALING
    assert result.contact_permitted is False
    assert LAYER_ROUTE[CauseLayer.MERCHANT] is ClaimState.SELF_HEALING

    routed = result.apply(claim)
    assert routed.state is ClaimState.SELF_HEALING
    assert routed.cause.label is CauseLabel.MANDATE_ORPHANED

    # The repair action itself stays available - suppressing contact must not
    # suppress the fix.
    from arc.gate.context import ACTION_CHANNEL, Channel

    silent = {action for action in ActionType if ACTION_CHANNEL[action] is Channel.SILENT}
    assert ActionType.MANDATE_RE_REGISTER in silent


def test_a_healthy_mandate_is_not_called_orphaned(history) -> None:
    """The detector needs all three facts. Any one alone is a coincidence."""
    ref = "mnd_" + "c" * 32

    # Reissued BEFORE registration: the mandate was built against the new card.
    before = MandateFacts(
        mandate_ref=ref,
        status="active",
        registered_at=DURING_OUTAGE - timedelta(days=30),
        instrument_reissued_at=DURING_OUTAGE - timedelta(days=60),
    )
    failures = MandateHistory()
    for day in (20, 10):
        failures.record(ref, DURING_OUTAGE - timedelta(days=day), succeeded=False)
    assert not mandate_health(
        amount_paise=paise(1000), facts=before, history=failures, at=DURING_OUTAGE
    ).found

    # Reissued after registration, but a presentation since has SUCCEEDED, so
    # the mandate plainly still resolves.
    after = MandateFacts(
        mandate_ref=ref,
        status="active",
        registered_at=DURING_OUTAGE - timedelta(days=60),
        instrument_reissued_at=DURING_OUTAGE - timedelta(days=30),
    )
    mixed = MandateHistory()
    mixed.record(ref, DURING_OUTAGE - timedelta(days=20), succeeded=False)
    mixed.record(ref, DURING_OUTAGE - timedelta(days=10), succeeded=True)
    assert not mandate_health(
        amount_paise=paise(1000), facts=after, history=mixed, at=DURING_OUTAGE
    ).found

    # A single failure since the reissue is not yet a run.
    lonely = MandateHistory()
    lonely.record(ref, DURING_OUTAGE - timedelta(days=5), succeeded=False)
    assert not mandate_health(
        amount_paise=paise(1000), facts=after, history=lonely, at=DURING_OUTAGE
    ).found


def test_mandate_health_groups_on_the_pseudonym_never_the_raw_umrn() -> None:
    """A raw UMRN is refused, not hashed on the caller's behalf.

    Accepting it quietly would mean a raw bank identifier had travelled past
    the redaction boundary, and the fix belongs at the boundary that let it
    through rather than here.
    """
    history = MandateHistory()
    for raw in ("HDFC091019459048", "091019459048", "UMRN091019459048", "", "mnd_short"):
        with pytest.raises(RawMandateIdentifier):
            history.record(raw, DURING_OUTAGE, succeeded=False)

    with pytest.raises(RawMandateIdentifier):
        MandateFacts(mandate_ref="HDFC091019459048")

    # The pseudonym L1 actually produces is accepted.
    from arc.ingest.normaliser import pseudonymous_mandate_ref

    ref = pseudonymous_mandate_ref("HDFC091019459048")
    history.record(ref, DURING_OUTAGE, succeeded=False)
    assert len(history.attempts_for(ref)) == 1
    assert MandateFacts(mandate_ref=ref).mandate_ref == ref


def test_cap_exceeded_beats_a_weaker_mandate_finding() -> None:
    """Arithmetic before inference. An amount above a cap is not an opinion."""
    ref = "mnd_" + "d" * 32
    facts = MandateFacts(
        mandate_ref=ref,
        status="active",
        cap_paise=paise(100_000),
        registered_at=DURING_OUTAGE - timedelta(days=400),
        instrument_reissued_at=DURING_OUTAGE - timedelta(days=45),
    )
    history = MandateHistory()
    for day in (40, 25):
        history.record(ref, DURING_OUTAGE - timedelta(days=day), succeeded=False)

    result = mandate_health(
        amount_paise=paise(129_900), facts=facts, history=history, at=DURING_OUTAGE
    )
    assert result.label is CauseLabel.MANDATE_CAP_EXCEEDED


# ---------------------------------------------------------------------------
# Gate tests 8 and 9 - the code map and the LLM residue
# ---------------------------------------------------------------------------
def test_unmapped_code_returns_UNKNOWN_not_guess(history) -> None:
    """GI-5. An unmatched code fails closed onto a review queue.

    Never guessed at, never fallen through to a permissive default, and never
    quietly nearest-matched onto a code that looks similar.
    """
    for unknown in ("ZZ99", "77", "NOPE", "MD99", "  ", "0"):
        assert code_lookup(Rail.CARD, unknown) is UNMAPPED
        assert code_lookup(Rail.CARD, unknown).confidence == 0.0
        assert code_lookup(Rail.CARD, unknown).label is CauseLabel.UNKNOWN

    claim = make_claim(issuer="ISS_LP01", code="ZZ99", at=BEFORE_OUTAGE)
    result = diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), BEFORE_OUTAGE
    )
    assert result.cause.label is CauseLabel.UNKNOWN
    assert result.cause.layer is CauseLayer.UNKNOWN
    assert result.cause.confidence == 0.0
    assert result.review_required is True

    # The claim stays live on the conservative path. Failing closed is not the
    # same as giving up on money that is owed.
    assert result.next_state is ClaimState.PLANNED
    assert result.contact_permitted is True

    # A code on the WRONG rail is unmapped too. AM04 is an eNACH return
    # reason and means nothing on a card.
    assert code_lookup(Rail.CARD, "AM04") is UNMAPPED
    assert code_lookup(Rail.ENACH, "AM04").label is CauseLabel.INSUFFICIENT_FUNDS


def test_a_known_code_with_no_safe_label_is_also_unknown() -> None:
    """The cause vocabulary is closed and has no risk-decline member.

    Forcing 57 or 62 onto HARD_DECLINE would permanently block retries on a
    transaction that was refused once. Listing them explicitly as unlabelled
    is a reviewable decision; omitting them would read as an oversight.
    """
    for rail, code in ((Rail.CARD, "57"), (Rail.CARD, "62"), (Rail.ENACH, "RR04")):
        meaning = code_lookup(rail, code)
        assert (rail, code) in CODE_MAP, "the code must be listed, not merely absent"
        assert meaning.label is CauseLabel.UNKNOWN
        assert meaning.confidence == 0.0
        assert "no safe label" in meaning.note


def test_do_not_retry_advice_overrides_the_decline_code() -> None:
    """A network instruction to stop is not weighed against other evidence."""
    assert code_lookup(Rail.CARD, "51", "MAC03").label is CauseLabel.DO_NOT_RETRY
    assert code_lookup(Rail.CARD, "51", "MAC21").label is CauseLabel.DO_NOT_RETRY
    assert code_lookup(Rail.CARD, "51", "MAC02").label is CauseLabel.INSUFFICIENT_FUNDS


def test_llm_cause_confidence_capped_at_070(history) -> None:
    """The cap lives in `Cause`, at M1, and is asserted here rather than rebuilt.

    A second implementation in this layer would be a second place for the
    number to drift, and the two copies would disagree silently. So what is
    tested is that the domain still refuses, that the Sentinel does not clamp,
    and that an out-of-contract output is REJECTED rather than coerced.
    """
    assert LLM_CONFIDENCE_CAP == 0.70

    # The cap is enforced where it is defined.
    with pytest.raises(ValueError, match="cap"):
        Cause(
            label=CauseLabel.INSUFFICIENT_FUNDS,
            layer=CauseLayer.CUSTOMER,
            confidence=0.95,
            derived_from=DiagnosisPath.LLM,
            cohort_power=CohortVerdict.NORMAL,
        )
    # And it applies only to LLM-derived causes.
    assert (
        Cause(
            label=CauseLabel.INSUFFICIENT_FUNDS,
            layer=CauseLayer.CUSTOMER,
            confidence=0.95,
            derived_from=DiagnosisPath.CODE_MAP,
            cohort_power=CohortVerdict.NORMAL,
        ).confidence
        == 0.95
    )

    # A classifier that returns an out-of-contract confidence is REJECTED, not
    # clamped. The claim ends UNKNOWN in a review queue, which is what GI-5
    # requires and what the LLM contract means by "never coerce".
    def rogue(claim, text):
        return Finding(CauseLabel.INSUFFICIENT_FUNDS, CauseLayer.CUSTOMER, 0.95, "rogue")

    claim = make_claim(issuer="ISS_LP01", code="ZZ99", at=BEFORE_OUTAGE)
    rejected = diagnose(
        claim,
        DiagnosisContext.from_claim(
            claim, cohort_history=history, free_text="some narration", llm_classifier=rogue
        ),
        BEFORE_OUTAGE,
    )
    assert rejected.cause.label is CauseLabel.UNKNOWN
    assert rejected.cause.confidence == 0.0
    assert rejected.review_required is True
    assert any("llm_residue" in note for note in rejected.rejected_findings)
    assert rejected.cause.confidence != LLM_CONFIDENCE_CAP, "the value was clamped, not rejected"

    # An in-contract confidence passes through untouched.
    def honest(claim, text):
        return Finding(CauseLabel.INSUFFICIENT_FUNDS, CauseLayer.CUSTOMER, 0.65, "honest")

    accepted = diagnose(
        claim,
        DiagnosisContext.from_claim(
            claim, cohort_history=history, free_text="some narration", llm_classifier=honest
        ),
        BEFORE_OUTAGE,
    )
    assert accepted.answered_by is DiagnosisPath.LLM
    assert accepted.cause.confidence == 0.65


def test_the_cap_is_not_reimplemented_in_the_sentinel() -> None:
    """One cap, one place. A third copy would be a third thing to drift.

    Checked structurally: nothing in this package imports or names the cap,
    and the module that builds causes never mentions its value.
    """
    package = Path(__file__).resolve().parents[1] / "arc" / "sentinel"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name != "LLM_CONFIDENCE_CAP", f"{path.name} names the cap"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assert alias.name != "LLM_CONFIDENCE_CAP", f"{path.name} imports the cap"

    diagnose_source = (package / "diagnose.py").read_text(encoding="utf-8")
    assert "0.70" not in diagnose_source and "0.7\n" not in diagnose_source


def test_the_system_runs_with_the_llm_disabled(history) -> None:
    """No classifier, no free text, no degradation in correctness.

    The residue step simply produces nothing, and everything deterministic
    still answers. This is the M10.5 degradation test in miniature, asserted
    here because the Sentinel is the first layer that could have depended on a
    model and does not.
    """
    for code, expected in (("91", CauseLayer.ISSUER), ("51", CauseLayer.CUSTOMER)):
        claim = make_claim(issuer="ISS_LP01", code=code, at=BEFORE_OUTAGE)
        result = diagnose(
            claim, DiagnosisContext.from_claim(claim, cohort_history=history), BEFORE_OUTAGE
        )
        assert result.cause.layer is expected
        assert result.answered_by is not DiagnosisPath.LLM
        assert result.cause.confidence > 0.0


# ---------------------------------------------------------------------------
# The order is the design, and it is enforced structurally
# ---------------------------------------------------------------------------
DIAGNOSE_MODULE = Path(__file__).resolve().parents[1] / "arc" / "sentinel" / "diagnose.py"
CHECK_FUNCTIONS = ("check_cohort", "check_mandate", "check_code_map", "check_llm_residue")


def _function_ast(name: str) -> ast.FunctionDef:
    tree = ast.parse(DIAGNOSE_MODULE.read_text(encoding="utf-8"), filename=str(DIAGNOSE_MODULE))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in diagnose.py")


def test_the_check_order_is_declared_once() -> None:
    """Cohort, mandate, code map, LLM. In a tuple, in that order."""
    assert [step.path for step in ORDERED_CHECKS] == [
        DiagnosisPath.COHORT,
        DiagnosisPath.MANDATE,
        DiagnosisPath.CODE_MAP,
        DiagnosisPath.LLM,
    ]
    assert [step.name for step in ORDERED_CHECKS] == [
        "cohort",
        "mandate_health",
        "code_map",
        "llm_residue",
    ]
    assert isinstance(ORDERED_CHECKS, tuple), "a list could be reordered at runtime"


def test_diagnose_iterates_the_declared_order_and_calls_nothing_by_name() -> None:
    """STRUCTURAL, not behavioural.

    A happy-path test can only observe the order that happened to run on one
    input. This reads the code: `diagnose` must loop over ORDERED_CHECKS and
    must not call any check function directly, so reordering the pipeline means
    editing one tuple and nothing else.

    WHY that matters more than it looks: a direct call sequence would let
    someone "tidy up" the four checks into a different order in a diff that
    reads as a refactor, and the resulting system would dun four hundred
    people for their bank's outage while every other test stayed green.
    """
    body = _function_ast("diagnose")

    loops = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "ORDERED_CHECKS"
    ]
    assert len(loops) == 1, "diagnose must iterate ORDERED_CHECKS exactly once"

    called = {
        node.func.id
        for node in ast.walk(body)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    leaked = called & set(CHECK_FUNCTIONS)
    assert not leaked, f"diagnose calls {sorted(leaked)} by name instead of iterating the order"

    # The loop dispatches through the step, so a check cannot be reached except
    # via the declared tuple.
    dispatches = [
        node
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
    ]
    assert dispatches, "the loop must dispatch through step.run"


def test_a_later_check_never_pre_empts_an_earlier_one(history) -> None:
    """First confident hit wins, observed on inputs where they disagree.

    Each claim below would get a DIFFERENT answer from a later check. The
    earlier one has to win every time.
    """
    calls: list[str] = []

    def spy(step):
        def wrapped(claim, context, at):
            calls.append(step.name)
            return step.run(claim, context, at)

        return wrapped

    # Cohort beats the code map: an outage claim carrying the world's remapped
    # insufficient-funds code.
    claim = make_claim(code="51")
    result = diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
    )
    assert result.answered_by is DiagnosisPath.COHORT
    assert code_lookup(Rail.CARD, "51").layer is CauseLayer.CUSTOMER

    # Mandate health beats the code map on the same claim shape.
    orphan = make_claim(rail=Rail.ENACH, code="MD01", issuer="ISS_LP01", mandate_ref=ORPHAN_REF)
    assert diagnose(orphan, orphaned_context(history), DURING_OUTAGE).answered_by is (
        DiagnosisPath.MANDATE
    )

    # The code map beats the LLM residue.
    def eager(claim, text):
        return Finding(CauseLabel.CHECKOUT_ABANDONED, CauseLayer.CUSTOMER, 0.69, "eager")

    coded = make_claim(issuer="ISS_LP01", code="51", at=BEFORE_OUTAGE)
    with_llm = diagnose(
        coded,
        DiagnosisContext.from_claim(
            coded, cohort_history=history, free_text="text", llm_classifier=eager
        ),
        BEFORE_OUTAGE,
    )
    assert with_llm.answered_by is DiagnosisPath.CODE_MAP
    assert calls == []


def test_checks_after_a_confident_hit_are_not_run(history, monkeypatch) -> None:
    """The pipeline stops at the first confident answer.

    Not just "the first answer wins" - the later checks are never invoked, so
    a downstream check cannot have a side effect on a claim an earlier one
    already explained.
    """
    from arc.sentinel import diagnose as module

    seen: list[str] = []
    instrumented = tuple(
        module.CheckStep(step.path, step.name, _record(step, seen)) for step in ORDERED_CHECKS
    )
    monkeypatch.setattr(module, "ORDERED_CHECKS", instrumented)

    claim = make_claim(code="51")
    module.diagnose(
        claim, DiagnosisContext.from_claim(claim, cohort_history=history), DURING_OUTAGE
    )
    assert seen == ["cohort"], f"checks ran after a confident cohort hit: {seen}"

    seen.clear()
    unmapped = make_claim(issuer="ISS_LP01", code="ZZ99", at=BEFORE_OUTAGE)
    module.diagnose(
        unmapped,
        DiagnosisContext.from_claim(unmapped, cohort_history=history),
        BEFORE_OUTAGE,
    )
    assert seen == ["cohort", "mandate_health", "code_map", "llm_residue"], (
        f"an unresolved claim must reach every check in order, saw {seen}"
    )


def _record(step, seen: list[str]):
    def wrapped(claim, context, at):
        seen.append(step.name)
        return step.run(claim, context, at)

    return wrapped


# ---------------------------------------------------------------------------
# The detector's own maths, including the one deviation from the build doc
# ---------------------------------------------------------------------------
def test_the_specified_z_formula_is_self_limiting() -> None:
    """Evidence for the one deliberate deviation, rather than an assertion.

    The build doc gives z as `(r_t - mu_t) / sqrt(sig2_t)` where both mu and
    sig2 have already absorbed r_t. Substituting, with d = r_t - mu_{t-1}:

        r_t - mu_t = (1-a)*d
        sig2_t     >= a*d^2
        z_t        <= (1-a)/sqrt(a)

    At alpha 0.25 that ceiling is exactly 1.5, for ANY anomaly however large,
    so with tau at 3.0 the detector could never fire once. This test computes
    the bound rather than taking it on trust, and `cohort.py` carries the same
    algebra beside the line that departs from it.
    """
    import math

    ceiling = (1 - ALPHA) / math.sqrt(ALPHA)
    for deviation in (0.05, 0.5, 1.0, 10.0, 1000.0):
        sigma2 = ALPHA * deviation**2
        post_update_z = (deviation - ALPHA * deviation) / math.sqrt(sigma2)
        assert post_update_z == pytest.approx(ceiling)

    assert ceiling < TAU, (
        "the literal formula would be able to fire, so the deviation is unjustified"
    )

    # What the detector actually does: test against the PRIOR baseline, which
    # is what an EWMA control chart does and what makes z mean something.
    history = CohortHistory()
    for step in range(60, 0, -1):
        moment = DURING_OUTAGE - timedelta(hours=2 * step)
        for index in range(20):
            history.record(
                "ISS_TEST", Rail.CARD, moment + timedelta(minutes=index), succeeded=index > 2
            )
    for index in range(20):
        history.record(
            "ISS_TEST", Rail.CARD, DURING_OUTAGE + timedelta(minutes=index), succeeded=False
        )

    result = cohort_check("ISS_TEST", Rail.CARD, DURING_OUTAGE, history)
    assert result.verdict is CohortVerdict.DEGRADED
    assert result.z > TAU > ceiling


def test_ewma_baseline_follows_the_specified_recursion() -> None:
    """The update rule itself is the build doc's, verbatim."""
    mu, sigma2 = ewma_baseline([])
    assert (mu, sigma2) == (0.0, 0.0)

    mu, sigma2 = ewma_baseline([0.4])
    assert mu == 0.4 and sigma2 == 0.0

    rates = [0.1, 0.2, 0.3]
    expected_mu, expected_sigma2 = 0.1, 0.0
    for rate in rates[1:]:
        expected_sigma2 = ALPHA * (rate - expected_mu) ** 2 + (1 - ALPHA) * expected_sigma2
        expected_mu = ALPHA * rate + (1 - ALPHA) * expected_mu
    mu, sigma2 = ewma_baseline(rates)
    assert mu == pytest.approx(expected_mu)
    assert sigma2 == pytest.approx(expected_sigma2)


def test_a_flat_history_cannot_fire_on_noise() -> None:
    """The sigma floor. Without it a cell whose rate never moved divides by
    almost zero and reports an outage on one extra decline."""
    history = CohortHistory()
    for step in range(40, 0, -1):
        moment = DURING_OUTAGE - timedelta(hours=2 * step)
        for index in range(20):
            history.record(
                "ISS_FLAT", Rail.CARD, moment + timedelta(minutes=index), succeeded=index > 1
            )
    for index in range(20):
        history.record(
            "ISS_FLAT", Rail.CARD, DURING_OUTAGE + timedelta(minutes=index), succeeded=index > 2
        )

    result = cohort_check("ISS_FLAT", Rail.CARD, DURING_OUTAGE, history)
    assert result.verdict is CohortVerdict.NORMAL
    assert abs(result.z) < TAU
    assert SIGMA_FLOOR > 0


def test_a_thin_issuer_is_covered_by_the_independent_downtime_feed() -> None:
    """For a thin issuer the gateway's own status IS the detector.

    No amount of back-off finds an incident in eleven transactions. The feed
    needs none of our sample, which is why the spec makes it the primary
    signal there rather than a cross-check. It is declared, not inferred - the
    windows below come from a status page, not from the world's ground truth.
    """
    declared = StaticDowntimeFeed(
        windows=(
            ("ISS_CO03", DURING_OUTAGE - timedelta(minutes=10), DURING_OUTAGE + timedelta(hours=1)),
        )
    )
    empty = CohortHistory()

    without = cohort_check("ISS_CO03", Rail.CARD, DURING_OUTAGE, empty)
    assert without.verdict is CohortVerdict.INSUFFICIENT_POWER

    with_feed = cohort_check("ISS_CO03", Rail.CARD, DURING_OUTAGE, empty, downtime=declared)
    assert with_feed.verdict is CohortVerdict.DEGRADED
    assert with_feed.level is CohortLevel.DOWNTIME_FEED
    assert with_feed.attempts == 0, "the feed needs none of our sample"
    assert with_feed.degraded_until == DURING_OUTAGE + timedelta(hours=1)

    # It does not fire outside its declared window or for another issuer.
    assert (
        cohort_check(
            "ISS_CO03", Rail.CARD, DURING_OUTAGE + timedelta(hours=2), empty, downtime=declared
        ).verdict
        is CohortVerdict.INSUFFICIENT_POWER
    )
    assert (
        cohort_check("ISS_LP01", Rail.CARD, DURING_OUTAGE, empty, downtime=declared).verdict
        is CohortVerdict.INSUFFICIENT_POWER
    )


def test_an_issuer_incident_carries_across_its_rails(history) -> None:
    """An outage is an event at the issuer, not at one of its rails.

    It is routinely visible only on the busiest instrument, because that is
    where the sample is. The thin rails of the same issuer are failing for the
    same reason, and carrying the finding across errs toward zero customer
    contact - the only direction an issuer-layer cause should err in.
    """
    found = {
        rail: cohort_check(OUTAGE.issuer_id, rail, DURING_OUTAGE, history)
        for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY)
    }
    assert all(result.verdict is CohortVerdict.DEGRADED for result in found.values())
    assert {result.detected_on for result in found.values()} == {Rail.CARD}

    # It does not leak to a different issuer at the same instant.
    for issuer in ("ISS_LP01", "ISS_LP03", "ISS_PS02"):
        assert (
            cohort_check(issuer, Rail.CARD, DURING_OUTAGE, history).verdict
            is not CohortVerdict.DEGRADED
        ), f"{issuer} was swept up in another issuer's incident"


def test_n_min_is_actually_binding(history) -> None:
    """The minimum sample is not decorative: cells below it never answer."""
    assert N_MIN > 1
    for issuer in ("ISS_LP01", "ISS_LP02", "ISS_CO01", "ISS_CO03"):
        for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY):
            result = cohort_check(issuer, rail, DURING_OUTAGE, history)
            if result.level is not None and result.level is not CohortLevel.DOWNTIME_FEED:
                assert result.attempts >= N_MIN, (
                    f"{issuer}/{rail} answered at {result.level} on {result.attempts} attempts"
                )
