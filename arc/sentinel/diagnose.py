"""The four checks, in the one order that is defensible.

    1. COHORT         is this systemic?              -> ISSUER
    2. MANDATE HEALTH is our own setup broken?       -> MERCHANT
    3. CODE MAP       deterministic lookup           -> CUSTOMER
    4. LLM RESIDUE    free text only, capped         -> any

First confident hit wins.

THE ORDER IS THE DESIGN. Running the cheap code map first would attribute a
four-hundred-account issuer outage to four hundred delinquent customers and dun
every one of them. The expensive systemic check has to come first precisely
because it is the one that prevents mass harm - and the ordering is worth more
than any single check's accuracy, because the failure it prevents is not a
wrong label on one claim but the same wrong label on a whole cohort.

The order is enforced structurally rather than by convention: `ORDERED_CHECKS`
is a tuple, `diagnose` iterates it, and no check is ever called by name from
the body. `tests/test_sentinel.py` walks this module's AST and fails the build
if that stops being true, because a reordering that looked like a tidy-up would
otherwise be invisible in review.

ON THE LLM CONFIDENCE CAP: it is NOT enforced here. `Cause.__post_init__` at M1
refuses an LLM-derived cause above the cap, and a database CHECK refuses the
row. A third copy in this file would be a third place for the number to drift.
What this module does is let the refusal happen and treat it as what it is - a
finding that failed validation, which under GI-5 falls through to UNKNOWN and a
review queue rather than being coerced into range.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import MappingProxyType

from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.core.types import (
    Cause,
    CauseLabel,
    CauseLayer,
    Claim,
    ClaimState,
    CohortVerdict,
    DiagnosisPath,
    transition,
)
from arc.sentinel.code_map import claim_type_fallback, code_lookup
from arc.sentinel.cohort import (
    CohortHistory,
    CohortLevel,
    CohortResult,
    DowntimeFeed,
    cohort_check,
)
from arc.sentinel.mandate_health import MandateFacts, MandateHistory, mandate_health

# When the cohort has no power, a customer-layer attribution is capped here.
# Under the Gate's 0.80 threshold for money-moving actions, so a claim
# diagnosed without cohort power spends its first cycle on conservative
# actions instead of presenting a debit on the strength of a guess.
INSUFFICIENT_POWER_CONFIDENCE_CAP = 0.75

# Where a layer sends the claim. The layer decides whether a human is ever
# contacted, which is why it matters more than the label.
LAYER_ROUTE: Mapping[CauseLayer, ClaimState] = MappingProxyType(
    {
        # Systemic. Freeze everything and requeue when the outage clears.
        CauseLayer.ISSUER: ClaimState.SUPPRESSED,
        # Our fault. Repair at the rail; the customer withheld nothing.
        CauseLayer.MERCHANT: ClaimState.SELF_HEALING,
        CauseLayer.CUSTOMER: ClaimState.PLANNED,
        # Unmatched fails closed onto the conservative path, not onto silence:
        # a claim nobody could diagnose is still money owed.
        CauseLayer.UNKNOWN: ClaimState.PLANNED,
    }
)

# Layers where contacting the customer would be wrong rather than merely
# unproductive. An issuer incident is not their fault and a broken mandate is
# ours, so in both cases a message asks them to fix something they did not do.
NO_CONTACT_LAYERS: frozenset[CauseLayer] = frozenset({CauseLayer.ISSUER, CauseLayer.MERCHANT})


@dataclass(frozen=True)
class Finding:
    """One check's answer. `confidence == 0` means it did not answer."""

    label: CauseLabel
    layer: CauseLayer
    confidence: float
    note: str = ""
    suppress_until: datetime | None = None

    @property
    def is_confident(self) -> bool:
        return self.confidence > 0.0


NO_FINDING = Finding(CauseLabel.UNKNOWN, CauseLayer.UNKNOWN, 0.0)

LlmClassifier = Callable[[Claim, str | None], Finding | None]


@dataclass(frozen=True)
class DiagnosisContext:
    """Everything the four checks may know. Passed in, never fetched.

    The Sentinel performs no I/O, for the same reason the Gate does not: a
    diagnosis that read a database cannot be reproduced under replay, and a
    cause that cannot be reproduced cannot be audited.
    """

    issuer_ref: str | None = None
    cohort_history: CohortHistory = field(default_factory=CohortHistory)
    downtime: DowntimeFeed | None = None

    mandate: MandateFacts = field(default_factory=MandateFacts)
    mandate_history: MandateHistory = field(default_factory=MandateHistory)

    decline_code: str | None = None
    advice_code: str | None = None

    # Redacted free text for the residue step, and the classifier that reads
    # it. Both absent by default: the system is fully functional with the LLM
    # disabled, degrading in quality but never in correctness.
    free_text: str | None = None
    llm_classifier: LlmClassifier | None = None

    @classmethod
    def from_claim(cls, claim: Claim, **overrides: object) -> DiagnosisContext:
        """Read what L1 already put in `evidence_structured`.

        Everything here is closed-vocabulary and pseudonymous, which is why
        the Sentinel can work from the claim alone without reopening the
        subject store.
        """
        evidence = claim.evidence_structured
        base: dict[str, object] = {
            "issuer_ref": evidence.get("issuer_ref"),
            "decline_code": evidence.get("decline_code"),
            "advice_code": evidence.get("advice_code"),
        }
        base.update(overrides)
        return cls(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The four checks. Each takes the same three arguments and returns a Finding.
# ---------------------------------------------------------------------------
def check_cohort(claim: Claim, context: DiagnosisContext, at: datetime) -> Finding:
    """Systemic first. The one check that prevents harm at cohort scale."""
    result = _cohort_result(claim, context, at)
    if result.verdict is not CohortVerdict.DEGRADED:
        return NO_FINDING
    return Finding(
        label=CauseLabel.ISSUER_OUTAGE,
        layer=CauseLayer.ISSUER,
        confidence=0.93,
        note=(
            f"decline rate {result.rate:.0%} against a baseline of "
            f"{result.baseline:.0%} at {result.level}, z={result.z:.1f}, "
            f"n={result.attempts}"
        ),
        suppress_until=result.degraded_until,
    )


def check_mandate(claim: Claim, context: DiagnosisContext, at: datetime) -> Finding:
    """Our own setup, before anyone else's behaviour."""
    result = mandate_health(
        amount_paise=claim.amount_paise,
        facts=context.mandate,
        history=context.mandate_history,
        at=at,
    )
    if not result.found or result.label is None:
        return NO_FINDING
    return Finding(
        label=result.label,
        layer=CauseLayer.MERCHANT,
        confidence=result.confidence,
        note=result.reason,
    )


def check_code_map(claim: Claim, context: DiagnosisContext, at: datetime) -> Finding:
    """The cheap deterministic lookup, third. Unmatched fails closed."""
    meaning = code_lookup(claim.rail, context.decline_code, context.advice_code)
    if not meaning.is_confident and context.decline_code is None:
        meaning = claim_type_fallback(claim.claim_type)
    if not meaning.is_confident:
        return NO_FINDING
    return Finding(
        label=meaning.label,
        layer=meaning.layer,
        confidence=meaning.confidence,
        note=meaning.note or f"decline code {context.decline_code!r} on {claim.rail}",
    )


def check_llm_residue(claim: Claim, context: DiagnosisContext, at: datetime) -> Finding:
    """Free text only, and only when everything deterministic has passed.

    Absent a classifier there is simply no finding, which is what makes a full
    run at LLM_ENABLED=false degrade in quality without degrading in
    correctness.
    """
    if context.llm_classifier is None or context.free_text is None:
        return NO_FINDING
    finding = context.llm_classifier(claim, context.free_text)
    return finding if finding is not None else NO_FINDING


@dataclass(frozen=True)
class CheckStep:
    """One rung of the ordered pipeline, with the path it records."""

    path: DiagnosisPath
    name: str
    run: Callable[[Claim, DiagnosisContext, datetime], Finding]


# THE ORDER. Cohort, mandate, code map, LLM residue. `diagnose` iterates this
# tuple and calls nothing by name, so reordering the pipeline means editing
# this one line and nothing else - and the AST test notices when it changes.
ORDERED_CHECKS: tuple[CheckStep, ...] = (
    CheckStep(DiagnosisPath.COHORT, "cohort", check_cohort),
    CheckStep(DiagnosisPath.MANDATE, "mandate_health", check_mandate),
    CheckStep(DiagnosisPath.CODE_MAP, "code_map", check_code_map),
    CheckStep(DiagnosisPath.LLM, "llm_residue", check_llm_residue),
)


def _cohort_result(claim: Claim, context: DiagnosisContext, at: datetime) -> CohortResult:
    return cohort_check(
        context.issuer_ref,
        claim.rail,
        at,
        context.cohort_history,
        downtime=context.downtime,
    )


@dataclass(frozen=True)
class Diagnosis:
    """The cause, and what it means for the claim.

    Routing is here rather than left to the caller because the layer decides
    it: an issuer-layer cause has exactly one correct response and letting
    each call site choose would be four chances to get it wrong.
    """

    cause: Cause
    next_state: ClaimState
    contact_permitted: bool
    review_required: bool
    answered_by: DiagnosisPath
    cohort: CohortResult
    cohort_level: CohortLevel | None = None
    suppress_until: datetime | None = None
    confidence_capped: bool = False
    rejected_findings: tuple[str, ...] = ()
    note: str = ""

    def apply(self, claim: Claim) -> Claim:
        """Move the claim to DIAGNOSED, then to where the layer sends it."""
        diagnosed = transition(replace(claim, cause=self.cause), ClaimState.DIAGNOSED)
        return transition(diagnosed, self.next_state)


def diagnose(claim: Claim, context: DiagnosisContext, at: datetime) -> Diagnosis:
    """Run the four checks in order. First confident hit wins.

    A check that produces a cause the domain refuses is not a hit: the finding
    is recorded as rejected and the pipeline continues. That is how an LLM
    output above the confidence cap is handled - the refusal comes from
    `Cause.__post_init__`, which is the one place the cap lives, and this
    function neither knows the number nor re-applies it.
    """
    ensure_utc(at)
    cohort = _cohort_result(claim, context, at)
    rejected: list[str] = []

    for step in ORDERED_CHECKS:
        finding = step.run(claim, context, at)
        if not finding.is_confident:
            continue

        confidence, capped = _apply_power_cap(finding, cohort)
        try:
            cause = Cause(
                label=finding.label,
                layer=finding.layer,
                confidence=confidence,
                derived_from=step.path,
                cohort_power=cohort.verdict,
            )
        except (ValueError, TypeError) as refusal:
            # The domain refused this cause. Under GI-5 that is a validation
            # failure, not something to coerce into range, so the pipeline
            # keeps going and the rejection is on the record.
            rejected.append(f"{step.name}: {refusal}")
            continue

        return _route(cause, finding, cohort, step, capped, tuple(rejected))

    return _unresolved(cohort, tuple(rejected))


def _apply_power_cap(finding: Finding, cohort: CohortResult) -> tuple[float, bool]:
    """Cap customer-layer confidence when the cohort had no power.

    WHY only the customer layer: the cap exists because "we could not tell
    whether this was systemic" is precisely the doubt about blaming the
    customer. It says nothing about whether our own mandate is broken, and
    capping a merchant-layer finding would delay a silent repair for no reason.
    """
    if cohort.verdict is not CohortVerdict.INSUFFICIENT_POWER:
        return finding.confidence, False
    if finding.layer is not CauseLayer.CUSTOMER:
        return finding.confidence, False
    if finding.confidence <= INSUFFICIENT_POWER_CONFIDENCE_CAP:
        return finding.confidence, False
    return INSUFFICIENT_POWER_CONFIDENCE_CAP, True


def _route(
    cause: Cause,
    finding: Finding,
    cohort: CohortResult,
    step: CheckStep,
    capped: bool,
    rejected: tuple[str, ...],
) -> Diagnosis:
    return Diagnosis(
        cause=cause,
        next_state=LAYER_ROUTE[cause.layer],
        contact_permitted=cause.layer not in NO_CONTACT_LAYERS,
        review_required=cause.label is CauseLabel.UNKNOWN,
        answered_by=step.path,
        cohort=cohort,
        cohort_level=cohort.level,
        suppress_until=finding.suppress_until,
        confidence_capped=capped,
        rejected_findings=rejected,
        note=finding.note,
    )


def _unresolved(cohort: CohortResult, rejected: tuple[str, ...]) -> Diagnosis:
    """Nothing answered. UNKNOWN at zero confidence, and a review queue.

    Not a guess and not silence. The claim stays live on the conservative path
    - the Gate blocks money-moving actions below its confidence threshold - and
    a human is asked to look at it.
    """
    cause = Cause(
        label=CauseLabel.UNKNOWN,
        layer=CauseLayer.UNKNOWN,
        confidence=0.0,
        derived_from=DiagnosisPath.CODE_MAP,
        cohort_power=cohort.verdict,
    )
    return Diagnosis(
        cause=cause,
        next_state=LAYER_ROUTE[CauseLayer.UNKNOWN],
        contact_permitted=True,
        review_required=True,
        answered_by=DiagnosisPath.CODE_MAP,
        cohort=cohort,
        cohort_level=cohort.level,
        confidence_capped=False,
        rejected_findings=rejected,
        note="no check produced a confident cause; conservative path and review queue",
    )


def blind_spot_share(diagnoses: Mapping[str, Diagnosis]) -> float:
    """Share of claims diagnosed without cohort power, for CB-COHORT-BLIND.

    An unmeasured blind spot is a defect; a measured one is a known
    limitation. This is what makes it the second thing.
    """
    if not diagnoses:
        return 0.0
    blind = sum(
        1
        for diagnosis in diagnoses.values()
        if diagnosis.cause.cohort_power is CohortVerdict.INSUFFICIENT_POWER
    )
    return blind / len(diagnoses)


__all__ = [
    "INSUFFICIENT_POWER_CONFIDENCE_CAP",
    "LAYER_ROUTE",
    "NO_CONTACT_LAYERS",
    "ORDERED_CHECKS",
    "CheckStep",
    "Diagnosis",
    "DiagnosisContext",
    "Finding",
    "Paise",
    "blind_spot_share",
    "diagnose",
]
