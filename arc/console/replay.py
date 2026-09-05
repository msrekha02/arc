"""The decision trace, written as prose a reviewer can read straight through.

WHAT A TRACE HAS TO ANSWER, IN THE ORDER SOMEBODY ASKS IT:

    what was wrong, and how sure were we
    what could we have done, and what was each worth
    what was scarce at that moment
    what did the rules say, and were they law or our own judgement
    what did we actually do, and how likely was that
    what happened

A JSON dump contains all of that and answers none of it, because the reader has
to reconstruct the reasoning from field names. Every number below is placed in
a sentence that says what it meant, and the sentences are ordered so that the
next one is the one you were about to ask for.

THE PROPENSITY SENTENCE IS THE ONE THAT MATTERS. "We sampled voice call with
probability 0.31" is the difference between a decision that can be evaluated
off-policy and one that cannot, and stating it in words is what makes a
reviewer notice that it is there at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from arc.console.screens import ReplayView, Stage
from arc.core.money import Paise, format_inr
from arc.core.types import ActionType, CauseLayer
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry
from arc.proving_ground.composed import ADMISSION_RULE_ID

_LAYER_MEANING: Mapping[CauseLayer, str] = {
    CauseLayer.ISSUER: ("the issuer's, not the customer's, so nobody should be contacted about it"),
    CauseLayer.MERCHANT: ("our own setup, so it is repaired at the rail with no customer contact"),
    CauseLayer.CUSTOMER: "customer-side",
    CauseLayer.UNKNOWN: "unattributed, so the conservative path applies",
}


@dataclass(frozen=True)
class RuleFiring:
    rule_id: str
    verdict: Verdict
    detail: str = ""


@dataclass(frozen=True)
class ConsideredAction:
    action: ActionType
    uplift: float
    adjusted_value: float
    propensity: float
    eligible: bool = True


@dataclass(frozen=True)
class Trace:
    """Everything one decision needs to explain itself."""

    claim_id: str
    subject_token: str
    at: datetime
    amount_paise: Paise
    ltv_paise: Paise

    cause_label: str
    cause_layer: CauseLayer
    confidence: float
    answered_by: str
    cohort_power: str
    confidence_capped: bool

    considered: Sequence[ConsideredAction]
    shadow_prices: Mapping[str, float]

    firings: Sequence[RuleFiring]
    verdict: Verdict

    sampled_action: ActionType
    sampled_propensity: float
    realized_action: ActionType
    realized_propensity: float
    veto_occurred: bool

    outcome: str
    recovered_paise: Paise
    registry: RuleRegistry


def narrate(trace: Trace) -> ReplayView:
    """The trace, as paragraphs. Nothing here emits a field name."""
    return ReplayView(
        paragraphs=tuple(_paragraphs(trace)),
        stages=tuple(_stages(trace)),
        claim_id=trace.claim_id,
        subject_token=trace.subject_token,
    )


def _paragraphs(t: Trace) -> list[str]:
    out: list[str] = []

    out.append(
        f"On {t.at.strftime('%d %B %Y at %H:%M UTC')} a claim for "
        f"{format_inr(t.amount_paise)} was considered for subject {t.subject_token}. "
        f"The relationship behind it is worth about {format_inr(t.ltv_paise)}, which is "
        f"what the objective weighs the recovery against rather than the failed amount "
        f"alone."
    )

    capped = (
        " Confidence was capped because the cohort check could not find enough sample "
        "to be sure, which is recorded as a known blind spot rather than passed off as "
        "a clean read."
        if t.confidence_capped
        else ""
    )
    out.append(
        f"The Sentinel attributed the failure to {t.cause_label.replace('_', ' ')}, "
        f"answered by the {t.answered_by.replace('_', ' ')} check at "
        f"{t.confidence:.0%} confidence. The layer is {t.cause_layer.value}, which "
        f"means the fault is {_LAYER_MEANING[t.cause_layer]}. Cohort power at the time "
        f"was {t.cohort_power.replace('_', ' ')}.{capped}"
    )

    if t.considered:
        ranked = sorted(t.considered, key=lambda c: -c.adjusted_value)[:5]
        phrases = [
            f"{c.action.value.replace('_', ' ')} at an estimated "
            f"{c.uplift:+.1%} effect, worth "
            f"{format_inr(Paise(int(c.adjusted_value)))} after prices"
            for c in ranked
        ]
        out.append(
            f"{len(t.considered)} actions survived the Gate's eligibility projection "
            f"and were scored. The strongest were: {'; '.join(phrases)}. Each figure is "
            f"a signed estimate of incremental effect, so an action that would make "
            f"things worse scores negative and can never outrank declining to act."
        )

    binding = {k: v for k, v in t.shadow_prices.items() if v > 0}
    if binding:
        parts = [
            f"{k} at {format_inr(Paise(int(v)))}"
            for k, v in sorted(binding.items(), key=lambda kv: -kv[1])
        ]
        slack = sorted(k for k, v in t.shadow_prices.items() if v <= 0)
        tail = (
            f" {', '.join(slack)} priced at zero, meaning those budgets had slack and "
            f"cost this decision nothing."
            if slack
            else ""
        )
        out.append(
            f"At that moment the binding budgets were priced at {'; '.join(parts)}. "
            f"A shadow price is what the marginal unit of that budget was worth in "
            f"foregone recovery elsewhere, so an action lost here lost on economics "
            f"rather than on a threshold somebody chose.{tail}"
        )

    if t.firings:
        out.append(
            f"The Gate evaluated all {len(t.registry)} rules and returned "
            f"{t.verdict.value}. What spoke: {'; '.join(_described(t))}. Every rule "
            f"was evaluated, not just the first to object, because the audit trail "
            f"needs the whole verdict list."
        )
    elif t.verdict is Verdict.ALLOW:
        out.append(
            f"The Gate evaluated all {len(t.registry)} rules and returned "
            f"{t.verdict.value}. Nothing objected."
        )
    else:
        # A refusal with nobody named is a broken trace, and saying "nothing
        # objected" under a refusing verdict is worse than saying nothing: it
        # reads as a clean pass. Report the gap as a gap.
        out.append(
            f"The Gate evaluated all {len(t.registry)} rules and returned "
            f"{t.verdict.value}, but this trace carries no rule id for the refusal. "
            f"That is a defect in the record rather than a decision anybody made, "
            f"and it is shown rather than smoothed over."
        )

    if t.veto_occurred:
        out.append(
            f"The allocator sampled {t.sampled_action.value.replace('_', ' ')} with "
            f"probability {t.sampled_propensity:.3f}, and the Gate refused it. The "
            f"refused probability was not discarded: it collapsed onto declining to "
            f"act, which is what actually happened, at a composed probability of "
            f"{t.realized_propensity:.3f}. Dropping it would have removed a "
            f"non-random slice of the sample and quietly biased the headline."
        )
    else:
        out.append(
            f"The allocator sampled {t.sampled_action.value.replace('_', ' ')} with "
            f"probability {t.sampled_propensity:.3f} and the Gate certified it. That "
            f"probability is recorded because the policy drew with it, not estimated "
            f"afterwards, which is what makes this decision measurable off-policy at "
            f"all."
        )

    recovered = (
        f"{format_inr(t.recovered_paise)} arrived"
        if int(t.recovered_paise) > 0
        else "no money arrived"
    )
    out.append(
        f"The outcome was {t.outcome.replace('_', ' ')} and {recovered}. That result "
        f"is attributed to this decision under the last-touch rule declared once for "
        f"the whole system, which is a convention and is stated as one."
    )
    return out


def _gate_refusers(t: Trace) -> list[str]:
    """Firings that are actual registry rules, not the admission step."""
    return [f.rule_id for f in t.firings if f.rule_id != ADMISSION_RULE_ID]


def _outside_registry(t: Trace) -> bool:
    """True when something refused that is not one of the compliance rules.

    The screen used to say "rules evaluated 33" and then name ALLOC-ADMISSION,
    which is not among the 33. Naming a refuser the count does not include
    invites the reader to assume it is rule 34.
    """
    return any(f.rule_id == ADMISSION_RULE_ID for f in t.firings)


def _refuser_label(t: Trace, rule_id: str) -> str:
    """How a refuser describes its own force.

    A GATE RULE AND A BUDGET ARE NOT THE SAME REFUSAL. Registry rules carry
    M3's basis and status and are rendered through `force_label` so nothing
    overstates its legal force. `ALLOC-ADMISSION` is not in the registry at
    all - it is the allocator's in-cycle admission step - and calling it a
    compliance rule would overstate exactly what M3's wording exists to keep
    honest. It is labelled as what it is: a budget decision.
    """
    if rule_id == ADMISSION_RULE_ID:
        return "the allocator's admission step, a budget limit and not a compliance rule"
    try:
        return t.registry[rule_id].force_label()
    except KeyError:
        return "unknown to the rule registry"


def _described(t: Trace) -> list[str]:
    return [
        f"{f.rule_id} returned {f.verdict.value} ({_refuser_label(t, f.rule_id)})"
        for f in t.firings
    ]


def _stages(t: Trace) -> list[Stage]:
    """The same trace as a timeline: seven stages, numbers in rows.

    SAME FACTS, DIFFERENT SHAPE. Six prose paragraphs is the right artifact for
    an auditor reading start to finish and the wrong one for a projector. The
    prose is kept - `text()` still returns it and the ledger-style reading is
    unchanged - and the SCREEN gets the same figures as labelled rows, so a
    reader across a room can find the propensity without parsing a sentence.
    Both are generated here, from one trace, so they cannot drift apart.
    """
    ranked = sorted(t.considered, key=lambda c: -c.adjusted_value)[:5]
    binding = {k: v for k, v in t.shadow_prices.items() if v > 0}
    slack = sorted(k for k, v in t.shadow_prices.items() if v <= 0)

    return [
        Stage(
            # NO CLAIM ID, NO SUBJECT TOKEN. Both are already the heading and
            # the subtitle of this screen, and repeating an identifier three
            # inches below itself spends the first stage of the timeline on
            # nothing. Worse, the copy here was truncated while the subtitle
            # was not, so the same subject appeared to be two subjects. The
            # header carries the identity; the timeline carries the decision.
            label="Claim",
            rows=(
                ("at", t.at.strftime("%d %b %Y %H:%M UTC")),
                ("amount", format_inr(t.amount_paise)),
                ("relationship value", format_inr(t.ltv_paise)),
            ),
            prose=(
                "The objective weighs recovery against the relationship, not against "
                "the failed amount alone."
            ),
        ),
        Stage(
            label="Diagnosis",
            rows=(
                ("cause", t.cause_label.replace("_", " ")),
                ("layer", t.cause_layer.value),
                # ONE CONFIDENCE ROW, and it says whether the number was capped.
                # The uncapped read is not carried on the trace, so this states
                # the cap and its reason rather than inventing a "from" value.
                (
                    "confidence",
                    f"{t.confidence:.0%}, capped here because the cohort was too "
                    "thin to support a higher read"
                    if t.confidence_capped
                    else f"{t.confidence:.0%}, uncapped",
                ),
                ("answered by", t.answered_by.replace("_", " ")),
                ("cohort power", t.cohort_power.replace("_", " ").replace("power", "sample")),
            ),
            prose=f"Outreach is permitted because the cause is {_LAYER_MEANING[t.cause_layer]}."
            if t.cause_layer is CauseLayer.CUSTOMER
            else f"The cause is {_LAYER_MEANING[t.cause_layer]}.",
        ),
        Stage(
            label="Actions scored",
            rows=(("actions the Gate allowed to be scored", f"{len(t.considered)}"),),
            table=(
                ("action", "estimated effect on recovery", "value after budget prices"),
                *(
                    (
                        c.action.value.replace("_", " "),
                        f"{c.uplift:+.1%}",
                        format_inr(Paise(int(c.adjusted_value))),
                    )
                    for c in ranked
                ),
            ),
            prose=(
                "Effect is signed, so an action that would make things worse can "
                "never outrank declining to act."
            ),
        ),
        Stage(
            label="Budget prices",
            rows=tuple(
                (name, format_inr(Paise(int(value))))
                for name, value in sorted(binding.items(), key=lambda kv: -kv[1])
            )
            or (("binding budgets", "none"),),
            prose=(
                "A shadow price is the marginal unit's worth in recovery foregone "
                "elsewhere"
                + (f"; {', '.join(slack)} had slack and cost nothing." if slack else ".")
            ),
        ),
        Stage(
            label="Gate verdict",
            rows=(
                (
                    "compliance rules evaluated",
                    f"{len(t.registry)}, none objected"
                    if not _gate_refusers(t)
                    else f"{len(t.registry)}, {len(_gate_refusers(t))} objected",
                ),
                ("verdict", t.verdict.value),
                *(
                    (f.rule_id, f"{f.verdict.value} - {_refuser_label(t, f.rule_id)}")
                    for f in t.firings
                ),
            ),
            prose=(
                "The refusal came from the allocator's admission step, which is a "
                "budget limit and not one of the compliance rules above."
                if _outside_registry(t)
                else "Every rule is evaluated, not just the first to object, because "
                "the audit trail needs the whole verdict list."
            ),
        ),
        Stage(
            label="Decision",
            rows=(
                ("action drawn", t.sampled_action.value.replace("_", " ")),
                ("probability it was drawn with", f"{t.sampled_propensity:.3f}"),
                ("action taken", t.realized_action.value.replace("_", " ")),
                (
                    "probability of what actually happened",
                    f"{t.realized_propensity:.3f}",
                ),
                ("refused", "yes" if t.veto_occurred else "no"),
            ),
            prose=(
                "The refused probability collapsed onto declining to act rather than "
                "being discarded; dropping it would bias the headline."
                if t.veto_occurred
                else "The policy drew with this probability, so the decision is "
                "measurable off-policy."
            ),
        ),
        Stage(
            label="Outcome",
            rows=(
                ("outcome", t.outcome.replace("_", " ")),
                ("recovered", format_inr(t.recovered_paise)),
            ),
            prose="Credited to this decision by the last-touch rule, which is our convention.",
        ),
    ]


def trace_lines(view: ReplayView) -> list[str]:
    """The same prose, wrapped for a terminal. Used by the demo harness."""
    import textwrap

    lines: list[str] = []
    for paragraph in view.paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=76))
        lines.append("")
    return lines
