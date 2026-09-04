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

from arc.console.screens import ReplayView
from arc.core.money import Paise, format_inr
from arc.core.types import ActionType, CauseLayer
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry

_LAYER_MEANING: Mapping[CauseLayer, str] = {
    CauseLayer.ISSUER: ("the issuer's, not the customer's, so nobody should be contacted about it"),
    CauseLayer.MERCHANT: ("our own setup, so it is repaired at the rail with no customer contact"),
    CauseLayer.CUSTOMER: "the customer's, so outreach is on the table",
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
            f"{c.uplift:+.1%} effect, worth {c.adjusted_value:,.0f} after prices"
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
        parts = [f"{k} at {v:,.0f}" for k, v in sorted(binding.items(), key=lambda kv: -kv[1])]
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
        described = []
        for firing in t.firings:
            rule = t.registry[firing.rule_id]
            # FORCE ALWAYS THROUGH M3. Never `rule.basis` in a sentence.
            described.append(
                f"{firing.rule_id} returned {firing.verdict.value} ({rule.force_label()})"
            )
        out.append(
            f"The Gate evaluated all {len(t.registry)} rules and returned "
            f"{t.verdict.value}. The rules that spoke: {'; '.join(described)}. Every "
            f"rule was evaluated, not just the first to object, because the audit "
            f"trail needs the whole verdict list."
        )
    else:
        out.append(
            f"The Gate evaluated all {len(t.registry)} rules and returned "
            f"{t.verdict.value}. Nothing objected."
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


def trace_lines(view: ReplayView) -> list[str]:
    """The same prose, wrapped for a terminal. Used by the demo harness."""
    import textwrap

    lines: list[str] = []
    for paragraph in view.paragraphs:
        lines.extend(textwrap.wrap(paragraph, width=76))
        lines.append("")
    return lines
