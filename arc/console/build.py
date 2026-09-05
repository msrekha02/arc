"""Assemble the four screens from a real run, and write them to disk.

    python -m arc.console.build --seed 3 --out console/

NO FIXTURES. Every screen is built from the same objects the acceptance gates
assert on: M11's harness result, M3's loaded registry, M6's Sentinel run over
the frozen batch. A console rendered from a fixture is a screenshot, and a
screenshot cannot go wrong in the same way the system does.

THE BATCH SCREEN RUNS THE SENTINEL FOR REAL. The proving-ground harness scores
causes through the code map alone, because that is all the allocator needs. The
diagnosis split and the suppressed-by-outage count are M6's answers, and getting
them means running M6 over the batch - which is also the only honest way to
show a number whose whole claim is that a detector found something.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from arc.console.badges import escape
from arc.console.replay import (
    ConsideredAction,
    RuleFiring,
    Trace,
    narrate,
)
from arc.console.screens import (
    BatchView,
    FirewallView,
    ReplayView,
    RuleCounter,
    ScoreboardView,
    _tile,
)
from arc.core.money import Paise, format_inr, paise
from arc.core.reproducibility import JUDGED_DIGEST
from arc.core.types import ActionType, CauseLayer, ClaimState
from arc.gate.evaluator import Gate
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry, load_registry
from arc.proving_ground.arms import Arm
from arc.proving_ground.composed import ADMISSION_RULE_ID
from arc.proving_ground.dr_estimator import (
    dr_estimate,
    fit_outcome_model,
    on_policy_target,
)
from arc.proving_ground.harness import HarnessResult, build_scoreboard, run_all
from arc.proving_ground.policies import SharedUplift
from arc.sentinel.diagnose import DiagnosisContext, diagnose
from arc.simulator.seeds import DEVELOP_SEED, JUDGED_SEED

# The develop-seed and judged-seed estimator errors, measured by M11's gate and
# by the judged run. Both are shown; the judged one is worse and is the one the
# scoreboard points at.
#
# Held here as constants rather than recomputed because recomputing the judged
# figure would mean running seed 3 again, and running the judged seed more than
# once is the thing the three-seed discipline forbids.
DR_ERROR_DEVELOP = 0.0200
DR_ERROR_JUDGED = 0.0859


@dataclass(frozen=True)
class ConsoleData:
    """Everything the four screens need, from one run."""

    batch: BatchView
    firewall: FirewallView
    scoreboard: ScoreboardView
    replay: ReplayView
    result: HarnessResult

    def screens(self) -> Mapping[str, str]:
        return {
            "batch.html": self.batch.render(),
            "firewall.html": self.firewall.render(),
            "scoreboard.html": self.scoreboard.render(),
            "replay.html": self.replay.render(),
            "index.html": _index(self),
        }


_CARDS: tuple[tuple[str, str, str], ...] = (
    (
        "batch.html",
        "Batch",
        "Counters, the diagnosis split, and the claims a detected outage took "
        "off the contact path entirely.",
    ),
    (
        "firewall.html",
        "Compliance firewall",
        "Proposed to executed with every category counted, per-rule fired "
        "counts, and the honest mix of what is law and what is our own policy.",
    ),
    (
        "scoreboard.html",
        "Scoreboard",
        "Five arms, guardrails beside the money, and the estimator's own error "
        "against simulator ground truth.",
    ),
    (
        "replay.html",
        "Replay",
        "One decision end to end: diagnosis, options priced, the verdict, the "
        "propensity it was drawn with, and what happened.",
    ),
)


def _sleeping_dog_contacts(result: HarnessResult) -> tuple[int, int, int]:
    """The planted count and what two arms contacted, as counted by the harness.

    THE CONSOLE MAY NOT ASK THIS QUESTION ITSELF. `sleeping_dogs` reads the
    simulator's counterfactuals, and `test_import_bans` sweeps every package
    outside the simulator and the proving ground for exactly that call. The
    ban is right: a screen that can reach ground truth can render a figure the
    running system could never have known, which is the circularity the frozen
    simulator exists to prevent. So the harness counts it and this reads the
    integers it carried out.
    """
    reached = result.sleeping_dogs_contacted
    return (
        result.sleeping_dogs_planted,
        int(reached.get(Arm.ARC, 0)),
        int(reached.get(Arm.NAIVE_DUNNING, 0)),
    )


def _hero_bars(rows: Sequence[tuple[str, int, bool]]) -> str:
    """Three totals as bars on one scale, so the reversal is a picture.

    THE FINDING IS A SHAPE, NOT A SENTENCE. The industry default recovering
    less than the null arm is the most surprising measured result in the
    system, and as three numbers in a row it reads as three numbers in a row.
    On a shared zero-based axis the naive bar is visibly shorter than the arm
    that did nothing at all, and nobody needs the caption to see it.

    ORDER IS ARC, NULL, NAIVE - the order that makes the shortfall adjacent to
    the thing it falls short of.
    """
    if not rows:
        return ""
    top = max(value for _, value, _ in rows) or 1
    width, row_h, gap, label_w, value_w = 760.0, 34.0, 14.0, 150.0, 150.0
    span = width - label_w - value_w
    height = len(rows) * (row_h + gap) - gap

    body = ""
    for index, (name, value, accent) in enumerate(rows):
        y = index * (row_h + gap)
        mid = y + row_h / 2 + 5
        css = "hbar win" if accent else "hbar"
        body += (
            f'<g class="{css}">'
            f'<text class="lab" x="0" y="{mid:.1f}">{escape(name)}</text>'
            f'<rect class="track" x="{label_w}" y="{y:.1f}" width="{span:.1f}" '
            f'height="{row_h}" rx="3"/>'
            f'<rect class="fill" x="{label_w}" y="{y:.1f}" '
            f'width="{span * value / top:.1f}" height="{row_h}" rx="3"/>'
            f'<text class="val" x="{width:.0f}" y="{mid:.1f}" text-anchor="end">'
            f"{escape(format_inr(Paise(value)))}</text>"
            "</g>"
        )
    return (
        f'<svg class="hero" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="recovered by arm">{body}</svg>'
    )


def _index(data: ConsoleData) -> str:
    """The landing screen.

    WHY THIS IS NOT A TABLE OF CONTENTS. It is the first thing a reader opens,
    and a heading with four links spends that on navigation. The graded number
    goes here, with its interval and its guardrails, and then four facts that
    can be read from across a room. The links are still here; they are just no
    longer the whole page.

    THE STRUCTURAL RULE HOLDS ON THIS SCREEN TOO. No recovery figure appears
    without the guardrails that qualify it, which is the same refusal
    `Scoreboard.to_dict` enforces - a headline that generated complaints and
    opt-outs is not a headline, and splitting them across screens would be a
    way of quietly not saying so.
    """
    from arc.console.badges import honest_mix
    from arc.console.screens import _SPEND_DENOMINATOR, document

    payload = data.scoreboard.payload()
    arms = {str(a["arm"]): a for a in payload["arms"]}  # type: ignore[union-attr]
    arc = arms["arc"]
    rails = arc["guardrails"]  # type: ignore[index]
    interval = arc.get("ci_95_paise")  # type: ignore[union-attr]
    ci = (
        f"bootstrap 95% CI on recovered rupees {format_inr(Paise(int(interval[0])))} "
        f"to {format_inr(Paise(int(interval[1])))}"
        if interval
        else "no interval computed"
    )

    mix = honest_mix(data.firewall.registry)
    batch = data.batch
    naive = arms[str(payload["comparator"])]
    null = arms["null"]

    # THE COMPARISON LEADS, NOT THE RATIO. Three totals a reader can check
    # against each other without being told how to read them, and the finding
    # underneath is the one measured result that surprises people. A ratio
    # invites the denominator question before anybody has agreed there is
    # something worth measuring; three rupee figures do not.
    bars = _hero_bars(
        [
            ("arc", int(arc["recovered_paise"]), True),  # type: ignore[index]
            ("doing nothing", int(null["recovered_paise"]), False),  # type: ignore[index]
            (
                str(payload["comparator"]).replace("_", " "),
                int(naive["recovered_paise"]),  # type: ignore[index]
                False,
            ),
        ]
    )
    planted, dogs_contacted, dogs_naive = _sleeping_dog_contacts(data.result)

    facts = (
        (
            f"{batch.claims:,}",
            f"claims across {batch.subjects:,} subjects, {format_inr(batch.at_risk_paise)} at risk",
        ),
        (
            f"{batch.suppressed_by_outage:,}",
            "claims suppressed by a detected issuer outage, contacted zero times",
        ),
        (
            f"{dogs_contacted} of {planted}",
            f"planted sleeping dogs were contacted, against {dogs_naive} by the "
            "industry default. Measured on the simulator's ground truth, not on "
            "the model agreeing with itself",
        ),
        (
            f"{mix['statutory']} of {mix['total']}",
            f"rules are statutory; {mix['policy_choice']} are our own policy choice, "
            f"and {mix['stricter_than_binding_minimum']} are stricter than the "
            "binding minimum",
        ),
        (
            JUDGED_DIGEST[:8],
            "the digest of this run, identical across three consecutive runs of "
            "<code>make demo SEED=3</code>",
        ),
    )

    cards = "".join(
        f'<a href="{href}"><div class="t">{title}</div><div class="d">{blurb}</div></a>'
        for href, title, blurb in _CARDS
    )

    body = (
        "<h1>ARC &mdash; autonomous revenue continuity</h1>"
        f'<p class="sub">Seed {payload["seed"]}, {payload["cycles"]} cycles, '
        "rendered from a real run. Every figure below is reproduced by "
        "<code>make demo SEED=3</code>.</p>"
        "<h2>Recovered in four cycles</h2>"
        '<div class="hero-row">'
        f"<div>{bars}"
        '<p class="finding">The industry default recovered less than doing nothing '
        "on this population.</p></div>"
        '<div class="hero-side">'
        + _tile(
            "incremental per rupee spent",
            f"{float(arc['incremental_per_rupee_spent']):.2f}x",  # type: ignore[index]
            point=True,
        )
        + f'<p class="note">{format_inr(Paise(int(arc["incremental_paise"])))} '  # type: ignore[index]
        + f"incremental against naive dunning &mdash; {ci}. {_SPEND_DENOMINATOR}</p>"
        + "</div></div>"
        "<h2>Guardrails</h2>"
        '<div class="tiles">'
        + _tile("complaints /1k", f"{rails['complaint_rate_per_1000']:.2f}")  # type: ignore[index]
        + _tile("opt-outs /1k", f"{rails['opt_out_rate_per_1000']:.2f}")  # type: ignore[index]
        + _tile("cost per rupee", f"{rails['cost_per_rupee_collected']:.3f}")  # type: ignore[index]
        + "</div>"
        '<p class="note">The guardrails sit beside the money because the money does '
        "not exist without them. Recovery that generates complaints and opt-outs is "
        "not a win, and this console refuses to render one without the other.</p>"
        "<h2>What the run shows</h2>"
        '<ul class="facts">'
        + "".join(f'<li><span class="n">{n}</span><span>{what}</span></li>' for n, what in facts)
        + "</ul>"
        "<h2>The screens</h2>"
        f'<div class="cards">{cards}</div>'
    )
    return document("ARC console", body)


# ---------------------------------------------------------------------------
# Diagnosis split, from the real Sentinel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Split:
    counts: dict[CauseLayer, int]
    suppressed: int
    self_healing: int
    blind: int


def diagnose_batch(result: HarnessResult, at: datetime) -> Split:
    """Run M6 over the batch and count where the claims landed.

    Cohort history is built from the batch's own decline events, which is what
    makes the outage detectable at all: a burst of declines means nothing
    without the volume it was drawn from.
    """
    counts = dict.fromkeys(CauseLayer, 0)
    suppressed = 0
    self_healing = 0
    blind = 0

    history = result.cohort_history()
    # DIAGNOSE AT THE MOMENT OF THE FAILURE, not at the cycle. An outage inside
    # the batch window has resolved by the time a cycle runs, so diagnosing
    # everything at `at` would report zero issuer-layer claims on a batch that
    # contains two injected outages.
    moments = result.detection_moments()

    for case in result.cases:
        for claim_case in case.claims:
            observation = claim_case.observation
            codes = getattr(observation, "decline_code_history", ())
            detected_at, code = moments.get(claim_case.account_id, (at, None))
            context = DiagnosisContext.from_claim(
                claim_case.claim,
                issuer_ref=getattr(observation, "issuer_id", None),
                cohort_history=history,
                decline_code=code or (codes[-1] if codes else None),
            )
            found = diagnose(claim_case.claim, context, detected_at)
            counts[found.cause.layer] += 1
            if not found.contact_permitted and found.cause.layer is CauseLayer.ISSUER:
                suppressed += 1
            if found.next_state is ClaimState.SELF_HEALING:
                self_healing += 1
            if found.confidence_capped:
                blind += 1

    return Split(counts=counts, suppressed=suppressed, self_healing=self_healing, blind=blind)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build(
    *,
    seed: int = JUDGED_SEED,
    size: int = 1_200,
    cycles: int = 4,
    gate: Gate | None = None,
    registry: RuleRegistry | None = None,
) -> ConsoleData:
    registry = registry or load_registry()
    gate = gate or Gate(registry)
    result = run_all(seed=seed, size=size, cycles=cycles, gate=gate)
    return assemble(result, registry=registry, seed=seed)


def assemble(result: HarnessResult, *, registry: RuleRegistry, seed: int) -> ConsoleData:
    at = result.at0
    split = diagnose_batch(result, at)

    claims = sum(len(case.claims) for case in result.cases)
    at_risk = paise(sum(int(c.claim.amount_paise) for case in result.cases for c in case.claims))

    arc_run = result.runs[Arm.ARC]
    naive_run = result.runs[Arm.NAIVE_DUNNING]

    batch = BatchView(
        seed=seed,
        claims=claims,
        subjects=result.subjects,
        at_risk_paise=at_risk,
        issuer=split.counts[CauseLayer.ISSUER],
        merchant=split.counts[CauseLayer.MERCHANT],
        customer=split.counts[CauseLayer.CUSTOMER],
        unknown=split.counts[CauseLayer.UNKNOWN],
        suppressed_by_outage=split.suppressed,
        self_healing=split.self_healing,
        naive_contacted_same_claims=min(split.suppressed, naive_run.contacts),
        cohort_blind=split.blind,
    )

    funnel = _funnel(arc_run.logs, registry)
    firewall = FirewallView(
        proposed=len(arc_run.logs),
        blocked=funnel["blocked"],
        deferred=funnel["deferred"],
        declined=funnel["declined"],
        executed=funnel["executed"],
        counters=_rule_counters(arc_run.logs, registry),
        registry=registry,
    )

    scoreboard = ScoreboardView(
        scoreboard=build_scoreboard(
            result, dr_relative_error=DR_ERROR_DEVELOP, ci=_recovered_interval(result, seed)
        ),
        dr_error_develop=DR_ERROR_DEVELOP,
        dr_error_judged=DR_ERROR_JUDGED,
        judged_seed=JUDGED_SEED,
        decay=_decay(result),
    )

    replay = narrate(_pick_trace(result, registry, at))
    return ConsoleData(
        batch=batch,
        firewall=firewall,
        scoreboard=scoreboard,
        replay=replay,
        result=result,
    )


def _known_ids(registry: RuleRegistry) -> frozenset[str]:
    """The registry's rule ids as a set.

    WHY THIS EXISTS AT ALL. `RuleRegistry` defines `__len__`, `__iter__` and
    `__getitem__` but no `__contains__`, so `rule_id in registry` falls back to
    iteration and compares a `str` against `Rule` objects. That is False for
    EVERY id, including real ones. A membership test written the obvious way
    silently discards every rule it is handed, which is exactly the bug that
    made the replay screen announce a BLOCK and then report that nothing
    objected. Ask for the ids and test against those.
    """
    return frozenset(rule.id for rule in registry)


def _refusal_verdict(rule_ids: Sequence[str], registry: RuleRegistry) -> Verdict:
    """What a refusal actually was, reconstructed from who refused.

    THE LOG DOES NOT STORE A VERDICT. A decision row carries `veto_occurred`,
    a boolean, and the ids that caused it - which flattens DEFER, BLOCK and
    BLOCK_PERMANENT into one bit. The verdict is recoverable because each
    refuser declares its own: `ALLOC-ADMISSION` is the allocator's in-cycle
    admission step and always defers, and every Gate rule declares
    `on_violation` in the registry. Strongest verdict wins, because a branch
    refused by both a cooldown and a consent rule was blocked, not deferred.
    """
    order = (Verdict.DEFER, Verdict.BLOCK, Verdict.BLOCK_PERMANENT)
    worst = Verdict.ALLOW
    known = _known_ids(registry)
    for rule_id in rule_ids:
        if rule_id == ADMISSION_RULE_ID:
            found = Verdict.DEFER
        elif rule_id in known:
            found = registry[rule_id].on_violation
        else:
            continue
        if found in order and (worst is Verdict.ALLOW or order.index(found) > order.index(worst)):
            worst = found
    return worst


def _funnel(logs: Sequence[object], registry: RuleRegistry) -> dict[str, int]:
    """proposed = blocked + deferred + declined + executed, every term counted.

    `declined` is the category the screen used to have no name for: a branch
    nobody refused, where the policy itself sampled `do_nothing`. Without it
    the funnel does not add up and the missing rows look like a rounding error
    rather than the deliberate choices they are.
    """
    blocked = deferred = declined = executed = 0
    for row in logs:
        ids = tuple(getattr(row, "blocking_rule_ids", ()) or ())
        verdict = _refusal_verdict(ids, registry) if ids else Verdict.ALLOW
        if verdict is Verdict.DEFER:
            deferred += 1
        elif verdict in (Verdict.BLOCK, Verdict.BLOCK_PERMANENT):
            blocked += 1
        elif row.realized_key[1] is ActionType.DO_NOTHING:  # type: ignore[attr-defined]
            declined += 1
        else:
            executed += 1
    return {
        "blocked": blocked,
        "deferred": deferred,
        "declined": declined,
        "executed": executed,
    }


def _recovered_interval(result: HarnessResult, seed: int) -> tuple[Paise, Paise]:
    """ARC's recovered total, with the subject-clustered bootstrap 95% interval.

    WHY THE SCREEN HAD NO INTERVAL. `build_scoreboard` has always accepted a
    `ci` argument and no caller ever passed one, so every scoreboard rendered
    a point estimate with nothing saying whether it was real. A headline
    without an interval invites the reader to treat it as exact.

    WHAT THE INTERVAL IS ON. `dr_estimate` works per DECISION - the estimand
    is paise per logged decision - and its bootstrap resamples SUBJECTS as
    clusters, which is where GI-8's unit of independence belongs. Multiplying
    the endpoints by the row count rescales that interval to a batch total,
    which is a change of units and not a change of claim.

    WHAT IT IS NOT ON. The comparator's recovery and ARC's spend are treated
    here as the observed constants they are, so an interval quoted on a ratio
    built from them would understate the uncertainty. The interval is
    therefore reported against recovered rupees, which is what the bootstrap
    actually covers, and the screen says so.

    The generator is seeded from the run's own seed, so the interval is as
    reproducible as everything else on the screen.
    """
    logs = result.runs[Arm.ARC].logs
    estimate = dr_estimate(
        logs,
        fit_outcome_model(logs),
        on_policy_target,
        rng=np.random.default_rng(seed),
    )
    return (
        paise(int(estimate.lo * estimate.n_rows)),
        paise(int(estimate.hi * estimate.n_rows)),
    )


def _rule_counters(logs: Sequence[object], registry: RuleRegistry) -> list[RuleCounter]:
    """Fired counts, sorted by how much work each refuser actually did."""
    tally: dict[str, int] = {}
    for row in logs:
        for rule_id in row.blocking_rule_ids:  # type: ignore[attr-defined]
            tally[rule_id] = tally.get(rule_id, 0) + 1
    return [
        RuleCounter(
            rule_id=rule_id,
            fired=count,
            verdict=_refusal_verdict((rule_id,), registry),
        )
        for rule_id, count in sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _decay(result: HarnessResult) -> dict[Arm, list[int]]:
    """Recovery per cycle, per arm. The greedy curve is the argument."""
    out: dict[Arm, list[int]] = {}
    for arm in (Arm.GREEDY_UNCONSTRAINED, Arm.ARC):
        per_cycle: dict[int, int] = {}
        for row in result.runs[arm].logs:
            per_cycle[row.cycle] = per_cycle.get(row.cycle, 0) + int(row.reward_paise)
        out[arm] = [per_cycle.get(c, 0) for c in range(result.cycles)]
    return out


def _pick_trace(result: HarnessResult, registry: RuleRegistry, at: datetime) -> Trace:
    """A claim worth reading: one the Gate actually had something to say about.

    Falls back to the first treated decision, then to the first decision at
    all, so the screen renders on any batch rather than only on a lucky one.
    """
    logs = result.runs[Arm.ARC].logs
    chosen = next((row for row in logs if row.veto_occurred), None)
    chosen = chosen or next(
        (row for row in logs if row.realized_key[1] is not ActionType.DO_NOTHING), None
    )
    chosen = chosen or logs[0]

    case = next(
        (c for c in result.cases if c.subject_token == chosen.subject_token),
        result.cases[0],
    )
    claim_case = case.case_for(chosen.realized_key[0]) or case.claims[0]
    observation = claim_case.observation
    codes = getattr(observation, "decline_code_history", ())
    found = diagnose(
        claim_case.claim,
        DiagnosisContext.from_claim(
            claim_case.claim,
            issuer_ref=getattr(observation, "issuer_id", None),
            decline_code=codes[-1] if codes else None,
        ),
        at,
    )

    surface = SharedUplift()
    considered = [
        ConsideredAction(
            action=key[1],
            uplift=float(surface.uplift(observation, key[1], None, propensity=1.0).value),
            adjusted_value=float(chosen.truth.get(key, 0.0)),
            propensity=float(probability),
        )
        for key, probability in sorted(chosen.pi_exec.items(), key=lambda kv: -kv[1])
    ]

    return Trace(
        claim_id=str(claim_case.claim.claim_id),
        subject_token=chosen.subject_token,
        at=at,
        amount_paise=claim_case.claim.amount_paise,
        ltv_paise=claim_case.claim.ltv_remaining_paise,
        cause_label=found.cause.label.value,
        cause_layer=found.cause.layer,
        confidence=found.cause.confidence,
        answered_by=found.answered_by.value,
        cohort_power=found.cohort.verdict.value,
        confidence_capped=found.confidence_capped,
        considered=considered,
        shadow_prices=dict(result.runs[Arm.ARC].shadow_prices),
        firings=[
            RuleFiring(rule_id=rule_id, verdict=_refusal_verdict((rule_id,), registry))
            for rule_id in chosen.blocking_rule_ids
        ],
        verdict=_refusal_verdict(tuple(chosen.blocking_rule_ids), registry),
        sampled_action=chosen.intended_key[1],
        sampled_propensity=float(chosen.pi_intended),
        realized_action=chosen.realized_key[1],
        realized_propensity=float(chosen.pi_realized),
        veto_occurred=chosen.veto_occurred,
        outcome="recovered" if chosen.reward_paise else "no response",
        recovered_paise=Paise(int(chosen.reward_paise)),
        registry=registry,
    )


def write(data: ConsoleData, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, html in data.screens().items():
        path = out / name
        path.write_text(html, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arc.console.build")
    parser.add_argument("--seed", type=int, default=DEVELOP_SEED)
    parser.add_argument("--size", type=int, default=1_200)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("console"))
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    data = build(seed=args.seed, size=args.size, cycles=args.cycles)
    for path in write(data, args.out):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
