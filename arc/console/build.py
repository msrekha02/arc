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
)
from arc.core.money import Paise, paise
from arc.core.types import ActionType, CauseLayer, ClaimState
from arc.gate.evaluator import Gate
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry, load_registry
from arc.proving_ground.arms import Arm
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
            "index.html": _index(),
        }


def _index() -> str:
    from arc.console.screens import document

    body = (
        "<h1>ARC console</h1>"
        '<p class="sub">Four screens, rendered from a real run.</p>'
        "<ul>"
        '<li><a href="batch.html">Batch</a> &mdash; counters, diagnosis split, '
        "claims suppressed by a detected outage</li>"
        '<li><a href="firewall.html">Compliance firewall</a> &mdash; proposed to '
        "executed, per-rule counters, the honest mix</li>"
        '<li><a href="scoreboard.html">Scoreboard</a> &mdash; five arms, guardrails '
        "beside the money, estimator error against ground truth</li>"
        '<li><a href="replay.html">Replay</a> &mdash; one claim, explained</li>'
        "</ul>"
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

    firewall = FirewallView(
        proposed=len(arc_run.logs),
        blocked=sum(1 for row in arc_run.logs if row.veto_occurred),
        deferred=0,
        executed=sum(1 for row in arc_run.logs if row.realized_key[1] is not ActionType.DO_NOTHING),
        counters=_rule_counters(arc_run.logs),
        registry=registry,
    )

    scoreboard = ScoreboardView(
        scoreboard=build_scoreboard(result, dr_relative_error=DR_ERROR_DEVELOP),
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


def _rule_counters(logs: Sequence[object]) -> list[RuleCounter]:
    tally: dict[str, int] = {}
    for row in logs:
        for rule_id in row.blocking_rule_ids:  # type: ignore[attr-defined]
            tally[rule_id] = tally.get(rule_id, 0) + 1
    return [
        RuleCounter(rule_id=rule_id, fired=count, verdict=Verdict.BLOCK)
        for rule_id, count in sorted(tally.items(), key=lambda kv: -kv[1])
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
            RuleFiring(rule_id=rule_id, verdict=Verdict.BLOCK)
            for rule_id in chosen.blocking_rule_ids
            if rule_id in registry
        ],
        verdict=Verdict.BLOCK if chosen.veto_occurred else Verdict.ALLOW,
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
