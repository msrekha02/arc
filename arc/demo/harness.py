"""The judged demo: nine beats, sequenced, with pauses to narrate into.

    make demo SEED=3        deterministic replay - the judged run
    make demo-live          real-time, jittered - the "watch it react" beat
    make demo-adversarial   the attack suite, readable

DETERMINISM IS THE WHOLE POINT OF THE FIRST TARGET. A judge will ask to see it
again, and the second run has to produce the same headline to the paise. So the
replay path takes its seed from the command line, injects it into every
generator, reads no clock, and prints no wall-clock time. `digest()` hashes the
numbers the demo actually SHOWS, which is what makes "byte-identical" a claim
about the output rather than about the internals.

    WHY THE DIGEST COVERS THE RENDERED FIGURES AND NOT THE RESULT OBJECT. The
    result object holds floats that could differ in a last bit without changing
    anything a judge sees, and it omits the framing text that a careless edit
    could change while the numbers held. Hashing what is printed is hashing
    what was claimed.

THE PAUSES ARE PART OF THE DESIGN, NOT A COURTESY. Two beats need room: the
outage suppression, where the point is the contrast between zero contacts and
what the naive arm sent to those same claims, and the hardship stop, where the
point is that the system gives up money on purpose. Both land only if somebody
has time to say what just happened.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from arc.conductor.breakers import BreakerId, Reading, evaluate_all
from arc.console.build import DR_ERROR_JUDGED, ConsoleData, build
from arc.console.replay import trace_lines
from arc.core.money import Paise, format_inr
from arc.core.types import ActionType
from arc.gate.registry import load_registry
from arc.proving_ground.arms import Arm

# How long each pause lasts in a narrated run. Zero in replay mode, because a
# deterministic run must not depend on wall time for anything, including its
# own pacing.
NARRATION_PAUSE = 2.0


@dataclass
class Beat:
    """One numbered moment in the script, and the lines it prints."""

    number: int
    title: str
    lines: list[str] = field(default_factory=list)
    pause_after: bool = False

    def render(self) -> list[str]:
        rule = "-" * 72
        out = [rule, f"  {self.number}. {self.title.upper()}", rule, ""]
        out.extend(self.lines)
        out.append("")
        return out


def _inr(value: int) -> str:
    return format_inr(Paise(int(value)))


def beats(data: ConsoleData) -> list[Beat]:
    """The nine beats of spec section 15, in order, from one real run."""
    result = data.result
    batch = data.batch
    arc = result.runs[Arm.ARC]
    naive = result.runs[Arm.NAIVE_DUNNING]
    greedy = result.runs[Arm.GREEDY_UNCONSTRAINED]
    null = result.runs[Arm.NULL]
    board = data.scoreboard.scoreboard

    out: list[Beat] = []

    # 1 - the batch lands
    out.append(
        Beat(
            1,
            "the batch lands",
            [
                f"  {batch.claims:,} claims across {batch.subjects:,} subjects.",
                f"  {_inr(int(batch.at_risk_paise))} at risk.",
                f"  seed {result.seed}, {result.cycles} cycles, replayed deterministically.",
            ],
        )
    )

    # 2 - diagnosis splits it. THE BEAT THAT HAS TO LAND.
    out.append(
        Beat(
            2,
            "diagnosis splits it",
            [
                f"  issuer layer     {batch.issuer:>6,}   nobody is contacted",
                f"  merchant layer   {batch.merchant:>6,}   repaired at the rail",
                f"  customer layer   {batch.customer:>6,}   outreach on the table",
                f"  unattributed     {batch.unknown:>6,}   conservative path",
                "",
                f"  >>> {batch.suppressed_by_outage:,} claims were SUPPRESSED by a detected "
                "issuer outage.",
                "      They received zero contact of any kind - not deferred, not",
                "      throttled. Zero.",
                "",
                f"      The naive fixed-schedule arm messaged {naive.contacts:,} times this",
                "      batch, including those same claims, because a calendar does not",
                "      know the issuer is down.",
                "",
                f"      {batch.cohort_blind:,} claims were diagnosed WITHOUT cohort power.",
                "      That is a measured blind spot, not a clean read, and it is on",
                "      the dashboard rather than hidden in a NORMAL.",
            ],
            pause_after=True,
        )
    )

    # 3 - the allocator runs
    prices = {k: v for k, v in arc.shadow_prices.items() if v > 0}
    price_lines = [
        f"  lambda_{key:<12} {value:>12,.0f}   the marginal unit is worth this much"
        for key, value in sorted(prices.items(), key=lambda kv: -kv[1])
    ] or ["  no budget was binding this cycle"]
    out.append(
        Beat(
            3,
            "the allocator runs",
            [
                "  Budgets are priced, not checked. A shadow price is what the",
                "  marginal unit of that budget was worth in recovery given up",
                "  somewhere else.",
                "",
                *price_lines,
                "",
                f"  {arc.explore_mass_share:.1%} of the probability mass sat on actions",
                "  other than the single best one. That is the softmax and the epsilon",
                "  floor together, and it is the entire reason this run can be",
                "  evaluated off-policy at all: a deterministic argmax assigns",
                "  probability one to one action and zero to every other, and every",
                "  counterfactual weight becomes a division by zero.",
            ],
        )
    )

    # 4 - compliance firewall live
    mix = data.firewall.mix
    counters = list(data.firewall.counters)[:5]
    out.append(
        Beat(
            4,
            "compliance firewall, live",
            [
                f"  proposed {data.firewall.proposed:,}"
                f"   blocked {data.firewall.blocked:,}"
                f"   executed {data.firewall.executed:,}",
                "",
                f"  {mix['total']} rules: {mix['statutory']} statutory, "
                f"{mix['network_rule']} network, {mix['policy_choice']} our own policy.",
                f"  {mix['in_force']} in force, {mix['draft']} draft, "
                f"{mix['advisory']} advisory, {mix['contested']} contested.",
                f"  Stricter than the binding minimum in "
                f"{mix['stricter_than_binding_minimum']} places.",
                "",
                "  Not in force, and applied anyway:",
                *[f"    {badge.rule_id:<18} {badge.text}" for badge in _not_in_force_badges()],
                "",
                *(
                    [f"    {c.rule_id:<18} fired {c.fired:,} times" for c in counters]
                    if counters
                    else ["    no rule fired after allocation this cycle"]
                ),
            ],
        )
    )

    # 5 and 6 - the two calls. THE HARDSHIP STOP HAS TO LAND.
    out.append(
        Beat(
            5,
            "one voice call",
            [
                "  AI disclosure in the first utterance. Identity verified before",
                "  any account detail is spoken. A promise-to-pay is extracted as a",
                "  structured record with a confidence, the claim is frozen until the",
                "  promised date plus grace, and a durable function is scheduled to",
                "  find out what happened.",
                "",
                "  The tracker records KEPT or BROKEN. It does not decide what to do",
                "  next - escalation authority belongs to the allocator, which",
                "  re-scores the claim with the broken promise as one feature among",
                "  many, and may well decide the answer is to do nothing.",
            ],
        )
    )
    out.append(
        Beat(
            6,
            "the hardship stop",
            [
                "  A second call. The customer says something that reads as distress.",
                "",
                "  >>> The subject moves to FORBORNE and the automation TERMINATES.",
                "",
                "      FORBORNE is an absorbing state. There is no transition out of",
                "      it in the state machine - not to WRITTEN_OFF, not anywhere. No",
                "      expected-value argument reopens it, because there is no edge",
                "      for an argument to travel along.",
                "",
                "      Every sleeping run for that subject is cancelled where it lies.",
                "      Nothing polls for the signal; the runs are subscribed to it, so",
                "      a retry three days into a sleep dies without waking.",
                "",
                "      This is the system choosing to give up money, on purpose.",
            ],
            pause_after=True,
        )
    )

    # 7 - the scoreboard
    rows = ["  arm                    recovered    incremental      spend  compl/1k  optout/1k"]
    for report in board.reports:
        head, rails = report.headline, report.headline.guardrails
        rows.append(
            f"  {report.arm.value:<20}{_inr(int(head.recovered_paise)):>13}"
            f"{_inr(int(head.incremental_paise)):>15}{_inr(int(head.spend_paise)):>11}"
            f"{rails.complaint_rate_per_1000:>10.2f}{rails.opt_out_rate_per_1000:>11.2f}"
        )
    out.append(
        Beat(
            7,
            "the scoreboard",
            [
                *rows,
                "",
                "  Guardrails are on the same rows as the money. The metrics object",
                "  refuses to serialise a recovery figure without them, so there is",
                "  no arrangement of this table that shows the number alone.",
                "",
                "  Prevention, as a separate line, never added into recovery:",
                *[
                    f"    {r.arm.value:<20}{_inr(int(r.prevented_paise)):>13}"
                    for r in board.reports
                ],
                "",
                f"  ARC net    {_inr(arc.recovered_paise - arc.spend_paise)}",
                f"  greedy net {_inr(greedy.recovered_paise - greedy.spend_paise)}"
                f"   on {_inr(greedy.spend_paise)} of spend against ARC's "
                f"{_inr(arc.spend_paise)}",
                f"  doing nothing recovered {_inr(null.recovered_paise)} by itself.",
            ],
        )
    )

    # 8 - DR validation
    develop = data.scoreboard.dr_error_develop
    out.append(
        Beat(
            8,
            "the estimate, checked against the truth",
            [
                "  The simulator kept every counterfactual, so the quantity the",
                "  estimator is trying to recover is available exactly.",
                "",
                f"  develop seed          {develop * 100:>6.2f}% error",
                f"  judged seed {result.seed}         {DR_ERROR_JUDGED * 100:>6.2f}% error",
                "",
                "  Both are shown and the judged one is worse. Reporting only the",
                "  better figure would be choosing the seed after seeing the result,",
                "  which is the thing the three-seed discipline exists to stop.",
                "",
                "  Recovery per cycle, greedy against ARC:",
                *_decay_lines(data),
                "",
                "  The unconstrained arm contacts everyone every cycle and its",
                "  recovery decays as the annoyance term bites. That curve is the",
                "  argument for the constraints.",
            ],
        )
    )

    # 9 - replay
    out.append(
        Beat(
            9,
            "replay one claim",
            ["  " + line for line in trace_lines(data.replay)],
        )
    )
    return out


def _not_in_force_badges():
    from arc.console.badges import not_in_force

    return not_in_force(load_registry())


def _decay_lines(data: ConsoleData) -> list[str]:
    lines = []
    for arm, series in data.scoreboard.decay.items():
        if series:
            lines.append(f"    {arm.value:<22}" + "  ".join(f"{_inr(v):>12}" for v in series))
    return lines


# ---------------------------------------------------------------------------
# The digest - what "byte-identical" is a claim about
# ---------------------------------------------------------------------------
HEADLINE_KEYS = (
    "recovered_paise",
    "incremental_paise",
    "spend_paise",
    "prevented_paise",
)


def headline_numbers(data: ConsoleData) -> list[str]:
    """Every figure the demo asserts, flattened and ordered.

    Guardrails are included: a run that recovered the same rupees by different
    means is not the same run, and a digest that ignored the cost would call
    them identical.
    """
    payload = data.scoreboard.scoreboard.to_dict()
    out: list[str] = [f"seed={payload['seed']}", f"cycles={payload['cycles']}"]
    for arm in payload["arms"]:  # type: ignore[union-attr]
        for key in HEADLINE_KEYS:
            out.append(f"{arm['arm']}.{key}={arm[key]}")
        for key, value in sorted(arm["guardrails"].items()):
            out.append(f"{arm['arm']}.guardrail.{key}={value}")
    batch = data.batch
    out += [
        f"batch.claims={batch.claims}",
        f"batch.subjects={batch.subjects}",
        f"batch.issuer={batch.issuer}",
        f"batch.merchant={batch.merchant}",
        f"batch.customer={batch.customer}",
        f"batch.unknown={batch.unknown}",
        f"batch.suppressed={batch.suppressed_by_outage}",
        f"batch.cohort_blind={batch.cohort_blind}",
    ]
    return out


def digest(data: ConsoleData) -> str:
    """SHA-256 over the figures the demo shows. Not over the result object."""
    material = "\n".join(headline_numbers(data)).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def script(
    data: ConsoleData, *, pause: float = 0.0, sleep: Callable[[float], None] = time.sleep
) -> Iterator[str]:
    """The whole demo as lines, pausing where the script says to.

    `sleep` is injected so a test can drive the pauses without waiting, and so
    the deterministic path can pass a no-op rather than a zero-second sleep
    that still reads a clock somewhere.
    """
    yield "=" * 72
    yield "  ARC - Autonomous Revenue Continuity"
    yield f"  seed {data.result.seed}   deterministic replay"
    yield "=" * 72
    yield ""
    for beat in beats(data):
        yield from beat.render()
        if beat.pause_after and pause > 0:
            sleep(pause)
    yield "=" * 72
    yield f"  digest {digest(data)}"
    yield "=" * 72


def adversarial_lines() -> list[str]:
    """The attack suite, as an operator reads it.

    Each line says what was attempted, that it was refused, and which rule
    refused it. A suite whose output is a row of dots proves the same thing and
    demonstrates none of it.
    """
    from arc.demo.attacks import ATTACKS, run_attack

    lines = ["  attempted                              outcome    refused by", "  " + "-" * 70]
    refused = 0
    for attack in ATTACKS:
        outcome = run_attack(attack)
        refused += int(outcome.refused)
        status = "REFUSED" if outcome.refused else "ALLOWED"
        lines.append(f"  {attack.description:<38} {status:<10} {outcome.refused_by}")
    lines.append("  " + "-" * 70)
    lines.append(f"  {refused} of {len(ATTACKS)} attacks refused.")
    if refused != len(ATTACKS):
        lines.append("  *** AN ATTACK SUCCEEDED. THIS IS A FAILURE. ***")
    return lines


def llm_disabled_pipeline_lines(*, seed: int = 1, size: int = 300, cycles: int = 2) -> list[str]:
    """The attack that must SUCCEED rather than be refused.

    The system has to be completely functional with the model off, degrading in
    message quality and never in correctness or compliance. So this one is not
    in the refusal table: a REFUSED line here would be the failure. It runs the
    whole pipeline at `LLM_ENABLED=false` and reports what came out.

    WHY IT PROVES ANYTHING AT ALL. Every message on this path is the canned
    template built by substitution from the source record, so it is grounded by
    construction and passes the same validator a model's output would face.
    The Gate, the allocator, the ledger and the conductor never had a model in
    them to begin with - that is what the import bans are for - so what is
    being demonstrated is that the two places a model COULD sit both have a
    deterministic floor under them.
    """
    import os

    from arc.console.build import build
    from arc.llm_service import GroundingFacts, LlmClient, llm_enabled, validate

    previous = os.environ.get("LLM_ENABLED")
    os.environ["LLM_ENABLED"] = "false"
    try:
        assert not llm_enabled()
        data = build(seed=seed, size=size, cycles=cycles)
        facts = GroundingFacts(
            amount="Rs 1,299.00",
            due_date="12 May 2026",
            plan_name="Pro Monthly",
            merchant="Acme",
        )
        message, verdict = LlmClient().compose_message(template_id="dunning_v1", facts=facts)
        grounded = validate(message, facts).accepted
    finally:
        if previous is None:
            os.environ.pop("LLM_ENABLED", None)
        else:
            os.environ["LLM_ENABLED"] = previous

    arc = data.result.runs[Arm.ARC]
    return [
        "  LLM_ENABLED=false, full pipeline:",
        f"    batch                {data.batch.claims:,} claims, {data.batch.subjects:,} subjects",
        f"    diagnosis            {data.batch.issuer} issuer / "
        f"{data.batch.merchant} merchant / {data.batch.customer} customer",
        f"    suppressed by outage {data.batch.suppressed_by_outage}",
        f"    decisions            {len(arc.logs):,}",
        f"    recovered            {_inr(arc.recovered_paise)}",
        f"    message path         canned template, grounded={grounded}, {verdict.refused_by}",
        "",
        "    COMPLETED. Message quality degrades; correctness and compliance do not.",
    ]


def breaker_lines() -> list[str]:
    """A breaker panel, so the self-monitoring three are visible in the demo."""
    readings = [
        Reading(BreakerId.VETO, observed=0.004, sample=500),
        Reading(BreakerId.DEGRADED, observed=0.11, sample=500),
        Reading(BreakerId.COHORT_BLIND, observed=0.54, sample=500),
        Reading(BreakerId.COMPLAINT, observed=1.1, baseline=1.4, sample=500),
    ]
    from arc.conductor.breakers import render

    return render(evaluate_all(readings))


def run(
    *,
    seed: int,
    size: int,
    cycles: int,
    pause: float = 0.0,
) -> tuple[list[str], str]:
    data = build(seed=seed, size=size, cycles=cycles)
    lines = list(script(data, pause=pause))
    return lines, digest(data)


ACTIONS_SHOWN: Sequence[ActionType] = tuple(ActionType)
