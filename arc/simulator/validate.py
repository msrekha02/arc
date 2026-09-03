"""`python -m arc.simulator.validate` - the simulator's own evidence.

A synthetic world is only worth something if its distributions can be checked
against something outside it. This prints what the generator produced next to
the published figure it was calibrated against, with the delta, and exits
non-zero if any metric has drifted outside its stated tolerance.

Two tables, kept apart on purpose:

  DISTRIBUTIONS   generated behaviour against an external published anchor.
                  These are the ones that make the world defensible
                  independently of the policy.

  INJECTED        structure this milestone deliberately planted, against the
                  rate it was specified at. These are self-referential and are
                  labelled as such - quoting our own constant back as though
                  it were external evidence would be worthless.

Tolerances are wide where the published figure is itself a range. A metric
that only passes on a knife edge is a metric that was fitted.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

from arc.core.types import ActionType, Rail
from arc.simulator import response_model as rm
from arc.simulator.codes import Semantic, code_for
from arc.simulator.seeds import DEVELOP_SEED, EPOCH, Stream, rng
from arc.simulator.wire_fake import WireFake, arrival_inversions, late_deliveries
from arc.simulator.world import (
    LAST_WORKING_DAY,
    ORPHANED_MANDATE_RATE,
    STALE_PHONE_RATE,
    EventKind,
    World,
)

VALIDATE_SIZE = 12_000
_SECRET = b"arc-simulator-validate-secret-key"


@dataclass(frozen=True, slots=True)
class Benchmark:
    """One published anchor, and what it is anchored to."""

    key: str
    label: str
    published: float
    tolerance: float
    unit: str
    instrument: str


@dataclass(frozen=True, slots=True)
class Injected:
    """One deliberately planted structure and the rate it was specified at."""

    key: str
    label: str
    specified: float
    tolerance: float
    unit: str


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        "nach_decline_rate",
        "eNACH first-presentation return rate",
        0.32,
        0.07,
        "share",
        "NPCI monthly NACH debit statistics - returns as a share of presentations",
    ),
    Benchmark(
        "card_decline_rate",
        "Card recurring decline rate",
        0.16,
        0.07,
        "share",
        "Card-network and PSP reporting on recurring authorisation rates",
    ),
    Benchmark(
        "upi_decline_rate",
        "UPI Autopay debit failure rate",
        0.24,
        0.09,
        "share",
        "NPCI UPI Autopay decline reporting",
    ),
    Benchmark(
        "funds_share_of_declines",
        "Insufficient funds as a share of declines",
        0.60,
        0.15,
        "share",
        "NPCI NACH return-reason mix, where funds-insufficient dominates",
    ),
    Benchmark(
        "hard_decline_share",
        "Hard declines as a share of declines",
        0.10,
        0.07,
        "share",
        "Card return-reason mix - lost, stolen, closed and stop-payment",
    ),
    Benchmark(
        "payday_cluster_share",
        "Salary credited on the 1st or last working day",
        0.60,
        0.12,
        "share",
        "Indian payroll convention for monthly salary credit dates",
    ),
    Benchmark(
        "natural_recovery",
        "Untouched claim recovers on its own",
        0.16,
        0.08,
        "share",
        "Involuntary-churn baselines for failed recurring charges",
    ),
    Benchmark(
        "retry_recovery",
        "Cause-blind retry recovers the claim",
        0.32,
        0.10,
        "share",
        "PSP re-presentment success reporting for failed recurring debits",
    ),
    Benchmark(
        "opt_out_per_1000",
        "Opt-outs per 1000 contacts",
        8.0,
        6.0,
        "per 1000",
        "Messaging opt-out benchmarks for transactional and utility traffic",
    ),
    Benchmark(
        "complaint_per_1000",
        "Complaints per 1000 contacts",
        2.0,
        2.5,
        "per 1000",
        "Collections-conduct guidance and card-network complaint thresholds",
    ),
)

INJECTED_STRUCTURE: tuple[Injected, ...] = (
    Injected(
        "orphan_share", "Silently orphaned mandate cohort", ORPHANED_MANDATE_RATE, 0.012, "share"
    ),
    Injected("stale_phone_share", "Stale contact numbers", STALE_PHONE_RATE, 0.012, "share"),
    Injected("remap_share", "Wrong or remapped decline codes", 0.05, 0.02, "share"),
    Injected("duplicate_share", "Duplicate webhook deliveries", 0.02, 0.005, "share"),
    Injected("late_share", "Late, out-of-order deliveries", 0.03, 0.005, "share"),
)


def measure(seed: int = DEVELOP_SEED, size: int = VALIDATE_SIZE) -> dict[str, float]:
    """Every metric in one pass over a generated world.

    Uses ground truth freely: this is the evaluation harness, which is the one
    place allowed to read the answer key.
    """
    world = World(seed=seed, size=size)
    events = world.batch_events()
    presentations = [e for e in events if e.kind is EventKind.PRESENTATION and e.attempt == 1]

    def decline_rate(rail: Rail) -> float:
        rows = [e for e in presentations if e.rail is rail]
        return sum(1 for e in rows if not e.succeeded) / len(rows) if rows else 0.0

    declined = [e for e in presentations if not e.succeeded and e.true_semantic]
    reasons = Counter(e.true_semantic for e in declined)
    total_declined = max(len(declined), 1)

    accounts = [world._account(account_id) for account_id in world.account_ids]
    payday = sum(1 for a in accounts if a.latent.salary_day in (LAST_WORKING_DAY, 1)) / len(
        accounts
    )

    # Ground-truth recovery, averaged over the claim population - the accounts
    # whose first presentation actually failed.
    failed = [e for e in presentations if not e.succeeded]
    sample = [e.account_id for e in failed][:2000]
    natural = sum(
        world.counterfactual(account_id, ActionType.DO_NOTHING, EPOCH) for account_id in sample
    ) / max(len(sample), 1)

    # Re-presentment success is published over the RETRYABLE population: a
    # lost card is never re-presented, so including it would compare our
    # number against a different denominator than the source used.
    retryable = [
        e.account_id
        for e in failed
        if e.true_semantic not in (Semantic.HARD_DECLINE, Semantic.DO_NOT_RETRY)
    ][:2000]
    retry = sum(
        world.counterfactual(account_id, ActionType.RETRY, EPOCH) for account_id in retryable
    ) / max(len(retryable), 1)

    # Opt-out and complaint benchmarks are published per message, on
    # populations that have not just been contacted three times. Measuring on
    # accounts with a clear seven-day window compares like with like; the
    # escalation with contact pressure is a separate, much larger number.
    uncontacted = [e.account_id for e in failed if world.contacts_7d(e.account_id, EPOCH) == 0][
        :4000
    ]
    harm = _contact_harm(world, uncontacted, seed)

    # The injected remap only: the emitted code differs from the code this
    # rail would truthfully carry for that reason. A declared collapse is not
    # a remap and must not be counted as one.
    codeable = [e for e in declined if code_for(e.rail, e.true_semantic) is not None]
    remapped = sum(1 for e in codeable if e.decline_code != code_for(e.rail, e.true_semantic).code)

    fake = WireFake(world, _SECRET)
    deliveries = fake.emit("replay", seed)
    ids = Counter(d.event_id for d in deliveries)

    return {
        "nach_decline_rate": decline_rate(Rail.ENACH),
        "card_decline_rate": decline_rate(Rail.CARD),
        "upi_decline_rate": decline_rate(Rail.UPI_AUTOPAY),
        "funds_share_of_declines": reasons[Semantic.INSUFFICIENT_FUNDS] / total_declined,
        "hard_decline_share": (reasons[Semantic.HARD_DECLINE] + reasons[Semantic.DO_NOT_RETRY])
        / total_declined,
        "payday_cluster_share": payday,
        "natural_recovery": natural,
        "retry_recovery": retry,
        "opt_out_per_1000": harm["opt_out"],
        "complaint_per_1000": harm["complaint"],
        "orphan_share": sum(1 for a in accounts if a.mandate_orphaned) / len(accounts),
        "stale_phone_share": sum(1 for a in accounts if a.latent.phone_stale) / len(accounts),
        "remap_share": remapped / max(len(codeable), 1),
        "duplicate_share": sum(1 for count in ids.values() if count > 1) / len(ids),
        "late_share": len(late_deliveries(deliveries)) / len(deliveries),
        "_events": float(len(events)),
        "_deliveries": float(len(deliveries)),
        "_inversions": float(arrival_inversions(deliveries)),
        "_claims": float(sum(1 for e in events if not e.succeeded)),
    }


def _semantic_of(rail: Rail, code: str) -> Semantic | None:
    from arc.simulator.codes import codes_for

    for entry in codes_for(rail):
        if entry.code == code:
            return entry.semantic
    return None


def _contact_harm(world: World, account_ids: list[str], seed: int) -> dict[str, float]:
    """One WhatsApp each, on a fresh world, counting what it cost.

    A fork rather than the live world: contacts accumulate, and measuring the
    first contact's harm on a population already contacted would report the
    wrong number.
    """
    twin = world.fork()
    generator = rng(seed, Stream.VALIDATE)
    counts: Counter[str] = Counter()
    for account_id in account_ids:
        outcome = twin.outcome(account_id, ActionType.WHATSAPP_UTILITY, EPOCH, generator)
        counts[str(outcome.kind)] += 1
    total = max(len(account_ids), 1)
    return {
        "opt_out": 1000.0 * counts[str(rm.OutcomeKind.OPT_OUT)] / total,
        "complaint": 1000.0 * counts[str(rm.OutcomeKind.COMPLAINT)] / total,
    }


def _row(label: str, actual: float, reference: float, tolerance: float, unit: str) -> str:
    delta = actual - reference
    verdict = "ok" if abs(delta) <= tolerance else "DRIFT"
    if unit == "share":
        return (
            f"  {label:<44s} {actual * 100:7.2f}%  {reference * 100:7.2f}%  "
            f"{delta * 100:+7.2f}pp  +/-{tolerance * 100:5.2f}pp  {verdict}"
        )
    return (
        f"  {label:<44s} {actual:8.2f}  {reference:8.2f}  "
        f"{delta:+8.2f}  +/-{tolerance:6.2f}  {verdict}"
    )


def report(seed: int = DEVELOP_SEED, size: int = VALIDATE_SIZE) -> tuple[str, bool]:
    """The full report and whether everything held."""
    measured = measure(seed, size)
    lines: list[str] = []
    passed = True

    lines.append(f"ARC world simulator - distribution validation (seed {seed}, {size} accounts)")
    lines.append("")
    lines.append(
        f"  generated {int(measured['_events'])} batch events, "
        f"{int(measured['_claims'])} of them failures, "
        f"delivered as {int(measured['_deliveries'])} webhooks with "
        f"{int(measured['_inversions'])} arrival inversions"
    )
    lines.append("")
    lines.append("DISTRIBUTIONS vs PUBLISHED BENCHMARK")
    lines.append(f"  {'metric':<44s} {'generated':>8s}  {'published':>8s}  {'delta':>9s}")
    for benchmark in BENCHMARKS:
        actual = measured[benchmark.key]
        if abs(actual - benchmark.published) > benchmark.tolerance:
            passed = False
        lines.append(
            _row(benchmark.label, actual, benchmark.published, benchmark.tolerance, benchmark.unit)
        )

    lines.append("")
    lines.append("INJECTED STRUCTURE vs SPECIFICATION")
    lines.append("  (self-referential by construction - these check the planting, not the world)")
    for injected in INJECTED_STRUCTURE:
        actual = measured[injected.key]
        if abs(actual - injected.specified) > injected.tolerance:
            passed = False
        lines.append(
            _row(injected.label, actual, injected.specified, injected.tolerance, injected.unit)
        )

    lines.append("")
    lines.append("SOURCES")
    for benchmark in BENCHMARKS:
        lines.append(f"  {benchmark.label}")
        lines.append(f"      {benchmark.instrument}")

    lines.append("")
    lines.append("PASS - every metric inside tolerance" if passed else "DRIFT - see rows above")
    return "\n".join(lines), passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the simulated world's distributions.")
    parser.add_argument("--seed", type=int, default=DEVELOP_SEED)
    parser.add_argument("--size", type=int, default=VALIDATE_SIZE)
    args = parser.parse_args(argv)

    text, passed = report(args.seed, args.size)
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
