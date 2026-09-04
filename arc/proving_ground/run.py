"""The judged run.

    python -m arc.proving_ground.run --seed 3 --once

SEED DISCIPLINE. Three seeds, announced in advance: develop on 1, tune on 2,
run seed 3 ONCE, live. `--once` is a declaration rather than a mechanism - it
prints the seed's role and refuses to pretend a judged run is a rehearsal. A
system tuned on the seed it is evaluated against has been fitted to its own
evaluation set, and announcing which is which beforehand is what makes the
final number a measurement rather than a selection.

WHAT IT PRINTS, AND WHY IN THAT ORDER. The scoreboard first, with every arm's
guardrails on the same row as its money - the metrics object refuses to
serialise them apart, so there is no arrangement of this output that shows the
headline alone. Then prevention, on its own line. Then the estimator's own
error against the simulator's ground truth, which is the number that says
whether to believe the rest.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from arc.gate.evaluator import Gate
from arc.gate.registry import load_registry
from arc.proving_ground.arms import Arm
from arc.proving_ground.dr_estimator import (
    dr_estimate,
    fit_outcome_model,
    ground_truth_value,
    on_policy_target,
)
from arc.proving_ground.harness import build_scoreboard, run_all
from arc.simulator.seeds import DEVELOP_SEED, JUDGED_SEED, TUNE_SEED

SEED_ROLE = {
    DEVELOP_SEED: "DEVELOP - iterate on this one freely",
    TUNE_SEED: "TUNE - tune on this one freely",
    JUDGED_SEED: "JUDGED - run once, live, and report whatever it says",
}


def main(argv: list[str] | None = None) -> int:
    # The scoreboard prints rupee amounts, and a Windows console defaults to a
    # code page that cannot encode the rupee sign. Without this the judged run
    # dies on its own output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="arc.proving_ground.run")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--size", type=int, default=1_800)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument(
        "--once",
        action="store_true",
        help="declare this the judged run; prints the seed's announced role",
    )
    parser.add_argument("--json", action="store_true", help="emit the scoreboard as JSON")
    args = parser.parse_args(argv)

    role = SEED_ROLE.get(args.seed, "UNANNOUNCED - not one of the three declared seeds")
    if args.once:
        print(f"seed {args.seed}: {role}")
        if args.seed != JUDGED_SEED:
            print("  note: --once was passed with a seed that is not the judged one")
        print()

    gate = Gate(load_registry())
    result = run_all(seed=args.seed, size=args.size, cycles=args.cycles, gate=gate)

    logs = result.runs[Arm.ARC].logs
    q_hat = fit_outcome_model(logs)
    estimate = dr_estimate(logs, q_hat, on_policy_target, rng=np.random.default_rng(args.seed))
    truth = ground_truth_value(logs, on_policy_target)
    error = estimate.relative_error(truth)

    scoreboard = build_scoreboard(result, dr_relative_error=error)

    if args.json:
        print(json.dumps(scoreboard.to_dict(), indent=2))
        return 0

    for line in scoreboard.render():
        print(line)

    print()
    print("doubly-robust estimator, validated against simulator ground truth:")
    print(f"  estimate       {estimate.point:>14,.0f} paise per decision")
    print(f"  ground truth   {truth:>14,.0f} paise per decision")
    print(f"  relative error {error * 100:>13.2f}%")
    print(f"  95% CI         [{estimate.lo:,.0f}, {estimate.hi:,.0f}]")
    print(f"  covers truth   {estimate.covers(truth)}")
    print(
        f"  rows {estimate.n_rows}, subjects {estimate.n_subjects}, "
        f"clipped {estimate.clipped_share:.1%}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
