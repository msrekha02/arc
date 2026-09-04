"""The three demo modes.

    python -m arc.demo.run --seed 3            deterministic replay
    python -m arc.demo.run --live              real-time, jittered
    python -m arc.demo.run --adversarial       the attack suite

REPLAY READS NO CLOCK AND PRINTS NO WALL TIME. Three consecutive runs must
produce byte-identical output, so anything that varies between runs - a
timestamp, an elapsed figure, a pause that depends on how fast the machine is -
is either injected or absent. The pauses in a narrated run are the one
exception, and they print nothing.
"""

from __future__ import annotations

import argparse
import sys

from arc.demo.harness import (
    NARRATION_PAUSE,
    adversarial_lines,
    breaker_lines,
    run,
)
from arc.simulator.seeds import JUDGED_SEED


def main(argv: list[str] | None = None) -> int:
    # The scoreboard prints rupee amounts and a Windows console defaults to a
    # code page that cannot encode the rupee sign.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(prog="arc.demo.run")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--size", type=int, default=1_200)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--live", action="store_true", help="real-time, jittered")
    parser.add_argument("--adversarial", action="store_true", help="the attack suite")
    parser.add_argument("--narrate", action="store_true", help="pause between beats")
    parser.add_argument("--digest-only", action="store_true")
    args = parser.parse_args(argv)

    if args.adversarial:
        print("=" * 72)
        print("  ADVERSARIAL SUITE - every attack goes through the real path")
        print("=" * 72)
        print()
        for line in adversarial_lines():
            print(line)
        print()
        print("  circuit breakers, including the three that watch the machinery:")
        for line in breaker_lines():
            print(line)
        return 0

    pause = NARRATION_PAUSE if (args.narrate or args.live) else 0.0
    lines, sha = run(seed=args.seed, size=args.size, cycles=args.cycles, pause=pause)

    if args.digest_only:
        print(sha)
        return 0

    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
