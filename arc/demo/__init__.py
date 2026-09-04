"""M17 - the demo harness. Sequencing what the console renders.

    harness.py   the nine beats, the digest, the three run modes
    attacks.py   the adversarial suite, each attack through the real path
    run.py       the CLI behind `make demo`

The demo shows the console's numbers rather than computing its own. A harness
with a second source of truth is a harness that can disagree with the system it
is demonstrating, on stage.
"""

from arc.demo.attacks import ATTACKS, Attack, Outcome, run_attack
from arc.demo.harness import Beat, adversarial_lines, beats, digest, headline_numbers, run, script

__all__ = [
    "ATTACKS",
    "Attack",
    "Beat",
    "Outcome",
    "adversarial_lines",
    "beats",
    "digest",
    "headline_numbers",
    "run",
    "run_attack",
    "script",
]
