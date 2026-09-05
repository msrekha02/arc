"""The judged run's identity, in a leaf both the demo and the console can read.

WHY THIS IS NOT IN `arc/demo`. The demo harness imports `arc.console.build`, so
anything the console needs back from the demo is a circular import. The digest
is a fact ABOUT the judged run rather than a part of the demo's machinery, and
a constant that two packages must agree on belongs below both of them.

WHY IT IS A CONSTANT AND NOT A COMPUTATION. Recomputing it would mean running
the judged seed to find out what the judged seed produces, which is not a
check. It is pinned here, asserted by `test_the_judged_digest_is_pinned` against
the real command, and changed only in the same commit as the change that moves
it.
"""

from __future__ import annotations

JUDGED_DIGEST = "5c60e67cf45646afd4e5ff094a1890b98f26e3be1e31a42df0c621a1ae916bef"

JUDGED_DEMO_SIZE = 1_200
JUDGED_DEMO_CYCLES = 4
