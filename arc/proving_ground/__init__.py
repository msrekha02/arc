"""L8's measurement half: arms, the doubly-robust estimator, and the metrics.

Only `arms.py` exists at M5, because randomisation has to happen before any
treatment decision. Assigning an arm retrospectively - after seeing which
claims the policy chose to act on - does not measure a policy, it describes
one.

This package and `arc.simulator` are the only two allowed to read ground
truth. Everything else is banned from it by name in CI.
"""
