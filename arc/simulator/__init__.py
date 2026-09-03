"""The world simulator and the wire-level fake.

FROZEN at `simulator-frozen-v1`. `world.py` and `response_model.py` are not
modified after that tag, and the commit timestamp is the evidence: a world
tuned after seeing policy results measures the policy against itself.

This package MUST NOT import from `arc.allocator`, `arc.forecaster` or
`arc.gate`. The simulated world does not know about the policy that will be
measured against it. Enforced by `tests/test_import_bans.py`.
"""
