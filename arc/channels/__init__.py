"""L7 - Channels. Effect the world, decide nothing.

    contracts.py   the closed outcome set and the reversibility table
    effectors.py   one class, one status mapping, no conditionals
    provider.py    the fake provider: idempotent, failure-injecting, observed
    registry.py    the channel map the Conductor dispatches through

Deliberately the least clever package in the build. A channel receives a
payload and an idempotency key, hands them to a provider, and reports one of
seven structured outcomes. It never reads claim state, cause, amount, arm or
confidence, and `tests/test_channels.py` walks this package's AST to keep it
that way - by string literal as well as by identifier, because a subscript
reaches a domain concept just as well as an attribute does.

This package may not import `arc.simulator` and may not touch simulator ground
truth; both bans are enforced in CI.
"""

from arc.channels.contracts import (
    ACTION_REVERSIBILITY,
    CONSENT_AFFECTING_OUTCOMES,
    SUCCESSFUL_OUTCOMES,
    ChannelOutcome,
    ChannelPort,
    ChannelResult,
    Reversibility,
    UnknownProviderStatus,
    channel_reversibility,
    reversibility_of,
)
from arc.channels.effectors import (
    EFFECTOR_CHANNELS,
    STATUS_TO_OUTCOME,
    Effector,
    build_effector,
    outcome_for,
)
from arc.channels.provider import (
    DEFAULT_MIX,
    PROVIDER_STATUSES,
    AlwaysOutcome,
    FakeProvider,
    Provider,
    ProviderResponse,
    forced_mix,
)
from arc.channels.registry import build_channels, channel_for, coverage, fake_channels

__all__ = [
    "ACTION_REVERSIBILITY",
    "CONSENT_AFFECTING_OUTCOMES",
    "DEFAULT_MIX",
    "EFFECTOR_CHANNELS",
    "PROVIDER_STATUSES",
    "STATUS_TO_OUTCOME",
    "SUCCESSFUL_OUTCOMES",
    "AlwaysOutcome",
    "ChannelOutcome",
    "ChannelPort",
    "ChannelResult",
    "Effector",
    "FakeProvider",
    "Provider",
    "ProviderResponse",
    "Reversibility",
    "UnknownProviderStatus",
    "build_channels",
    "build_effector",
    "channel_for",
    "channel_reversibility",
    "coverage",
    "fake_channels",
    "forced_mix",
    "outcome_for",
    "reversibility_of",
]
