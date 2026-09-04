"""Assembling the channel map the Conductor dispatches through.

The Conductor looks a channel up by the string on the outbox row and calls
`send`. This builds that mapping and nothing else - there is no routing logic,
because choosing a channel is choosing an action, and choosing an action is the
Allocator's job.

A channel absent from the map is not a silent no-op. `worker.dispatch` marks
the row dead and frees its budget, which is the correct answer to "we have an
authorised action and no way to perform it".
"""

from __future__ import annotations

from collections.abc import Mapping

from arc.channels.effectors import EFFECTOR_CHANNELS, Effector, build_effector
from arc.channels.provider import FakeProvider, Provider
from arc.core.types import ActionType
from arc.gate.context import ACTION_CHANNEL, Channel


def build_channels(provider: Provider) -> dict[str, Effector]:
    """Every carrying channel, keyed by the value the outbox row holds."""
    return {
        channel.value: build_effector(channel, provider)
        for channel in dict.fromkeys(EFFECTOR_CHANNELS)
    }


def fake_channels(**provider_options: object) -> tuple[dict[str, Effector], FakeProvider]:
    """A full channel map over one recording provider.

    Returns the provider as well, because every interesting assertion is about
    what it saw: which keys arrived, how often, and what came back.
    """
    provider = FakeProvider(**provider_options)  # type: ignore[arg-type]
    return build_channels(provider), provider


def channel_for(action: ActionType) -> Channel:
    """Which transport an action travels on. A lookup into the Gate's table."""
    return ACTION_CHANNEL[action]


def coverage(channels: Mapping[str, object]) -> list[ActionType]:
    """Actions with no effector behind them.

    A gap here means the Allocator can authorise something the Conductor cannot
    perform, which surfaces as dead rows rather than as an error, so it is
    worth being able to ask directly.
    """
    return [
        action
        for action, channel in ACTION_CHANNEL.items()
        if channel is not Channel.NONE and channel.value not in channels
    ]
