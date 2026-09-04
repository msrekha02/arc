"""The effectors. One class, one table, no decisions.

EVERY CHANNEL IS THE SAME CLASS. There is no `SmsChannel` with its own
retry rule and no `VoiceChannel` that knows about consent, because the moment
one of them acquires a special case, that case is policy living outside the
Gate. What differs between channels is the transport name and the reversibility
they inherit from the actions that route to them, and both of those are data.

THE PROVIDER'S VOCABULARY IS MAPPED, NOT BRANCHED ON. `STATUS_TO_OUTCOME` is a
dictionary lookup, so there is no conditional here at all for the AST scan to
object to and no place for a rule to grow. An unrecognised status raises rather
than falling back to `failed`: a new delivery state the vendor added last week
is a gap in the mapping, not a known failure, and folding it into the residual
bucket is how it disappears from the guardrail metrics for a year (GI-5).

WHAT THIS MODULE MAY NOT DO, enforced by the AST scan in
`tests/test_channels.py`: branch on claim state, cause, amount, LTV, arm or
confidence, by identifier or by string literal. It also does not validate the
payload. The payload was assembled upstream by code that was allowed to know
what the fields mean; a required-field check here would be a branch on
`"amount_paise"`, which is the scan's point exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from arc.channels.contracts import (
    ChannelOutcome,
    ChannelResult,
    Reversibility,
    UnknownProviderStatus,
    channel_reversibility,
)
from arc.channels.provider import Provider
from arc.gate.context import ACTION_CHANNEL, Channel

# The seam between the vendor's words and the domain's. Every provider status
# has exactly one outcome, and every outcome except the ones a provider cannot
# observe has at least one status.
STATUS_TO_OUTCOME: Mapping[str, ChannelOutcome] = MappingProxyType(
    {
        "accepted": ChannelOutcome.DELIVERED,
        "opened": ChannelOutcome.READ,
        "responded": ChannelOutcome.REPLIED,
        "undeliverable": ChannelOutcome.BOUNCED,
        "reached_third_party": ChannelOutcome.WRONG_NUMBER,
        "recipient_unsubscribed": ChannelOutcome.OPTED_OUT,
        "carrier_rejected": ChannelOutcome.FAILED,
    }
)


def outcome_for(status: str) -> ChannelOutcome:
    """One lookup. Unknown fails closed rather than becoming `failed`."""
    try:
        return STATUS_TO_OUTCOME[status]
    except KeyError:
        raise UnknownProviderStatus(
            f"provider status {status!r} has no outcome in the closed set "
            f"{sorted(ChannelOutcome)}; mapping it to `failed` would hide a new "
            "delivery state from every guardrail that reads these"
        ) from None


@dataclass
class Effector:
    """One channel. Hands the payload to the provider and reports back.

    Holds no state about any claim and makes no choice. The only thing it knows
    is which transport it is and what the Conductor may take back afterwards.
    """

    channel: Channel
    provider: Provider
    reversibility: Reversibility

    async def send(self, payload: Mapping[str, Any], idempotency_key: str) -> ChannelResult:
        response = await self.provider.deliver(self.channel.value, payload, idempotency_key)
        return ChannelResult(
            outcome=outcome_for(response.status),
            channel=self.channel,
            idempotency_key=idempotency_key,
            provider_reference=response.reference,
            deduplicated=response.deduplicated,
            detail=response.detail,
            metadata={"provider_status": response.status},
        )


# Channels that actually carry something. `NONE` is `do_nothing`, which never
# reaches an effector at all - the Conductor has nothing to dispatch.
EFFECTOR_CHANNELS: tuple[Channel, ...] = tuple(
    channel for channel in ACTION_CHANNEL.values() if channel is not Channel.NONE
)


def build_effector(channel: Channel, provider: Provider) -> Effector:
    return Effector(
        channel=channel,
        provider=provider,
        reversibility=channel_reversibility(channel),
    )
