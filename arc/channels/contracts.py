"""What a channel returns, and what can be taken back.

THIS LAYER IS DELIBERATELY STUPID. A channel receives a payload and an
idempotency key, hands them to a provider, and reports what happened. It does
not read claim state, cause, amount, arm or confidence, and it does not decide
anything - not whether to send, not what to send instead, not what to do next.
`tests/test_channels.py` walks the AST of this package and fails the build if a
branch condition so much as mentions a domain concept, by identifier OR by
string literal, because `payload["amount_paise"] > 100000` reaches the same
place through a subscript.

WHY THAT MATTERS MORE HERE THAN ANYWHERE ELSE. The effector is the last code
before the world. Policy that lives here executes without having passed the
Gate, cannot be replayed from the ledger, and cannot be tested without a
provider fixture. A compliance rule that only fires inside an SMS client is a
compliance rule nobody can audit.

THE CERTIFICATE IS NOT CHECKED HERE, and that is a decision rather than an
omission. GI-1 is asserted at the Conductor's dispatch boundary - Gate
touchpoint 3, the last code that runs before this one, mutation-verified at M9.
Re-checking it inside the effector would put compliance semantics into the one
layer this milestone exists to keep free of them, and the AST scan below would
be right to flag it.

OUTCOMES ARE STRUCTURED AND THE SET IS CLOSED. A boolean would be smaller and
would destroy the signal two later milestones depend on: M7 trains on the
difference between `bounced` and `wrong_number`, and M11's guardrail metrics
have no source for opt-out and complaint rates without them. Failure modes are
training data, not noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from arc.core.types import ActionType
from arc.gate.context import ACTION_CHANNEL, Channel


class ChannelOutcome(StrEnum):
    """The seven things that can happen to one dispatched action.

    Closed, and each member is a distinct fact that something downstream reads:

      DELIVERED     reached the handset, inbox or rail
      READ          the recipient opened it - engagement, not just delivery
      REPLIED       the recipient responded, which opens a service window
      BOUNCED       the address or number rejected it as unreachable
      WRONG_NUMBER  it reached somebody who is not the subject
      OPTED_OUT     the recipient withdrew consent, on this attempt or before
      FAILED        attempted and failed at the carrier for another reason

    WRONG_NUMBER IS NOT BOUNCED. A bounce means nobody received it; a wrong
    number means a stranger did, which is a third-party disclosure risk and
    triggers number suppression rather than a retry. Collapsing the two would
    lose exactly the distinction that decides whether it is safe to try again.

    OPTED_OUT IS NOT FAILED. An opt-out is a permanent consent change the Gate
    must honour forever after, and M11 reports its rate as a guardrail. A
    failure is transport noise.
    """

    DELIVERED = "delivered"
    READ = "read"
    REPLIED = "replied"
    BOUNCED = "bounced"
    WRONG_NUMBER = "wrong_number"
    OPTED_OUT = "opted_out"
    FAILED = "failed"


# Outcomes that mean the action reached its target. The rest are the signal.
SUCCESSFUL_OUTCOMES: frozenset[ChannelOutcome] = frozenset(
    {ChannelOutcome.DELIVERED, ChannelOutcome.READ, ChannelOutcome.REPLIED}
)

# Outcomes that change what the system may do next, and therefore have to reach
# the Gate rather than only a dashboard.
CONSENT_AFFECTING_OUTCOMES: frozenset[ChannelOutcome] = frozenset(
    {ChannelOutcome.OPTED_OUT, ChannelOutcome.WRONG_NUMBER}
)


class Reversibility(StrEnum):
    """Whether a dispatched action can still be taken back, and how.

    The Conductor needs this to answer one question: on a freeze, a hardship
    signal or a recovery landing mid-flight, what can still be pulled and what
    is already in the world?

      IRREVERSIBLE  the effect exists the moment dispatch returns. A read
                    message cannot be unread and a spoken call cannot be
                    unspoken.
      CANCELLABLE   dispatch schedules something that has not happened yet, so
                    it can be withdrawn before it takes effect. A debit queued
                    for presentation is the case that matters.
      REVOCABLE     the effect happened and can be undone afterwards. A payment
                    link can be deactivated, though the message announcing it
                    was already irreversibly sent.
    """

    IRREVERSIBLE = "irreversible"
    CANCELLABLE = "cancellable"
    REVOCABLE = "revocable"


# Declared per ACTION rather than per channel, because the finer grain is where
# the real differences are: a retry and a card-updater refresh both travel the
# silent rail, and only one of them is a debit waiting to be presented.
ACTION_REVERSIBILITY: Mapping[ActionType, Reversibility] = MappingProxyType(
    {
        # Nothing happened, so there is nothing to take back.
        ActionType.DO_NOTHING: Reversibility.CANCELLABLE,
        # Queued for presentation to the rail. Until the file goes, it can be
        # pulled - which is what makes a freeze meaningful for money movement.
        ActionType.RETRY: Reversibility.CANCELLABLE,
        ActionType.RAIL_FALLBACK: Reversibility.CANCELLABLE,
        ActionType.CARD_UPDATER: Reversibility.CANCELLABLE,
        ActionType.MANDATE_RE_REGISTER: Reversibility.CANCELLABLE,
        # Sent is sent.
        ActionType.WHATSAPP_UTILITY: Reversibility.IRREVERSIBLE,
        ActionType.SMS: Reversibility.IRREVERSIBLE,
        ActionType.EMAIL: Reversibility.IRREVERSIBLE,
        ActionType.VOICE_CALL: Reversibility.IRREVERSIBLE,
        ActionType.HUMAN_HANDOFF: Reversibility.IRREVERSIBLE,
        ActionType.STATUTORY_NOTICE: Reversibility.IRREVERSIBLE,
        # The message went and cannot be recalled, but the instrument it
        # carried can be deactivated, which is a different and useful answer.
        ActionType.PAYMENT_LINK: Reversibility.REVOCABLE,
        ActionType.INSTALMENT_OFFER: Reversibility.REVOCABLE,
    }
)


def reversibility_of(action: ActionType) -> Reversibility:
    return ACTION_REVERSIBILITY[action]


def channel_reversibility(channel: Channel) -> Reversibility:
    """The reversibility shared by every action on one channel.

    DERIVED, NEVER DECLARED TWICE. The build document's contract puts a
    `reversible` flag on the channel, and the finer truth is per action, so
    this reads the action table rather than restating it. If a channel ever
    carries actions that disagree, that is a real modelling problem and it
    raises here instead of one of the two answers quietly winning.
    """
    found = {
        ACTION_REVERSIBILITY[action]
        for action, mapped in ACTION_CHANNEL.items()
        if mapped is channel
    }
    if not found:
        raise KeyError(f"no action maps to {channel}")
    if len(found) > 1:
        raise ValueError(
            f"{channel} carries actions with different reversibility {sorted(found)}; "
            "the Conductor cannot be told one answer for both"
        )
    return found.pop()


class UnknownProviderStatus(ValueError):
    """A provider said something the outcome vocabulary does not cover.

    Raised rather than mapped to `failed`, because an unrecognised status is
    not a known failure - it is a gap between the provider's vocabulary and
    ours, and silently folding it into the residual bucket is how a new
    delivery state disappears from the guardrail metrics for a year (GI-5).
    """


@dataclass(frozen=True)
class ChannelResult:
    """What one dispatch produced. Consumed by the ledger, M7 and M11."""

    outcome: ChannelOutcome
    channel: Channel
    idempotency_key: str
    provider_reference: str | None = None
    deduplicated: bool = False
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ChannelOutcome):
            raise TypeError(f"{self.outcome!r} is not a ChannelOutcome")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def reached_target(self) -> bool:
        return self.outcome in SUCCESSFUL_OUTCOMES

    @property
    def affects_consent(self) -> bool:
        return self.outcome in CONSENT_AFFECTING_OUTCOMES


@runtime_checkable
class ChannelPort(Protocol):
    """The effector, as the Conductor sees it.

    Two arguments and one return. No claim, no cause, no state - a channel that
    could see those could branch on them.
    """

    channel: Channel
    reversibility: Reversibility

    async def send(self, payload: Mapping[str, Any], idempotency_key: str) -> ChannelResult: ...
