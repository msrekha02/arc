"""Everything the Gate is allowed to know, and the only way it learns anything.

The Gate performs no I/O, so every fact a rule could need is a field here:
consent, contact history, retry history, subject flags, the timezone basis, the
bank-holiday calendar. If a rule wants something that is not in this object,
the answer is to widen the context, never to let the Gate fetch it.

WHY that is worth the verbosity: a rule that reads a database is a rule whose
verdict cannot be reproduced six months later during a replay, and replay is
the whole reason the audit trail is worth having.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from enum import StrEnum
from types import MappingProxyType
from uuid import UUID

from arc.core.money import Paise
from arc.core.time_authority import TimezoneBasis, ensure_utc
from arc.core.types import ActionType, Cause, ClaimState, Rail


class Channel(StrEnum):
    """How an action reaches the world, which is what rule scope keys on."""

    NONE = "none"  # do_nothing
    SILENT = "silent"  # rail-level, zero customer contact
    VOICE = "voice"
    WHATSAPP = "whatsapp"
    SMS = "sms"
    EMAIL = "email"
    PAYMENT_LINK = "payment_link"
    INSTALMENT = "instalment"
    HUMAN = "human"
    POSTAL = "postal"


ACTION_CHANNEL: Mapping[ActionType, Channel] = MappingProxyType(
    {
        ActionType.DO_NOTHING: Channel.NONE,
        ActionType.RETRY: Channel.SILENT,
        ActionType.CARD_UPDATER: Channel.SILENT,
        ActionType.MANDATE_RE_REGISTER: Channel.SILENT,
        ActionType.RAIL_FALLBACK: Channel.SILENT,
        ActionType.WHATSAPP_UTILITY: Channel.WHATSAPP,
        ActionType.SMS: Channel.SMS,
        ActionType.EMAIL: Channel.EMAIL,
        ActionType.PAYMENT_LINK: Channel.PAYMENT_LINK,
        ActionType.VOICE_CALL: Channel.VOICE,
        ActionType.INSTALMENT_OFFER: Channel.INSTALMENT,
        ActionType.HUMAN_HANDOFF: Channel.HUMAN,
        ActionType.STATUTORY_NOTICE: Channel.POSTAL,
    }
)

# Channels that reach a person. Cooldowns and frequency caps count these and
# nothing else, which is why a rail-level retry does not spend a contact slot.
CONTACT_CHANNELS: frozenset[Channel] = frozenset(
    {
        Channel.VOICE,
        Channel.WHATSAPP,
        Channel.SMS,
        Channel.EMAIL,
        Channel.PAYMENT_LINK,
        Channel.INSTALMENT,
        Channel.HUMAN,
        Channel.POSTAL,
    }
)

# Channels whose legality depends on the subject's local wall time.
TIME_BOUNDED_CHANNELS: frozenset[Channel] = CONTACT_CHANNELS - {Channel.POSTAL}

# Actions that present a debit to a rail.
MONEY_MOVING_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.RETRY, ActionType.RAIL_FALLBACK}
)

# Actions that state account detail to a person, so identity has to be verified
# before they run.
DISCLOSING_ACTIONS: frozenset[ActionType] = frozenset(
    {
        ActionType.VOICE_CALL,
        ActionType.HUMAN_HANDOFF,
        ActionType.INSTALMENT_OFFER,
        ActionType.STATUTORY_NOTICE,
    }
)


class ConsentState(StrEnum):
    GRANTED = "granted"
    WITHDRAWN = "withdrawn"
    NEVER_GIVEN = "never_given"
    # Missing or unreadable. Distinct from NEVER_GIVEN so the audit trail can
    # tell "we asked and they said no" apart from "we could not find out".
    UNKNOWN = "unknown"


class TargetRelationship(StrEnum):
    OBLIGOR = "obligor"
    GUARANTOR = "guarantor"
    THIRD_PARTY = "third_party"
    EMPLOYER = "employer"
    UNKNOWN = "unknown"


class ContactOutcome(StrEnum):
    DELIVERED = "delivered"
    READ = "read"
    REPLIED = "replied"
    CONNECTED = "connected"
    NO_ANSWER = "no_answer"
    BOUNCED = "bounced"
    WRONG_NUMBER = "wrong_number"
    OPTED_OUT = "opted_out"
    FAILED = "failed"


class DeclineCategory(StrEnum):
    """Category 1 is the do-not-retry family: retrying is network-punished."""

    SOFT = "soft"
    LOST_OR_STOLEN = "lost_or_stolen"
    ACCOUNT_CLOSED = "account_closed"
    STOP_PAYMENT = "stop_payment"
    DO_NOT_HONOUR = "do_not_honour"
    NONE = "none"


HARD_DECLINE_CATEGORIES: frozenset[DeclineCategory] = frozenset(
    {
        DeclineCategory.LOST_OR_STOLEN,
        DeclineCategory.ACCOUNT_CLOSED,
        DeclineCategory.STOP_PAYMENT,
    }
)

# Advice codes that instruct the merchant to stop presenting.
DO_NOT_RETRY_CODES: frozenset[str] = frozenset({"MAC03", "MD06", "R08", "R16", "AC03"})


class RetryInitiator(StrEnum):
    """The gateway retries on its own schedule and those attempts still count.

    Omitting them makes the attempt counter wrong and lets the network cap be
    exceeded by attempts the system never issued itself.
    """

    ARC = "arc"
    GATEWAY = "gateway"


@dataclass(frozen=True)
class ContactEvent:
    at: datetime
    channel: Channel
    outcome: ContactOutcome

    def __post_init__(self) -> None:
        ensure_utc(self.at)


@dataclass(frozen=True)
class RetryEvent:
    at: datetime
    rail: Rail
    initiated_by: RetryInitiator = RetryInitiator.ARC

    def __post_init__(self) -> None:
        ensure_utc(self.at)


@dataclass(frozen=True)
class SubjectFlags:
    """State-based freezes and absolute prohibitions, as plain facts.

    Each freeze that has an end carries it, because a freeze without a
    computable end is a BLOCK and must be reported as one.
    """

    forborne: bool = False
    minor: bool = False
    erasure_requested: bool = False
    hardship: bool = False
    complaint_open: bool = False
    disputed: bool = False
    identity_verified: bool = False

    ptp_active: bool = False
    ptp_freeze_until: datetime | None = None

    payment_pending: bool = False
    payment_pending_until: datetime | None = None

    issuer_degraded: bool = False
    issuer_degraded_until: datetime | None = None


@dataclass(frozen=True)
class GateContext:
    """A complete, self-contained snapshot. Passed in; never fetched."""

    claim_id: UUID
    subject_token: str
    rail: Rail
    claim_state: ClaimState
    amount_paise: Paise

    target: TargetRelationship = TargetRelationship.OBLIGOR
    tz_basis: TimezoneBasis | None = None
    region: str = "IN"

    cause: Cause | None = None
    mandate_cap_paise: Paise | None = None
    predebit_notice_at: datetime | None = None
    decline_category: DeclineCategory = DeclineCategory.NONE
    advice_code: str | None = None

    consent: Mapping[Channel, ConsentState] = field(default_factory=dict)
    opted_out_channels: frozenset[Channel] = frozenset()

    quiet_hours: tuple[time, time] | None = None
    bank_holidays: frozenset[date] = frozenset()

    contacts: tuple[ContactEvent, ...] = ()
    retries: tuple[RetryEvent, ...] = ()

    # Set at touchpoints 3 and 4, where an already-issued certificate is being
    # re-checked rather than a new one issued.
    certificate_valid_until: datetime | None = None
    flags: SubjectFlags = field(default_factory=SubjectFlags)

    def __post_init__(self) -> None:
        object.__setattr__(self, "consent", MappingProxyType(dict(self.consent)))
        object.__setattr__(self, "contacts", tuple(self.contacts))
        object.__setattr__(self, "retries", tuple(self.retries))

    def channel_for(self, action: ActionType) -> Channel:
        return ACTION_CHANNEL[action]

    def consent_for(self, channel: Channel) -> ConsentState:
        """Absent means UNKNOWN, never GRANTED. Missing consent fails closed."""
        return self.consent.get(channel, ConsentState.UNKNOWN)
