"""M16 - the voice conversation, and why it is not in `channels/`.

M10's rule is that a channel effector must contain zero decision logic: it is
the last code before the world, so a rule living there runs without having
passed the Gate and cannot be reconstructed from the ledger. That rule is
right, and M10's AST scanner enforces it by refusing any branch on claim state,
cause, amount or confidence.

A VOICE CONVERSATION BRANCHES ON ALL OF THOSE, BY CONSTRUCTION. It has to: the
six non-removable rules fire mid-call, on a turn the Gate never saw and could
not have seen, because the turn had not happened when the certificate was
issued. Verification gates disclosure. Distress ends the call. Confidence gates
whether a promise is recorded.

    SO IT IS NOT AN EFFECTOR AND IT DOES NOT LIVE WITH THEM. Putting it in
    `channels/` would make M10's scanner right to complain, and the only ways
    to quiet it would be to exempt the file - widening a ban to cover the one
    case it was written for - or to weaken the forbidden-name list for
    everybody. Both are worse than a package boundary.

WHAT THIS PACKAGE IS: a bounded state machine that DRIVES an effector. The
effector still places the call and still contains no decisions. The
conversation decides what to say next from a closed set, and stops.
"""

from arc.voice.conversation import (
    ALLOWED,
    DISCLOSURE,
    FORBIDDEN_CONFIG_FIELDS,
    MAY_DISCLOSE,
    NON_REMOVABLE,
    TRANSACTIONAL_SERIES,
    VOX_CLI,
    VOX_DISCLOSE,
    VOX_DISTRESS,
    VOX_RECORD,
    VOX_VERIFY,
    VOX_WRONG_PARTY,
    CallOutcome,
    ConfigurationCanDisableRule,
    DisclosureViolation,
    PromiseRecord,
    Turn,
    VoiceCall,
    VoiceConfig,
    VoiceState,
    assert_no_configuration_can_disable,
)

__all__ = [
    "ALLOWED",
    "DISCLOSURE",
    "FORBIDDEN_CONFIG_FIELDS",
    "MAY_DISCLOSE",
    "NON_REMOVABLE",
    "TRANSACTIONAL_SERIES",
    "VOX_CLI",
    "VOX_DISCLOSE",
    "VOX_DISTRESS",
    "VOX_RECORD",
    "VOX_VERIFY",
    "VOX_WRONG_PARTY",
    "CallOutcome",
    "ConfigurationCanDisableRule",
    "DisclosureViolation",
    "PromiseRecord",
    "Turn",
    "VoiceCall",
    "VoiceConfig",
    "VoiceState",
    "assert_no_configuration_can_disable",
]
