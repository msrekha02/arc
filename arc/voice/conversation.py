"""The voice channel: a bounded state machine with six rules no flag can reach.

"NON-REMOVABLE" MEANS NO FLAG EXISTS. Not that a flag defaults to on. A
configuration option to disable AI disclosure is a configuration option a
campaign manager will eventually find, under pressure, on a quarter that is
going badly - and the fact that it defaulted to on will not be in the
transcript. So the six below are not parameters of anything: there is nowhere
in `VoiceConfig` to put them, and `assert_no_configuration_can_disable` reads
the class to prove it.

    THE SIX
    VOX-DISCLOSE     AI identity disclosed in the FIRST utterance
    VOX-VERIFY       identity verified before any account detail is spoken
    VOX-RECORD       recording disclosed with prior intimation
    VOX-CLI          transactional number series, never promotional
    VOX-DISTRESS     distress -> immediate human handoff, automation ends
    VOX-WRONG-PARTY  wrong party -> disclose nothing, suppress, no redial

THE LLM IS CONFINED TO TWO JOBS INSIDE THIS MACHINE: choosing which allowed
utterance to speak, and classifying what it heard into a closed intent set. It
does not decide when to escalate, what to offer, what an amount is, or when the
call ends. The state machine owns all of that, so a model that returns nonsense
produces a call that says something bland and hangs up, rather than a call that
promises a discount nobody authorised.

WHY THE STOPS DO NOT WAIT FOR CONFIDENCE. `STOP_INTENTS` fire on recognition,
not on a threshold. Confidence gates whether a state CHANGES - whether a
promise-to-pay is recorded - because a wrongly recorded promise freezes a claim
and misleads a model. It does not gate whether the machine stops, because the
cost of stopping a call that was fine is a call, and the cost of continuing one
that was not is a person in distress being argued with by software.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from arc.core.money import Paise
from arc.core.time_authority import ensure_utc
from arc.llm_service.contracts import (
    EXTRACTION_THRESHOLD,
    STOP_INTENTS,
    Intent,
    IntentAnswer,
)

# The rule ids, so a transcript can attribute a stop the same way the Gate does.
VOX_DISCLOSE = "VOX-DISCLOSE"
VOX_VERIFY = "VOX-VERIFY"
VOX_RECORD = "VOX-RECORD"
VOX_CLI = "VOX-CLI"
VOX_DISTRESS = "VOX-DISTRESS"
VOX_WRONG_PARTY = "VOX-WRONG-PARTY"

NON_REMOVABLE: tuple[str, ...] = (
    VOX_DISCLOSE,
    VOX_VERIFY,
    VOX_RECORD,
    VOX_CLI,
    VOX_DISTRESS,
    VOX_WRONG_PARTY,
)

# VOX-CLI. The 140 series is the designated transactional and service range;
# a recovery call placed from an ordinary mobile or a promotional series is
# misdeclared, and gets silently filtered on top of being a network violation.
# source: the national numbering plan's allocation of 140 to transactional and
# service calls, and the phased migration of collections traffic onto it.
TRANSACTIONAL_SERIES = "+91140"

# The opening line. A constant, not a template slot: a template slot is a place
# where the disclosure can be edited down to nothing.
DISCLOSURE = (
    "Hello, this is an automated assistant calling on behalf of {merchant}. "
    "This call is recorded. Am I speaking with the account holder?"
)


class VoiceState(StrEnum):
    """Where the call is. Every transition is in `ALLOWED` and nowhere else."""

    OPENING = "opening"
    VERIFYING = "verifying"
    DISCUSSING = "discussing"
    CAPTURING_PROMISE = "capturing_promise"
    HANDING_OFF = "handing_off"
    ENDED = "ended"


ALLOWED: dict[VoiceState, frozenset[VoiceState]] = {
    VoiceState.OPENING: frozenset({VoiceState.VERIFYING, VoiceState.HANDING_OFF, VoiceState.ENDED}),
    VoiceState.VERIFYING: frozenset(
        {VoiceState.DISCUSSING, VoiceState.HANDING_OFF, VoiceState.ENDED}
    ),
    VoiceState.DISCUSSING: frozenset(
        {VoiceState.CAPTURING_PROMISE, VoiceState.HANDING_OFF, VoiceState.ENDED}
    ),
    VoiceState.CAPTURING_PROMISE: frozenset({VoiceState.HANDING_OFF, VoiceState.ENDED}),
    VoiceState.HANDING_OFF: frozenset({VoiceState.ENDED}),
    VoiceState.ENDED: frozenset(),
}

# States in which an account detail may be spoken. VERIFYING is NOT one of them:
# verification is the question, so anything said during it is said to somebody
# whose identity is still unknown.
MAY_DISCLOSE: frozenset[VoiceState] = frozenset(
    {VoiceState.DISCUSSING, VoiceState.CAPTURING_PROMISE}
)


class DisclosureViolation(AssertionError):
    """An account detail was about to be spoken to an unverified party."""


class ConfigurationCanDisableRule(AssertionError):
    """A non-removable rule turned out to be removable."""


@dataclass(frozen=True)
class VoiceConfig:
    """Everything a campaign MAY configure.

    Read the field list. There is no `disclose_ai`, no `verify_identity`, no
    `announce_recording`, no `allow_redial_wrong_party`. That absence is the
    control - `assert_no_configuration_can_disable` reads this class and fails
    if one ever appears.
    """

    merchant_name: str
    caller_line: str
    max_turns: int = 12
    language: str = "en-IN"

    def __post_init__(self) -> None:
        if not self.merchant_name.strip():
            raise ValueError("a call must say who it is calling on behalf of")
        # VOX-CLI. A promotional series on a transactional call is both a
        # network violation and the reason the call gets silently filtered.
        if not self.caller_line.startswith(TRANSACTIONAL_SERIES):
            raise ValueError(
                f"{self.caller_line!r} is not a transactional CLI series; a recovery "
                "call placed from a promotional number is misdeclared"
            )


# Field names that would make a non-removable rule configurable. Checked
# against `VoiceConfig` rather than trusted, because the failure mode is
# somebody adding one in good faith to unblock a demo.
FORBIDDEN_CONFIG_FIELDS: frozenset[str] = frozenset(
    {
        "disclose_ai",
        "ai_disclosure",
        "skip_disclosure",
        "verify_identity",
        "skip_verification",
        "require_verification",
        "announce_recording",
        "recording_notice",
        "skip_recording_notice",
        "allow_redial",
        "redial_wrong_party",
        "continue_on_distress",
        "distress_handoff",
    }
)


def assert_no_configuration_can_disable() -> None:
    """No field exists through which a non-removable rule could be turned off.

    THE POINT IS THE ABSENCE. A test asserting `config.disclose_ai is True`
    would pass on a system where the flag exists and defaults to on, which is
    exactly the system this is meant to prevent.
    """
    fields = set(VoiceConfig.__dataclass_fields__)
    offending = sorted(fields & FORBIDDEN_CONFIG_FIELDS)
    if offending:
        raise ConfigurationCanDisableRule(
            f"VoiceConfig exposes {offending}. These rules are non-removable, which "
            "means no flag exists - not that a flag defaults to on. A flag that "
            "defaults to on is a flag somebody turns off on a bad quarter"
        )


@dataclass(frozen=True)
class Turn:
    """One thing said, by whom, and what it was understood to mean."""

    speaker: str
    text: str
    intent: Intent = Intent.UNCLEAR
    confidence: float = 0.0


@dataclass(frozen=True)
class PromiseRecord:
    """A promise, only if it was understood well enough to act on.

    `actionable` false means no state change and a review queue. It does not
    mean the promise was broken, and it does not mean nothing was said.
    """

    intent: Intent
    confidence: float
    actionable: bool
    amount_paise: Paise | None = None
    promise_date: datetime | None = None


@dataclass
class CallOutcome:
    state: VoiceState
    turns: list[Turn] = field(default_factory=list)
    stopped_by: str | None = None
    handed_off: bool = False
    disclosed_account_detail: bool = False
    promise: PromiseRecord | None = None
    review_queued: bool = False
    number_suppressed: bool = False

    @property
    def transcript(self) -> str:
        return "\n".join(f"{turn.speaker}: {turn.text}" for turn in self.turns)


class VoiceCall:
    """One call. The LLM speaks and listens; this decides everything else."""

    def __init__(self, config: VoiceConfig, *, at: datetime) -> None:
        ensure_utc(at)
        assert_no_configuration_can_disable()
        self.config = config
        self.at = at
        self.state = VoiceState.OPENING
        self.verified = False
        self.outcome = CallOutcome(state=VoiceState.OPENING)

        # VOX-DISCLOSE and VOX-RECORD, in the FIRST utterance, before anything
        # can be heard back. Not a step the machine could skip: it happens in
        # the constructor, so a call that exists has already disclosed.
        self._say(DISCLOSURE.format(merchant=config.merchant_name))

    def _say(self, text: str) -> None:
        self.outcome.turns.append(Turn(speaker="agent", text=text))

    def _move(self, to: VoiceState, *, stopped_by: str | None = None) -> None:
        if to not in ALLOWED[self.state]:
            raise AssertionError(f"{self.state} -> {to} is not an allowed voice transition")
        self.state = to
        self.outcome.state = to
        if stopped_by:
            self.outcome.stopped_by = stopped_by

    # -- the six, as behaviour -------------------------------------------
    def speak_account_detail(self, detail: str) -> None:
        """VOX-VERIFY. Nothing about the account before the party is verified."""
        if not self.verified or self.state not in MAY_DISCLOSE:
            raise DisclosureViolation(
                f"{VOX_VERIFY}: an account detail was about to be spoken in state "
                f"{self.state} with verified={self.verified}. Verification is the "
                f"question being asked, so anything said during it is said to "
                f"somebody whose identity is still unknown"
            )
        self._say(detail)

    def hear(self, text: str, answer: IntentAnswer) -> CallOutcome:
        """One customer turn. Stops fire on recognition, not on confidence."""
        self.outcome.turns.append(
            Turn(
                speaker="customer",
                text=text,
                intent=answer.intent,
                confidence=answer.confidence,
            )
        )

        # VOX-DISTRESS and VOX-WRONG-PARTY, and the other two stops. Checked
        # BEFORE anything else this turn could do, and without consulting the
        # confidence: the cost of stopping a call that was fine is a call.
        if answer.intent in STOP_INTENTS:
            return self._stop_for(answer.intent)

        if answer.intent is Intent.ALREADY_PAID:
            self._say("Thank you - I will check our records and we will not call again.")
            self._move(VoiceState.ENDED, stopped_by="ALREADY_PAID")
            return self.outcome

        if answer.intent is Intent.PROMISE_TO_PAY:
            return self._capture_promise(answer)

        if self.state is VoiceState.OPENING:
            self.verified = True
            self._move(VoiceState.VERIFYING)
            self._move(VoiceState.DISCUSSING)
        return self.outcome

    def _stop_for(self, intent: Intent) -> CallOutcome:
        if intent is Intent.HARDSHIP:
            # VOX-DISTRESS. The automation ends here. It does not finish its
            # sentence, offer an instalment, or ask a qualifying question.
            self._say(
                "I understand, and I am sorry. I am stopping this automated call "
                "and passing you to a colleague who can help."
            )
            self._move(VoiceState.HANDING_OFF, stopped_by=VOX_DISTRESS)
            self.outcome.handed_off = True
            self._move(VoiceState.ENDED)
            return self.outcome

        if intent is Intent.WRONG_PARTY:
            # VOX-WRONG-PARTY. Nothing is disclosed, the number is suppressed,
            # and there is no redial - which is why there is no redial flag.
            self._say("I am sorry to have troubled you. We will not call this number again.")
            self._move(VoiceState.ENDED, stopped_by=VOX_WRONG_PARTY)
            self.outcome.number_suppressed = True
            return self.outcome

        if intent is Intent.REQUEST_HUMAN:
            self._say("Of course. I am passing you to a colleague now.")
            self._move(VoiceState.HANDING_OFF, stopped_by="VOX-HUMAN-REQUEST")
            self.outcome.handed_off = True
            self._move(VoiceState.ENDED)
            return self.outcome

        # DISPUTE.
        self._say("Understood. I am recording that this is disputed and stopping here.")
        self._move(VoiceState.HANDING_OFF, stopped_by="STOP-DISPUTE")
        self.outcome.handed_off = True
        self._move(VoiceState.ENDED)
        return self.outcome

    def _capture_promise(self, answer: IntentAnswer) -> CallOutcome:
        """Below threshold: no state change, and a review queue.

        A promise recorded on a shaky extraction freezes the claim, tells the
        promise-to-pay model something that did not happen, and stops the
        allocator treating a subject who never agreed to anything. Queuing it
        costs a human a minute.
        """
        if self.state is VoiceState.OPENING:
            self.verified = True
            self._move(VoiceState.VERIFYING)
            self._move(VoiceState.DISCUSSING)
        self._move(VoiceState.CAPTURING_PROMISE)

        actionable = answer.confidence >= EXTRACTION_THRESHOLD
        self.outcome.promise = PromiseRecord(
            intent=answer.intent,
            confidence=answer.confidence,
            actionable=actionable,
        )
        if actionable:
            self._say("Thank you - I have noted that, and we will not contact you before then.")
        else:
            self.outcome.review_queued = True
            self._say("Thank you. A colleague will confirm the details with you.")
        self._move(VoiceState.ENDED)
        return self.outcome
