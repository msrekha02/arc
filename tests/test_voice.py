"""M16 acceptance gate - the voice channel.

    test_ai_disclosure_cannot_be_disabled_by_config
    test_no_account_detail_before_verification
    test_hardship_phrase_terminates_automation
    test_wrong_party_discloses_nothing
    test_ptp_extracted_as_structured_record
    test_ptp_below_confidence_threshold_no_state_change

"NON-REMOVABLE" IS TESTED AS AN ABSENCE, NOT A DEFAULT. A test asserting that
`config.disclose_ai is True` passes on exactly the system this milestone exists
to prevent: one where the flag exists, defaults to on, and gets turned off on a
quarter that is going badly. So the assertions below read the field list and
fail if a flag ever appears.

`test_hardship_phrase_terminates_automation` is the demo beat. It is the moment
the system chooses to stop rather than to collect, and it lands because the
automation does not finish its sentence first.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime

import pytest
from arc.llm_service.contracts import (
    EXTRACTION_THRESHOLD,
    STOP_INTENTS,
    Intent,
    IntentAnswer,
)
from arc.voice.conversation import (
    ALLOWED,
    DISCLOSURE,
    FORBIDDEN_CONFIG_FIELDS,
    MAY_DISCLOSE,
    NON_REMOVABLE,
    VOX_DISTRESS,
    VOX_WRONG_PARTY,
    CallOutcome,
    ConfigurationCanDisableRule,
    DisclosureViolation,
    VoiceCall,
    VoiceConfig,
    VoiceState,
    assert_no_configuration_can_disable,
)

AT = datetime(2026, 6, 2, 11, 0, tzinfo=UTC)


def config(**overrides: object) -> VoiceConfig:
    fields_: dict[str, object] = {
        "merchant_name": "Acme",
        "caller_line": "+911402000000",
    }
    fields_.update(overrides)
    return VoiceConfig(**fields_)  # type: ignore[arg-type]


def call() -> VoiceCall:
    return VoiceCall(config(), at=AT)


def heard(intent: Intent, confidence: float = 0.9) -> IntentAnswer:
    return IntentAnswer(intent=intent, confidence=confidence)


# ---------------------------------------------------------------------------
# 1. The six are non-removable
# ---------------------------------------------------------------------------
def test_ai_disclosure_cannot_be_disabled_by_config() -> None:
    """No flag exists. That is what non-removable means.

    Three checks, because each catches a different way of getting it wrong: no
    field name that could disable a rule; the disclosure happens in the
    constructor so a call that exists has already made it; and the wording is a
    constant rather than a template slot somebody can empty.
    """
    assert_no_configuration_can_disable()

    names = {f.name for f in fields(VoiceConfig)}
    offending = names & FORBIDDEN_CONFIG_FIELDS
    assert not offending, (
        f"VoiceConfig exposes {sorted(offending)}. A flag that defaults to on is a "
        f"flag somebody turns off; non-removable means the flag does not exist"
    )

    # The disclosure is the first thing said, before anything can be heard.
    outcome = call().outcome
    assert outcome.turns, "the call said nothing at all"
    first = outcome.turns[0]
    assert first.speaker == "agent"
    assert "automated assistant" in first.text, "the first utterance does not disclose AI"
    assert "recorded" in first.text, "the first utterance does not disclose recording"

    # And the wording is a constant, not a slot that can be emptied.
    assert "{merchant}" in DISCLOSURE
    assert "automated assistant" in DISCLOSURE and "recorded" in DISCLOSURE


def test_a_config_flag_that_could_disable_a_rule_is_caught() -> None:
    """THE SUITE IS NOT VACUOUSLY GREEN.

    The check above passes on a `VoiceConfig` that simply has few fields. This
    plants the flag and confirms the guard fires, and that it fires with the
    reason rather than on some incidental difference.
    """
    from dataclasses import dataclass

    import arc.voice.conversation as voice

    @dataclass(frozen=True)
    class Permissive:
        merchant_name: str = "Acme"
        caller_line: str = "+911402000000"
        disclose_ai: bool = True  # the flag that must not exist

    original = voice.VoiceConfig
    voice.VoiceConfig = Permissive  # type: ignore[misc]
    try:
        with pytest.raises(ConfigurationCanDisableRule) as caught:
            voice.assert_no_configuration_can_disable()
    finally:
        voice.VoiceConfig = original  # type: ignore[misc]

    message = str(caught.value)
    assert "disclose_ai" in message, f"the guard fired without naming the flag: {message!r}"
    assert "defaults to on" in message, (
        "the guard fired but does not explain why a default is not enough"
    )


def test_the_six_rules_are_named_and_a_promotional_cli_is_refused() -> None:
    """VOX-CLI is config-checked, because the CLI genuinely is configurable.

    The number a call is placed from has to be settable - it differs per
    merchant. What is not settable is placing a recovery call from a
    promotional series, so the constructor refuses it.
    """
    assert len(NON_REMOVABLE) == 6
    assert set(NON_REMOVABLE) == {
        "VOX-DISCLOSE",
        "VOX-VERIFY",
        "VOX-RECORD",
        "VOX-CLI",
        "VOX-DISTRESS",
        "VOX-WRONG-PARTY",
    }

    with pytest.raises(ValueError, match="transactional CLI"):
        config(caller_line="+919000000000")


# ---------------------------------------------------------------------------
# 2. Verification before disclosure
# ---------------------------------------------------------------------------
def test_no_account_detail_before_verification() -> None:
    """VOX-VERIFY. Nothing about the account to an unverified party.

    Including during verification itself: that is the question being asked, so
    anything said while it is open is said to somebody unknown.
    """
    live = call()
    assert live.state is VoiceState.OPENING
    assert not live.verified

    with pytest.raises(DisclosureViolation) as caught:
        live.speak_account_detail("Your balance is Rs 1,299.00.")
    assert "VOX-VERIFY" in str(caught.value)

    # VERIFYING is deliberately not a state in which detail may be spoken.
    assert VoiceState.VERIFYING not in MAY_DISCLOSE
    assert {VoiceState.DISCUSSING, VoiceState.CAPTURING_PROMISE} == MAY_DISCLOSE

    # After a verifying turn, it is allowed.
    live.hear("Yes, speaking.", heard(Intent.UNCLEAR))
    assert live.verified and live.state is VoiceState.DISCUSSING
    live.speak_account_detail("Your payment of Rs 1,299.00 did not go through.")
    assert any("1,299.00" in turn.text for turn in live.outcome.turns)


def test_the_state_machine_is_bounded_and_ended_is_absorbing() -> None:
    """The LLM speaks inside this machine; it does not get to leave it."""
    assert ALLOWED[VoiceState.ENDED] == frozenset(), "a call can resume after ending"
    for state, onward in ALLOWED.items():
        assert VoiceState.ENDED in onward or state is VoiceState.ENDED, (
            f"{state} cannot reach ENDED, so a call in it never terminates"
        )

    live = call()
    live.hear("Yes.", heard(Intent.UNCLEAR))
    live.hear("I have lost my job.", heard(Intent.HARDSHIP))
    with pytest.raises(AssertionError, match="not an allowed voice transition"):
        live._move(VoiceState.DISCUSSING)


# ---------------------------------------------------------------------------
# 3. The demo beat
# ---------------------------------------------------------------------------
def test_hardship_phrase_terminates_automation() -> None:
    """The moment the system stops rather than collects.

    It does not finish its sentence, offer an instalment, or ask a qualifying
    question. It stops, says so, and hands to a person - and it does that on
    RECOGNITION rather than on a confidence threshold, because the cost of
    stopping a call that was fine is a call.
    """
    live = call()
    live.hear("Yes, speaking.", heard(Intent.UNCLEAR))

    outcome = live.hear("I lost my job last month and I have been unwell.", heard(Intent.HARDSHIP))

    assert outcome.state is VoiceState.ENDED, "the automation continued after distress"
    assert outcome.stopped_by == VOX_DISTRESS
    assert outcome.handed_off, "distress did not reach a human"

    last = outcome.turns[-1].text.lower()
    assert "stopping this automated call" in last
    assert "colleague" in last

    # Nothing was collected on the way out.
    assert outcome.promise is None
    for turn in outcome.turns:
        if turn.speaker == "agent":
            assert "instalment" not in turn.text.lower()
            assert "pay" not in turn.text.lower(), (
                f"the agent asked for money after distress: {turn.text!r}"
            )


def test_hardship_stops_even_on_a_low_confidence_reading() -> None:
    """Confidence gates state changes, not stops.

    A promise recorded on a shaky extraction is a bad record. A call continued
    through a possible distress signal is a person being argued with by
    software, and those two costs are not comparable.
    """
    assert Intent.HARDSHIP in STOP_INTENTS

    live = call()
    live.hear("Yes.", heard(Intent.UNCLEAR))
    outcome = live.hear("things are very hard", heard(Intent.HARDSHIP, confidence=0.31))

    assert outcome.stopped_by == VOX_DISTRESS
    assert outcome.handed_off
    assert EXTRACTION_THRESHOLD > 0.31, "the test no longer exercises a below-threshold read"


# ---------------------------------------------------------------------------
# 4. Wrong party
# ---------------------------------------------------------------------------
def test_wrong_party_discloses_nothing() -> None:
    """VOX-WRONG-PARTY. Nothing said, number suppressed, no redial.

    There is no redial flag to check, which is the point: the absence is the
    control.
    """
    live = call()
    outcome = live.hear("You have the wrong number.", heard(Intent.WRONG_PARTY))

    assert outcome.state is VoiceState.ENDED
    assert outcome.stopped_by == VOX_WRONG_PARTY
    assert outcome.number_suppressed, "the number was not suppressed"
    assert not outcome.disclosed_account_detail

    spoken = " ".join(t.text for t in outcome.turns if t.speaker == "agent").lower()
    for leak in ("balance", "overdue", "payment of", "account number", "rs "):
        assert leak not in spoken, f"a wrong party was told {leak!r}"
    assert "will not call this number again" in spoken

    assert not any(f.name in FORBIDDEN_CONFIG_FIELDS for f in fields(VoiceConfig)), (
        "a redial flag exists"
    )


# ---------------------------------------------------------------------------
# 5 and 6. Promise extraction
# ---------------------------------------------------------------------------
def test_ptp_extracted_as_structured_record() -> None:
    """A structured record with a confidence, not free text."""
    live = call()
    live.hear("Yes, speaking.", heard(Intent.UNCLEAR))
    outcome = live.hear("I can pay on the 20th.", heard(Intent.PROMISE_TO_PAY, 0.91))

    promise = outcome.promise
    assert promise is not None, "no promise record was produced"
    assert promise.intent is Intent.PROMISE_TO_PAY
    assert promise.confidence == 0.91
    assert promise.actionable
    assert not outcome.review_queued
    assert outcome.state is VoiceState.ENDED

    # THE MODEL NEVER SUPPLIES A FIGURE. `IntentAnswer` has nowhere to put an
    # amount or a date, so a hallucinated one has nowhere to travel.
    answer_fields = set(IntentAnswer.__dataclass_fields__)
    assert not answer_fields & {"amount", "amount_paise", "date", "promise_date"}, (
        f"IntentAnswer can carry a figure the model made up: {sorted(answer_fields)}"
    )


def test_ptp_below_confidence_threshold_no_state_change() -> None:
    """Below threshold: nothing changes, and a human is asked.

    A promise recorded on a shaky extraction freezes the claim, teaches the
    promise-to-pay model something that did not happen, and stops the allocator
    treating a subject who never agreed to anything. Queuing it costs a minute.
    """
    live = call()
    live.hear("Yes.", heard(Intent.UNCLEAR))
    outcome = live.hear("maybe soon, I think", heard(Intent.PROMISE_TO_PAY, 0.42))

    promise = outcome.promise
    assert promise is not None, "the turn was discarded rather than queued"
    assert not promise.actionable, "a 0.42-confidence extraction changed state"
    assert outcome.review_queued, "it was neither actioned nor queued"
    assert promise.amount_paise is None and promise.promise_date is None

    last = outcome.turns[-1].text.lower()
    assert "colleague will confirm" in last
    assert "we will not contact you" not in last, (
        "the customer was told a freeze was applied when no state changed"
    )


@pytest.mark.parametrize(
    ("confidence", "actionable"),
    [(0.0, False), (0.74, False), (EXTRACTION_THRESHOLD, True), (0.99, True)],
)
def test_the_threshold_is_the_boundary_it_claims_to_be(confidence: float, actionable: bool) -> None:
    """Inclusive at the threshold, and nowhere else."""
    live = call()
    live.hear("Yes.", heard(Intent.UNCLEAR))
    outcome = live.hear("the 20th", heard(Intent.PROMISE_TO_PAY, confidence))
    assert outcome.promise is not None
    assert outcome.promise.actionable is actionable
    assert outcome.review_queued is (not actionable)


def test_every_stop_intent_ends_the_call() -> None:
    """All four, not just the two the gate names by title."""
    for intent in sorted(STOP_INTENTS):
        live = call()
        live.hear("Yes.", heard(Intent.UNCLEAR))
        outcome = live.hear("...", heard(intent))
        assert outcome.state is VoiceState.ENDED, f"{intent} did not end the call"
        assert outcome.stopped_by, f"{intent} ended the call without saying why"
        assert isinstance(outcome, CallOutcome)
