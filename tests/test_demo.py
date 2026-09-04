"""M17 acceptance gate - the demo harness.

    three consecutive `make demo SEED=3` runs produce byte-identical numbers
    the nine beats of spec section 15, in order
    the outage-suppression beat lands
    the FORBORNE hardship beat lands
    the adversarial suite is readable, and every attack is refused

DETERMINISM IS THE GATE THAT MATTERS. A judge will ask to see it again. The
replay path reads no clock and prints no wall time, so the check here is not
"the numbers are close" but "the bytes are equal", run three times through the
real entry point rather than through an internal function that might be
deterministic while the command is not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest
from arc.console.build import build
from arc.demo.attacks import ATTACKS, run_attack
from arc.demo.harness import adversarial_lines, beats, digest, headline_numbers, script
from arc.proving_ground.arms import Arm
from arc.simulator.seeds import JUDGED_SEED

REPO_ROOT = Path(__file__).resolve().parents[1]

# The demo the gate drives. Smaller than the judged run so the suite stays
# inside a sensible time; determinism is a property of the machinery, not of
# the population size.
POPULATION = 400
CYCLES = 2


@pytest.fixture(scope="module")
def data():
    return build(seed=JUDGED_SEED, size=POPULATION, cycles=CYCLES)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "arc.demo.run", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )


# ---------------------------------------------------------------------------
# Gate 1 - three identical runs
# ---------------------------------------------------------------------------
def test_three_consecutive_runs_are_byte_identical() -> None:
    """`make demo SEED=3`, three times, same bytes.

    Through the real command, not through `script()`. A harness can be
    deterministic while the entry point that wraps it is not - an argument
    default read from the environment, a timestamp in a header - and the thing
    a judge runs is the command.
    """
    runs = [
        _run_cli(
            "--seed",
            str(JUDGED_SEED),
            "--size",
            str(POPULATION),
            "--cycles",
            str(CYCLES),
        )
        for _ in range(3)
    ]
    for index, run in enumerate(runs):
        assert run.returncode == 0, f"run {index} failed:\n{run.stderr[-2500:]}"

    first, second, third = (r.stdout for r in runs)
    assert first == second == third, (
        "three consecutive runs of the judged seed differed. The demo must "
        "reproduce exactly, because a judge will ask to see it again"
    )

    # And the output actually contains a headline, so identical-but-empty
    # cannot pass.
    assert "THE SCOREBOARD" in first
    assert re.search(r"digest [0-9a-f]{64}", first), "no digest was printed"


def test_the_digest_covers_the_numbers_the_demo_shows(data) -> None:
    """Byte-identical is a claim about the output, not about the internals."""
    numbers = headline_numbers(data)
    assert len(numbers) > 40, "the digest covers too little to mean anything"

    joined = "\n".join(numbers)
    for arm in Arm:
        assert f"{arm.value}.recovered_paise=" in joined
        assert f"{arm.value}.prevented_paise=" in joined
        # Guardrails are inside the digest: a run that recovered the same
        # rupees by different means is not the same run.
        assert f"{arm.value}.guardrail.complaint_rate_per_1000=" in joined
        assert f"{arm.value}.guardrail.opt_out_rate_per_1000=" in joined
    assert "batch.suppressed=" in joined

    assert digest(data) == digest(data)


def test_a_changed_number_changes_the_digest(data) -> None:
    """A digest that ignored a figure would call two different runs identical."""
    import hashlib

    baseline = digest(data)
    tampered = list(headline_numbers(data))
    tampered[2] = tampered[2] + "1"
    changed = hashlib.sha256("\n".join(tampered).encode("utf-8")).hexdigest()
    assert changed != baseline


# ---------------------------------------------------------------------------
# Gate 2 - the nine beats, in order
# ---------------------------------------------------------------------------
def test_nine_beats_in_the_order_of_the_script(data) -> None:
    """Spec section 15, sequenced."""
    sequence = beats(data)
    assert [b.number for b in sequence] == list(range(1, 10))

    titles = [b.title for b in sequence]
    assert titles[0] == "the batch lands"
    assert titles[1] == "diagnosis splits it"
    assert titles[2] == "the allocator runs"
    assert titles[3] == "compliance firewall, live"
    assert titles[5] == "the hardship stop"
    assert titles[6] == "the scoreboard"
    assert titles[8] == "replay one claim"

    for beat in sequence:
        assert beat.lines, f"beat {beat.number} prints nothing"


def test_the_two_beats_that_must_land_have_pauses(data) -> None:
    """The outage suppression and the hardship stop both need narration room.

    A pause is not a courtesy here. Both beats are contrasts, and a contrast
    delivered at the same pace as a table is a table.
    """
    paused = {b.title for b in beats(data) if b.pause_after}
    assert paused == {"diagnosis splits it", "the hardship stop"}, (
        f"the wrong beats pause: {paused}"
    )


def test_the_outage_beat_lands(data) -> None:
    """Zero contact, and what the naive arm sent to those same claims."""
    beat = next(b for b in beats(data) if b.title == "diagnosis splits it")
    # Whitespace-normalised: the beat is wrapped for a terminal, so a phrase
    # the reader sees as one sentence spans two lines in the source.
    text = re.sub(r"\s+", " ", " ".join(beat.lines))

    assert data.batch.suppressed_by_outage > 0, (
        "no claim was suppressed, so the beat has nothing to show"
    )
    assert "SUPPRESSED by a detected issuer outage" in text
    assert "zero contact of any kind" in text
    assert "not deferred, not throttled" in text, "the beat does not rule out the weaker reading"

    naive = data.result.runs[Arm.NAIVE_DUNNING]
    assert naive.contacts > 0, "the naive arm sent nothing, so the contrast is empty"
    assert f"{naive.contacts:,}" in text, (
        "the contrast with the naive arm is the point and its number is missing"
    )
    assert "a calendar does not" in text


def test_the_forborne_beat_lands(data) -> None:
    """The system gives up money on purpose, and says so.

    Checked against the domain rather than only against the words: FORBORNE
    really is absorbing, and the beat claims exactly that.
    """
    from arc.core.types import LEGAL_TRANSITIONS, ClaimState

    assert LEGAL_TRANSITIONS[ClaimState.FORBORNE] == frozenset(), (
        "FORBORNE has an outgoing edge, so the beat's central claim is false"
    )

    beat = next(b for b in beats(data) if b.title == "the hardship stop")
    text = re.sub(r"\s+", " ", " ".join(beat.lines))
    assert "FORBORNE" in text
    assert "absorbing" in text
    assert "TERMINATES" in text
    assert "no expected-value argument reopens it" in text.lower()
    assert "give up money, on purpose" in text
    # And the cancellation story, which is what makes it work mid-sleep.
    assert "Nothing polls" in text


def test_the_script_renders_end_to_end(data) -> None:
    """Every beat, plus the header and the digest, as one stream."""
    lines = list(script(data))
    body = "\n".join(lines)

    for number in range(1, 10):
        assert re.search(rf"^\s+{number}\. ", body, re.M), f"beat {number} is missing"
    assert f"seed {data.result.seed}" in body
    assert re.search(r"digest [0-9a-f]{64}", body)
    # The replay trace is carried into the demo as prose.
    assert "The Sentinel attributed the failure to" in body


def test_narrated_pauses_are_driven_not_slept(data) -> None:
    """The pauses are injected, so the gate does not wait on them."""
    waited: list[float] = []
    lines = list(script(data, pause=2.0, sleep=waited.append))
    assert waited == [2.0, 2.0], f"expected two narration pauses, got {waited}"
    assert lines


# ---------------------------------------------------------------------------
# Gate 3 - the adversarial suite
# ---------------------------------------------------------------------------
def test_every_attack_is_refused_with_an_attributable_rule() -> None:
    """A refusal nobody can attribute is indistinguishable from a lucky bug."""
    assert len(ATTACKS) >= 12, "the suite is too small to be convincing"

    for attack in ATTACKS:
        outcome = run_attack(attack)
        assert outcome.refused, f"{attack.description!r} was ALLOWED. {outcome.refused_by}"
        assert outcome.refused_by and "NOTHING" not in outcome.refused_by, (
            f"{attack.description!r} was refused but nothing was named as refusing it"
        )


def test_the_named_attacks_from_the_build_document_are_present() -> None:
    """The specific attacks the build document lists, by description."""
    described = " | ".join(a.description for a in ATTACKS).lower()
    for expected in (
        "19:01",
        "16th retry",
        "forborne",
        "cooldown",
        "name into the ledger",
        "expired certificate",
        "no certificate",
        "draft rule as statutory",
    ):
        assert expected in described, f"the suite does not attempt: {expected}"


def test_adversarial_output_is_readable() -> None:
    """Each line says what was attempted, that it was refused, and by which rule."""
    lines = adversarial_lines()
    body = "\n".join(lines)

    assert "attempted" in body and "refused by" in body
    assert f"{len(ATTACKS)} of {len(ATTACKS)} attacks refused." in body
    assert "AN ATTACK SUCCEEDED" not in body

    # Named rules, not a row of ticks.
    for rule_id in ("TIME-WINDOW", "ABS-FORBORNE", "ABS-CONSENT", "NET-CAT1"):
        assert rule_id in body, f"{rule_id} does not appear as a refusing rule"


def test_adversarial_command_runs_and_shows_the_self_monitoring_breakers() -> None:
    """`make demo-adversarial`, end to end."""
    run = _run_cli("--adversarial")
    assert run.returncode == 0, run.stderr[-2500:]
    out = run.stdout

    assert "ADVERSARIAL SUITE" in out
    assert f"{len(ATTACKS)} of {len(ATTACKS)} attacks refused." in out
    assert "AN ATTACK SUCCEEDED" not in out

    # The three that measure the machinery rather than customer harm.
    assert "[self] CB-VETO" in out
    assert "[self] CB-DEGRADED" in out
    assert "[self] CB-COHORT-BLIND" in out
    assert "[harm] CB-COMPLAINT" in out


def test_make_targets_exist() -> None:
    """The three commands the build document names."""
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ("demo:", "demo-live:", "demo-adversarial:", "console:"):
        assert f"\n{target}" in makefile, f"the Makefile has no {target} target"
    assert "SEED ?= 3" in makefile, "the demo does not default to the judged seed"
