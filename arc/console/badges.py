"""Badges, and the single chokepoint every claim about regulatory force passes.

GI-9 SAYS NO RULE MAY BE REPRESENTED AS CARRYING MORE FORCE THAN ITS SOURCE
INSTRUMENT. That is easy to honour in one renderer and impossible to honour in
six, because the sixth will read `rule.basis` and print it, and `basis` says
`statutory` for a rule whose instrument is a draft consultation.

    SO THE BADGE TEXT IS NEVER DERIVED FROM `basis`. It is `rule.force_label()`
    - M3's function, the one place the honesty rule lives - and the visual tone
    is keyed off that LABEL rather than off the basis behind it. A draft rule
    therefore cannot be given a statutory tone by a renderer that forgot,
    because no renderer here ever sees the basis in a position to print it.

WHY A TONE TABLE KEYED ON THE LABEL. If tone were keyed on `basis`, a draft
statutory rule would get the statutory colour while carrying the draft words,
and a reader scanning colours - which is what a reader does - would come away
with the wrong impression from a technically accurate screen. Colour is a claim
about force too.

`assert_no_overstated_force` is the audit: it re-reads RENDERED OUTPUT and
fails if the word `statutory` appears in the badge of a rule that is not
binding law. Checking the output rather than the inputs is deliberate, because
the failure mode is a call site that bypassed this module entirely.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from arc.gate.registry import Rule, RuleRegistry, RuleStatus

# The word a screen must never attach to a rule that is not binding law.
OVERSTATED = re.compile(r"\bstatutory\b", re.IGNORECASE)


class ForceOverstated(AssertionError):
    """A rule was rendered as carrying more force than its instrument has.

    An assertion rather than a warning: overstating regulatory force does not
    degrade the compliance story, it discredits the whole of it, and a panel
    that has done it once cannot be trusted on the other thirty-two rules.
    """


class Tone(StrEnum):
    """Visual weight. Keyed on the force LABEL, never on the basis."""

    LAW = "law"
    NETWORK = "network"
    OURS = "ours"
    PROVISIONAL = "provisional"


# Keyed on the exact text `Rule.force_label()` returns. A label this table does
# not know falls to PROVISIONAL, which is the conservative direction: an
# unrecognised force claim is shown as weaker than it might be, never stronger.
_TONE_BY_LABEL: dict[str, Tone] = {
    "statutory, in force": Tone.LAW,
    "network rule, in force": Tone.NETWORK,
    "our policy choice": Tone.OURS,
    "heuristic": Tone.OURS,
    "draft, not in force; we apply it anyway": Tone.PROVISIONAL,
    "advisory, not binding; we apply it as policy": Tone.PROVISIONAL,
    "contested: published guidance disagrees, we take the conservative reading": (Tone.PROVISIONAL),
}

_TONE_COLOURS: dict[Tone, tuple[str, str]] = {
    Tone.LAW: ("#1a3a5c", "#d6e4f0"),
    Tone.NETWORK: ("#4a3520", "#f0e4d0"),
    Tone.OURS: ("#2d3a2d", "#dceadc"),
    Tone.PROVISIONAL: ("#5c2020", "#f5dede"),
}


@dataclass(frozen=True)
class ForceBadge:
    """One rule's force, as it may be shown. Built only by `badge_for`."""

    rule_id: str
    text: str
    tone: Tone
    binding_law: bool
    status: RuleStatus

    @property
    def colours(self) -> tuple[str, str]:
        return _TONE_COLOURS[self.tone]

    def html(self) -> str:
        fg, bg = self.colours
        return (
            f'<span class="badge badge-{self.tone.value}" '
            f'style="color:{fg};background:{bg}" '
            f'data-rule="{self.rule_id}" data-tone="{self.tone.value}">'
            f"{_escape(self.text)}</span>"
        )

    def text_line(self) -> str:
        marker = "law " if self.binding_law else "ours"
        return f"{self.rule_id:<20} [{marker}] {self.text}"


def badge_for(rule: Rule) -> ForceBadge:
    """The ONLY constructor. Text comes from M3; tone is keyed off that text."""
    text = rule.force_label()
    return ForceBadge(
        rule_id=rule.id,
        text=text,
        tone=_TONE_BY_LABEL.get(text, Tone.PROVISIONAL),
        binding_law=rule.is_binding_law(),
        status=rule.status,
    )


def badges_for(registry: RuleRegistry) -> list[ForceBadge]:
    return [badge_for(rule) for rule in registry]


def not_in_force(registry: RuleRegistry) -> list[ForceBadge]:
    """The rules that are NOT in force, by name.

    Shown on the panel rather than filtered off it. A compliance summary that
    lists only the rules it is proud of is the summary an experienced reviewer
    stops believing.
    """
    return [badge_for(rule) for rule in registry if rule.status is not RuleStatus.IN_FORCE]


def honest_mix(registry: RuleRegistry) -> dict[str, int]:
    """The counts, with the stricter-than-minimum figure spelled out."""
    summary = registry.summary()
    return {
        "total": summary["total"],
        "statutory": summary.get("basis:statutory", 0),
        "network_rule": summary.get("basis:network_rule", 0),
        "policy_choice": summary.get("basis:policy_choice", 0),
        "heuristic": summary.get("basis:heuristic", 0),
        "in_force": summary.get("status:in_force", 0),
        "draft": summary.get("status:draft", 0),
        "advisory": summary.get("status:advisory", 0),
        "contested": summary.get("status:contested", 0),
        "stricter_than_binding_minimum": summary["stricter_than_binding_minimum"],
    }


def assert_no_overstated_force(rendered: str, registry: RuleRegistry) -> None:
    """Re-read rendered output and refuse anything that overstates force.

    Reads the OUTPUT, not the model. The failure this exists to catch is a call
    site that never went through `badge_for` at all, and inspecting the inputs
    would miss exactly that.
    """
    offenders: list[str] = []
    for rule in registry:
        if rule.is_binding_law():
            continue
        for fragment in _fragments_for(rendered, rule.id):
            if OVERSTATED.search(fragment):
                offenders.append(
                    f"{rule.id} (basis={rule.basis.value}, status={rule.status.value}) "
                    f"rendered as: {fragment.strip()[:120]!r}"
                )
    if offenders:
        raise ForceOverstated(
            "rules rendered as carrying more force than their instrument has (GI-9):\n  "
            + "\n  ".join(offenders)
        )


def _fragments_for(rendered: str, rule_id: str) -> Iterable[str]:
    """Every rendered span that names this rule.

    Both the HTML badge, which tags itself with `data-rule`, and the plain-text
    line, which starts with the rule id. A renderer that emitted neither would
    not be showing the rule at all.
    """
    for match in re.finditer(
        rf'<span[^>]*data-rule="{re.escape(rule_id)}"[^>]*>(.*?)</span>', rendered, re.S
    ):
        yield match.group(0)
    for line in rendered.splitlines():
        if line.strip().startswith(rule_id):
            yield line


def assert_every_rule_shown(rendered: str, registry: RuleRegistry) -> None:
    """A panel that silently omits a rule is not a compliance panel."""
    missing = [rule.id for rule in registry if rule.id not in rendered]
    if missing:
        raise AssertionError(f"{len(missing)} rule(s) missing from the compliance panel: {missing}")


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def escape(text: object) -> str:
    """Shared HTML escaping. Every screen renders through it."""
    return _escape(str(text))


def badge_legend() -> Sequence[tuple[Tone, str]]:
    """What the colours mean, shown beside them.

    A colour key is not decoration here: the panel's whole claim is that it
    distinguishes law from our own judgement, and a reader who cannot tell
    which blue means which has not been told anything.
    """
    return (
        (Tone.LAW, "binding law, in force"),
        (Tone.NETWORK, "card or rail network rule, in force"),
        (Tone.OURS, "our own policy choice"),
        (Tone.PROVISIONAL, "draft, advisory or contested - we apply it anyway"),
    )
