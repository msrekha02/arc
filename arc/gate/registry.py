"""The one source of law: rules as data, loaded once, pinned into certificates.

A rule declares four things that are easy to conflate and must not be:

    class        when it can be decided, which is what lets `project` and
                 `certify` share one evaluator instead of two rule sets
    basis        whether it is law, a network rule, or our own judgement
    status       the force of the instrument behind it
    on_violation the remedy, and for DEFER, how to compute when

`basis` and `status` are separate on purpose. A rule can be our policy choice
informed by an advisory report, or a network rule whose published number is
contested. Collapsing the two into "compliance" is how a system ends up
claiming regulatory force it does not have (GI-9).

The registry version is derived from the content of the rules, so pinning a
version into a certificate pins the exact text that was evaluated. A replay
cannot silently re-decide under today's rules.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from arc.core.types import ActionType
from arc.gate.context import Channel
from arc.gate.lattice import Verdict

RULES_DIR = Path(__file__).parent / "rules"


class RuleClass(StrEnum):
    """When a rule can be decided, which determines who may consult it."""

    # Fully decidable from subject state at plan time. `project` treats these
    # as authoritative.
    INVARIANT = "INVARIANT"
    # Decidable as a prediction at the planned execution time.
    TEMPORAL = "TEMPORAL"
    # Decidable because the resource is reserved at plan time.
    RESERVED = "RESERVED"
    # Not decidable until execution: certificate expiry, kill-switch state.
    # `certify` only. `project` must never consult these.
    RUNTIME = "RUNTIME"


PROJECT_CLASSES: frozenset[RuleClass] = frozenset(
    {RuleClass.INVARIANT, RuleClass.TEMPORAL, RuleClass.RESERVED}
)
ALL_CLASSES: frozenset[RuleClass] = frozenset(RuleClass)


class RuleBasis(StrEnum):
    STATUTORY = "statutory"
    NETWORK_RULE = "network_rule"
    POLICY_CHOICE = "policy_choice"
    HEURISTIC = "heuristic"


class RuleStatus(StrEnum):
    IN_FORCE = "in_force"
    DRAFT = "draft"
    ADVISORY = "advisory"
    CONTESTED = "contested"


# A rule resting on a draft or advisory instrument is never described by its
# basis, because that is precisely how "informed by an advisory report" becomes
# "required by law" in a slide deck.
NON_BINDING_STATUSES: frozenset[RuleStatus] = frozenset({RuleStatus.DRAFT, RuleStatus.ADVISORY})


@dataclass(frozen=True)
class Citation:
    instrument: str
    force: str
    date: date | None = None
    note: str | None = None


@dataclass(frozen=True)
class Rule:
    id: str
    rule_class: RuleClass
    basis: RuleBasis
    status: RuleStatus
    check: str
    on_violation: Verdict
    scope: frozenset[Channel]
    actions: frozenset[ActionType] | None
    params: Mapping[str, Any]
    rationale: str
    informed_by: tuple[Citation, ...] = ()
    in_force_from: date | None = None

    def applies_to(self, action: ActionType, channel: Channel) -> bool:
        if self.actions is not None:
            return action in self.actions
        return channel in self.scope

    def is_in_force_at(self, at: datetime) -> bool:
        """A future-dated obligation is not a current one."""
        return self.in_force_from is None or at.date() >= self.in_force_from

    def force_label(self) -> str:
        """How this rule may be described, anywhere, to anyone.

        The basis word is withheld whenever the instrument behind the rule is
        draft or advisory. This is the single function every renderer goes
        through, so the honesty rule cannot be forgotten at one call site.
        """
        if self.status is RuleStatus.DRAFT:
            return "draft, not in force; we apply it anyway"
        if self.status is RuleStatus.ADVISORY:
            return "advisory, not binding; we apply it as policy"
        if self.status is RuleStatus.CONTESTED:
            return "contested: published guidance disagrees, we take the conservative reading"
        return {
            RuleBasis.STATUTORY: "statutory, in force",
            RuleBasis.NETWORK_RULE: "network rule, in force",
            RuleBasis.POLICY_CHOICE: "our policy choice",
            RuleBasis.HEURISTIC: "heuristic",
        }[self.basis]

    def is_binding_law(self) -> bool:
        return self.basis is RuleBasis.STATUTORY and self.status is RuleStatus.IN_FORCE

    def canonical(self) -> dict[str, Any]:
        """The bytes the registry version is computed over."""
        return {
            "id": self.id,
            "class": self.rule_class.value,
            "basis": self.basis.value,
            "status": self.status.value,
            "check": self.check,
            "on_violation": self.on_violation.value,
            "scope": sorted(c.value for c in self.scope),
            "actions": sorted(a.value for a in self.actions) if self.actions else None,
            "params": dict(self.params),
            "in_force_from": self.in_force_from.isoformat() if self.in_force_from else None,
        }


class RegistryError(Exception):
    """The registry could not be loaded or validated. Nothing runs ungated."""


class RuleRegistry:
    """An immutable, content-versioned set of rules."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        if not rules:
            raise RegistryError("an empty registry would allow everything")

        ordered = sorted(rules, key=lambda rule: rule.id)
        seen: set[str] = set()
        for rule in ordered:
            if rule.id in seen:
                raise RegistryError(f"duplicate rule id {rule.id}")
            seen.add(rule.id)

        self._rules: tuple[Rule, ...] = tuple(ordered)
        self._by_id: Mapping[str, Rule] = MappingProxyType({r.id: r for r in ordered})
        payload = json.dumps(
            [rule.canonical() for rule in ordered], sort_keys=True, separators=(",", ":")
        )
        self._version = "rr-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    @property
    def version(self) -> str:
        return self._version

    @property
    def rules(self) -> tuple[Rule, ...]:
        return self._rules

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self._rules)

    def __getitem__(self, rule_id: str) -> Rule:
        try:
            return self._by_id[rule_id]
        except KeyError as exc:
            raise RegistryError(f"unknown rule {rule_id}") from exc

    def of_class(self, classes: frozenset[RuleClass]) -> tuple[Rule, ...]:
        return tuple(rule for rule in self._rules if rule.rule_class in classes)

    def summary(self) -> dict[str, int]:
        """The honest mix, for the compliance panel."""
        counts: dict[str, int] = {"total": len(self._rules)}
        for rule in self._rules:
            counts[f"basis:{rule.basis.value}"] = counts.get(f"basis:{rule.basis.value}", 0) + 1
            counts[f"status:{rule.status.value}"] = counts.get(f"status:{rule.status.value}", 0) + 1
        counts["stricter_than_binding_minimum"] = sum(
            1 for rule in self._rules if not rule.is_binding_law()
        )
        return counts


def _parse_citation(raw: Mapping[str, Any]) -> Citation:
    return Citation(
        instrument=str(raw["instrument"]),
        force=str(raw["force"]),
        date=raw.get("date"),
        note=raw.get("note"),
    )


def _require(raw: Mapping[str, Any], key: str, rule_id: str) -> Any:
    if key not in raw or raw[key] is None:
        raise RegistryError(f"rule {rule_id} is missing required field {key!r}")
    return raw[key]


def parse_rule(raw: Mapping[str, Any]) -> Rule:
    rule_id = str(raw.get("id", "<unnamed>"))
    try:
        scope = frozenset(Channel(value) for value in raw.get("scope", []))
        actions = (
            frozenset(ActionType(value) for value in raw["actions"])
            if raw.get("actions") is not None
            else None
        )
        if not scope and actions is None:
            raise RegistryError(f"rule {rule_id} applies to nothing")

        return Rule(
            id=rule_id,
            rule_class=RuleClass(_require(raw, "class", rule_id)),
            basis=RuleBasis(_require(raw, "basis", rule_id)),
            status=RuleStatus(_require(raw, "status", rule_id)),
            check=str(_require(raw, "check", rule_id)),
            on_violation=Verdict(str(_require(raw, "on_violation", rule_id)).lower()),
            scope=scope,
            actions=actions,
            params=MappingProxyType(dict(raw.get("params") or {})),
            rationale=str(_require(raw, "rationale", rule_id)),
            informed_by=tuple(_parse_citation(c) for c in raw.get("informed_by") or []),
            in_force_from=raw.get("in_force_from"),
        )
    except RegistryError:
        raise
    except (KeyError, ValueError, TypeError) as exc:
        raise RegistryError(f"rule {rule_id} is malformed: {exc}") from exc


def load_registry(directory: Path | None = None) -> RuleRegistry:
    """Read every YAML file in the rules directory, in filename order.

    This is the only place in `arc/gate/` that touches a filesystem. The Gate
    itself receives the loaded registry and performs no I/O at all.
    """
    directory = directory or RULES_DIR
    if not directory.is_dir():
        raise RegistryError(f"rule registry directory {directory} does not exist")

    raw_rules: list[Mapping[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise RegistryError(f"{path.name} must contain a list of rules")
        raw_rules.extend(loaded)

    if not raw_rules:
        raise RegistryError(f"no rules found in {directory}; refusing to run ungated")

    return RuleRegistry([parse_rule(raw) for raw in raw_rules])
