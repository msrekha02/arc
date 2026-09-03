"""The Gate. A pure function over a versioned registry, called at four moments.

    project()   advisory eligibility mask for the Allocator, INVARIANT +
                TEMPORAL + RESERVED classes. Prunes; cannot authorise.
    certify()   binding. Every class. Issues or refuses a certificate.

Both go through `evaluate`, differing only in which decidability classes they
filter to. There is deliberately no second rule set and no fast path for
`project`, because two rule sets drift apart silently and the drift is
invisible until it produces an action nobody authorised (GI-6).

No I/O. No clock. No model calls. `at` and the whole `GateContext` are
parameters, which is what makes replay honest and lets the adversarial suite
mean something: a Gate that reads the world cannot be re-run against the past.

Every rule evaluates on every call. The certificate carries the full verdict
list, not just the blocking one, because an audit trail that records only the
refusal cannot show what else was considered.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from arc.core.ids import ARC_NAMESPACE, deterministic_uuid
from arc.core.time_authority import ensure_utc
from arc.core.types import ActionType
from arc.gate.checks import get_check
from arc.gate.context import GateContext
from arc.gate.lattice import DeferWithoutTimestamp, Resolution, Verdict, resolve
from arc.gate.registry import (
    ALL_CLASSES,
    NON_BINDING_STATUSES,
    PROJECT_CLASSES,
    Rule,
    RuleBasis,
    RuleClass,
    RuleRegistry,
    RuleStatus,
    load_registry,
)

CERTIFICATE_NAMESPACE = deterministic_uuid(ARC_NAMESPACE, "compliance-certificate")

# Half-width of the certificate validity window. An ALLOW issued at 18:58 for a
# voice call must not still be executable at 19:02.
DEFAULT_CERT_HALF_WINDOW = timedelta(minutes=15)


@dataclass(frozen=True)
class RuleVerdict:
    """One rule's answer, carried whether or not it was the one that refused."""

    rule_id: str
    verdict: Verdict
    rule_class: RuleClass
    basis: RuleBasis
    status: RuleStatus
    applicable: bool
    defer_until: datetime | None = None
    detail: str = ""

    def force_label(self) -> str:
        return _FORCE_LABELS[self.rule_id]

    def to_audit_dict(self, registry: RuleRegistry) -> dict[str, Any]:
        """Audit rendering. Force always goes through `Rule.force_label`."""
        rule = registry[self.rule_id]
        return {
            "rule_id": self.rule_id,
            "verdict": self.verdict.value,
            "class": self.rule_class.value,
            "force": rule.force_label(),
            "binding_law": rule.is_binding_law(),
            "applicable": self.applicable,
            "defer_until": self.defer_until.isoformat() if self.defer_until else None,
            "detail": self.detail,
        }


# Populated lazily so RuleVerdict.force_label works without a registry handle.
_FORCE_LABELS: dict[str, str] = {}


@dataclass(frozen=True)
class Evaluation:
    """The shared result of one evaluation, before it becomes a certificate."""

    action: ActionType
    at: datetime
    decision: Verdict
    defer_until: datetime | None
    verdicts: tuple[RuleVerdict, ...]
    classes: frozenset[RuleClass]
    rule_registry_version: str

    @property
    def blocking_rule_ids(self) -> tuple[str, ...]:
        """The rules actually responsible for the outcome, in registry order."""
        if self.decision is Verdict.ALLOW:
            return ()
        return tuple(v.rule_id for v in self.verdicts if v.verdict is self.decision)


@dataclass(frozen=True)
class Certificate:
    """Binding authorisation, valid for a window, pinned to a rule version."""

    certificate_id: UUID
    decision: Verdict
    valid_from: datetime
    valid_until: datetime
    evaluated_rules: tuple[RuleVerdict, ...]
    blocking_rule_ids: tuple[str, ...]
    defer_until: datetime | None
    rule_registry_version: str
    action: ActionType
    issued_at: datetime
    claim_id: UUID = field(default=None)  # type: ignore[assignment]

    def is_valid_at(self, moment: datetime) -> bool:
        return self.decision is Verdict.ALLOW and self.valid_from <= moment <= self.valid_until

    def authorises(self, action: ActionType, moment: datetime) -> bool:
        return self.action is action and self.is_valid_at(moment)

    def to_audit_dict(self, registry: RuleRegistry) -> dict[str, Any]:
        return {
            "certificate_id": str(self.certificate_id),
            "claim_id": str(self.claim_id) if self.claim_id else None,
            "action": self.action.value,
            "decision": self.decision.value,
            "issued_at": self.issued_at.isoformat(),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat(),
            "defer_until": self.defer_until.isoformat() if self.defer_until else None,
            "blocking_rule_ids": list(self.blocking_rule_ids),
            "rule_registry_version": self.rule_registry_version,
            "evaluated_rules": [v.to_audit_dict(registry) for v in self.evaluated_rules],
        }


class Gate:
    """PURE. No I/O, no clock, no model calls. All inputs passed in."""

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        self._registry = registry if registry is not None else load_registry()
        # Fail at construction, not at the first veto, if a rule names a check
        # that does not exist. Nothing should discover an unloadable registry
        # while deciding whether to call someone.
        for rule in self._registry:
            get_check(rule.check)
            _FORCE_LABELS[rule.id] = rule.force_label()

    @property
    def registry(self) -> RuleRegistry:
        return self._registry

    # -- the one evaluator -------------------------------------------------
    def evaluate(
        self,
        ctx: GateContext,
        action: ActionType,
        at: datetime,
        *,
        classes: frozenset[RuleClass] = ALL_CLASSES,
    ) -> Evaluation:
        """Evaluate every rule of the given classes. No short-circuit, ever."""
        ensure_utc(at)
        channel = ctx.channel_for(action)
        verdicts: list[RuleVerdict] = []

        for rule in self._registry:
            if rule.rule_class not in classes:
                continue
            verdicts.append(self._evaluate_rule(rule, ctx, action, at, channel))

        try:
            resolution = resolve([(v.verdict, v.defer_until) for v in verdicts])
        except DeferWithoutTimestamp:  # pragma: no cover - defended twice
            resolution = Resolution(decision=Verdict.BLOCK, defer_until=None)

        return Evaluation(
            action=action,
            at=at,
            decision=resolution.decision,
            defer_until=resolution.defer_until,
            verdicts=tuple(verdicts),
            classes=frozenset(classes),
            rule_registry_version=self._registry.version,
        )

    def _evaluate_rule(
        self,
        rule: Rule,
        ctx: GateContext,
        action: ActionType,
        at: datetime,
        channel: Any,
    ) -> RuleVerdict:
        def allow(reason: str) -> RuleVerdict:
            return RuleVerdict(
                rule_id=rule.id,
                verdict=Verdict.ALLOW,
                rule_class=rule.rule_class,
                basis=rule.basis,
                status=rule.status,
                applicable=False,
                detail=reason,
            )

        if not rule.applies_to(action, channel):
            return allow(f"out of scope for {action}")
        if not rule.is_in_force_at(at):
            # A future-dated obligation is not a current one.
            return allow(f"not in force until {rule.in_force_from}")

        try:
            outcome = get_check(rule.check)(ctx, action, at, rule.params)
        except Exception as exc:  # noqa: BLE001 - a throwing rule must not pass
            return RuleVerdict(
                rule_id=rule.id,
                verdict=Verdict.BLOCK,
                rule_class=rule.rule_class,
                basis=rule.basis,
                status=rule.status,
                applicable=True,
                detail=f"rule evaluation failed, failing closed: {type(exc).__name__}",
            )

        if not outcome.violated:
            return RuleVerdict(
                rule_id=rule.id,
                verdict=Verdict.ALLOW,
                rule_class=rule.rule_class,
                basis=rule.basis,
                status=rule.status,
                applicable=True,
                detail=outcome.detail,
            )

        verdict = rule.on_violation
        detail = outcome.detail
        if verdict is Verdict.DEFER and outcome.until is None:
            # A DEFER nobody can sleep on is a BLOCK wearing the wrong label.
            verdict = Verdict.BLOCK
            detail = f"{detail} (no computable next-eligible time, downgraded to BLOCK)"

        return RuleVerdict(
            rule_id=rule.id,
            verdict=verdict,
            rule_class=rule.rule_class,
            basis=rule.basis,
            status=rule.status,
            applicable=True,
            defer_until=outcome.until if verdict is Verdict.DEFER else None,
            detail=detail,
        )

    # -- call site 1: the Allocator prunes ---------------------------------
    def project(
        self, ctx: GateContext, actions: Sequence[ActionType], at: datetime
    ) -> set[ActionType]:
        """Advisory eligibility mask. Cannot authorise anything.

        RUNTIME rules are excluded because they are not decidable at plan time.
        DEFER is not eligible either: the Allocator pins a planned execution
        time, and an action that must wait is not an action for that slot.
        """
        return {
            action
            for action in actions
            if self.evaluate(ctx, action, at, classes=PROJECT_CLASSES).decision is Verdict.ALLOW
        }

    def project_evaluations(
        self, ctx: GateContext, actions: Sequence[ActionType], at: datetime
    ) -> dict[ActionType, Evaluation]:
        """The same pass, with reasons kept, for the drop log and the console."""
        return {
            action: self.evaluate(ctx, action, at, classes=PROJECT_CLASSES) for action in actions
        }

    # -- call site 2: the decision commits ---------------------------------
    def certify(self, ctx: GateContext, action: ActionType, at: datetime) -> Certificate:
        """Binding. Evaluates every class and issues or refuses a certificate."""
        evaluation = self.evaluate(ctx, action, at, classes=ALL_CLASSES)
        half_window = self._cert_half_window()
        valid_from, valid_until = self._validity_window(ctx, action, at, half_window, evaluation)

        return Certificate(
            certificate_id=self._certificate_id(ctx, evaluation),
            decision=evaluation.decision,
            valid_from=valid_from,
            valid_until=valid_until,
            evaluated_rules=evaluation.verdicts,
            blocking_rule_ids=evaluation.blocking_rule_ids,
            defer_until=evaluation.defer_until,
            rule_registry_version=evaluation.rule_registry_version,
            action=action,
            issued_at=at,
            claim_id=ctx.claim_id,
        )

    def _validity_window(
        self,
        ctx: GateContext,
        action: ActionType,
        at: datetime,
        half_window: timedelta,
        evaluation: Evaluation,
    ) -> tuple[datetime, datetime]:
        """The contiguous interval around `at` in which the Gate still allows.

        A flat plus-or-minus window is not enough. An ALLOW issued at 18:58 for
        a voice call would otherwise stay valid until 19:13, and the dispatcher
        would happily place a 19:02 call under an authorisation that was honest
        when it was written. Both edges are therefore walked back to the last
        moment the Gate itself would still say ALLOW.

        Pure and deterministic: a bounded binary search to minute resolution
        over an interval this object already knows how to evaluate.
        """
        if evaluation.decision is not Verdict.ALLOW:
            # A refusal is not an authorisation; the window is informational.
            return at - half_window, at + half_window

        def allows(moment: datetime) -> bool:
            return self.evaluate(ctx, action, moment, classes=ALL_CLASSES).decision is (
                Verdict.ALLOW
            )

        def walk(edge: datetime, *, forward: bool) -> datetime:
            if allows(edge):
                return edge
            near, far = at, edge
            while abs(far - near) > timedelta(minutes=1):
                mid = near + (far - near) / 2
                if allows(mid):
                    near = mid
                else:
                    far = mid
            return near.replace(second=0, microsecond=0) if forward else near

        return walk(at - half_window, forward=False), walk(at + half_window, forward=True)

    def _cert_half_window(self) -> timedelta:
        for rule in self._registry:
            if rule.check == "certificate_window":
                return timedelta(minutes=float(rule.params.get("half_window_minutes", 15)))
        return DEFAULT_CERT_HALF_WINDOW

    def _certificate_id(self, ctx: GateContext, evaluation: Evaluation) -> UUID:
        """Derived, not random, so identical inputs give an identical result.

        This is what makes the Gate observably pure, and it makes the M9
        idempotency key stable across dispatch retries for free.
        """
        parts = [
            evaluation.rule_registry_version,
            str(ctx.claim_id),
            evaluation.action.value,
            evaluation.at.isoformat(),
            evaluation.decision.value,
        ]
        for verdict in evaluation.verdicts:
            parts.append(
                f"{verdict.rule_id}={verdict.verdict.value}"
                f"@{verdict.defer_until.isoformat() if verdict.defer_until else '-'}"
            )
        return deterministic_uuid(CERTIFICATE_NAMESPACE, *parts)


def render_rule_mix(registry: RuleRegistry) -> list[str]:
    """The honest summary line for the compliance panel.

    Goes through `force_label`, so a draft or advisory instrument is never
    described by its basis here either.
    """
    lines = [f"{len(registry)} rules in registry {registry.version}"]
    for rule in registry:
        marker = "law" if rule.is_binding_law() else "ours"
        lines.append(f"  {rule.id:22} {marker:5} {rule.force_label()}")
    return lines


def statutory_rules(registry: RuleRegistry) -> Iterable[Rule]:
    """Only rules that are genuinely binding law may be listed as such."""
    return (
        rule
        for rule in registry
        if rule.basis is RuleBasis.STATUTORY and rule.status not in NON_BINDING_STATUSES
    )
