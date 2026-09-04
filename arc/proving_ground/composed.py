"""The composed allocator-and-Gate policy, and why vetoes collapse.

THE PROBLEM THIS SOLVES. The Allocator samples an action and logs the exact
probability it sampled with. Then the Gate certifies, and it can refuse. What
actually reached the world is therefore not always what the Allocator drew,
and an estimator that conditions on the INTENDED action's propensity is
conditioning on something that did not happen.

So the behaviour policy is the composition of the two:

    pi_exec(a|s) = sum_a'  pi_alloc(a'|s) * 1[ gate.certify(a', s) resolves to a ]

The Gate is deterministic given state and time, so that indicator is a
function rather than an expectation, and the composition is computable in
closed form. No sampling, no estimation - the number is exact.

VETOED BRANCHES COLLAPSE, THEY ARE NEVER DROPPED. A refused branch does not
vanish; the subject receives nothing that cycle, which is `do_nothing`. So the
probability mass of every refused branch is added to `do_nothing`, and the
distribution still sums to one.

    WHY NOT DROP THEM. Dropping vetoed decisions and renormalising looks
    harmless and is selection bias of the worst kind: the Gate refuses
    precisely the subjects whose state makes them refusable - inside a
    cooldown, mid-freeze, out of hours - and those subjects differ
    systematically in their recovery. Removing them removes a non-random slice
    of the sample and the estimate drifts toward whatever the survivors do.
    There is no symptom. `veto_mass` below exists so that the quantity a
    dropping estimator would have discarded is a number on the record rather
    than an absence.

    WHY NOT RENORMALISE THE SURVIVORS EITHER. The same defect, differently
    spelled: it moves mass onto actions that were never more likely, and every
    importance ratio inherits the error.

THE COLLAPSE TARGET IS A FIXED POINT, BY CONSTRUCTION. `do_nothing` resolves
to itself without a Gate call. It is the absence of an action, there is
nothing to authorise, and GI-1 governs effects rather than their absence. If
the target could itself be refused, refused mass would have nowhere to go and
the distribution would not sum to one - so this is load-bearing rather than a
shortcut, and it is asserted rather than assumed.

MULTIPLE BRANCHES RESOLVING TO ONE OUTCOME MUST ADD. Three refused actions are
three separate probabilities landing on the same outcome. Assigning rather
than accumulating would silently keep only the last, and the loss would read
as a rounding error rather than as a bug.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from arc.core.types import ActionType
from arc.gate.context import GateContext
from arc.gate.evaluator import Certificate
from arc.gate.lattice import Verdict

# A decision is (claim, action), never an action alone. A subject holding
# three claims can be offered the same action three times, and keying by the
# action would merge them and lose most of the mass.
DecisionKey = tuple[UUID | None, ActionType]

# The null decision. `claim_id` is None because declining to act is a property
# of the subject's cycle rather than of any one claim, which is exactly how
# the Allocator emits it.
DO_NOTHING: DecisionKey = (None, ActionType.DO_NOTHING)

# Float error tolerated when asserting the distribution sums to one. Tighter
# than any effect this measures, loose enough that summing a few hundred
# doubles is not itself a failure.
MASS_TOLERANCE = 1e-9


class MassNotConserved(AssertionError):
    """A composition lost or invented probability mass.

    An assertion rather than a returned flag: a behaviour policy that does not
    sum to one makes every importance ratio downstream wrong by an unknown
    factor, and continuing would produce a headline number instead of an error.
    """


class CollapseTargetRefused(AssertionError):
    """The outcome that vetoed mass collapses onto was itself refused."""


class CertifyingGate(Protocol):
    """The Gate, as the composition needs it: binding certification only."""

    def certify(self, ctx: GateContext, action: ActionType, at: datetime) -> Certificate: ...


@dataclass(frozen=True)
class Resolution:
    """What became of one branch of the allocator's distribution."""

    key: DecisionKey
    mass: float
    verdict: Verdict
    resolved_to: DecisionKey
    blocking_rule_ids: tuple[str, ...] = ()
    certificate_id: UUID | None = None

    @property
    def vetoed(self) -> bool:
        return self.resolved_to != self.key


@dataclass(frozen=True)
class ComposedPolicy:
    """pi_exec for one subject, with the full record of how it got there."""

    pi_exec: Mapping[DecisionKey, float]
    resolutions: tuple[Resolution, ...]
    collapse_to: DecisionKey = DO_NOTHING

    @property
    def veto_mass(self) -> float:
        """Probability the Gate moved. Exactly what a dropping estimator loses."""
        return sum(r.mass for r in self.resolutions if r.vetoed)

    @property
    def vetoed_keys(self) -> tuple[DecisionKey, ...]:
        return tuple(r.key for r in self.resolutions if r.vetoed)

    @property
    def veto_rate(self) -> float:
        """Share of BRANCHES refused, the CB-VETO diagnostic's numerator.

        Reported beside `veto_mass` because the two answer different
        questions: how many branches were refused, and how much probability
        that actually was.
        """
        if not self.resolutions:
            return 0.0
        return sum(1 for r in self.resolutions if r.vetoed) / len(self.resolutions)

    def realized(self, intended: DecisionKey) -> DecisionKey:
        """Where a sampled branch actually landed."""
        for resolution in self.resolutions:
            if resolution.key == intended:
                return resolution.resolved_to
        raise KeyError(f"{intended} was not a branch of this distribution")

    def propensity_of(self, key: DecisionKey) -> float:
        """pi_exec(key). Zero is a real answer: unreachable under this policy."""
        return float(self.pi_exec.get(key, 0.0))

    def verdict_for(self, key: DecisionKey) -> Verdict:
        for resolution in self.resolutions:
            if resolution.key == key:
                return resolution.verdict
        raise KeyError(f"{key} was not a branch of this distribution")


# The rule id recorded when the Allocator's own in-cycle admission step, not
# the Gate, is what refused a branch. Named so it is greppable in an audit
# trail and so a reader can tell a compliance refusal from a budget one.
ADMISSION_RULE_ID = "ALLOC-ADMISSION"


def resolve_branch(
    key: DecisionKey,
    gate: CertifyingGate,
    contexts: Mapping[UUID | None, GateContext],
    at: datetime,
    *,
    collapse_to: DecisionKey = DO_NOTHING,
    admissible: Callable[[DecisionKey], bool] | None = None,
) -> tuple[Verdict, DecisionKey, tuple[str, ...], UUID | None]:
    """Ask the filters about one branch and say where it lands.

    TWO FILTERS, IN THE ORDER THE SYSTEM APPLIES THEM. Admission first, then
    the Gate. Both are deterministic given the cycle, so both compose in closed
    form, and both collapse a refused branch onto the target.

    WHY ADMISSION BELONGS HERE AND NOT IN THE ENVIRONMENT. The Allocator's
    in-cycle admission step can replace a sampled action with `do_nothing`
    when the draw overshoots a cap, and it deliberately leaves the logged
    propensity alone - M8 documents that and delegates the accounting to this
    composition. If the accounting is not done, the behaviour policy on record
    says `do_nothing` had probability 0.13 while it actually occurred 0.40 of
    the time, every importance ratio is wrong by that factor, and the
    doubly-robust estimate runs about thirty percent high with no symptom.
    Admission is part of the policy, so it is composed like part of the policy.

    ALLOW resolves to itself. Everything else - BLOCK, BLOCK_PERMANENT and
    DEFER alike - resolves to the collapse target, because a deferred action
    is an action that did not happen this cycle. It may happen in a later one
    under a fresh decision with a fresh propensity, and that cycle is a
    different row in the log.
    """
    claim_id, action = key
    if key == collapse_to:
        return Verdict.ALLOW, key, (), None

    if admissible is not None and not admissible(key):
        return Verdict.DEFER, collapse_to, (ADMISSION_RULE_ID,), None

    ctx = contexts.get(claim_id)
    if ctx is None:
        raise KeyError(
            f"no GateContext for claim {claim_id}; the composition cannot certify "
            "a branch it has no state for, and guessing would fail open"
        )

    certificate = gate.certify(ctx, action, at)
    if certificate.decision is Verdict.ALLOW:
        return Verdict.ALLOW, key, (), certificate.certificate_id
    return (
        certificate.decision,
        collapse_to,
        tuple(certificate.blocking_rule_ids),
        certificate.certificate_id,
    )


def composed_propensity(
    pi_alloc: Mapping[DecisionKey, float],
    gate: CertifyingGate,
    contexts: Mapping[UUID | None, GateContext],
    at: datetime,
    *,
    collapse_to: DecisionKey = DO_NOTHING,
    admissible: Callable[[DecisionKey], bool] | None = None,
) -> ComposedPolicy:
    """pi_exec(.|s) in closed form, with every branch accounted for.

    `pi_alloc` is the Allocator's distribution over one subject's decisions.
    `contexts` maps claim id to the Gate state for that claim; the collapse
    target needs no entry, because it is never certified.

    `admissible` is the Allocator's own in-cycle budget admission, as a
    per-branch predicate. Passing it is not optional in any run where
    admission can fire - see `resolve_branch` for what omitting it costs.

    Every branch appears in `resolutions` whether it survived or not, which is
    what makes "collapsed, not dropped" checkable rather than merely asserted.
    """
    if not pi_alloc:
        raise ValueError("a subject with no allocator distribution has no policy")

    total = float(sum(pi_alloc.values()))
    if abs(total - 1.0) > MASS_TOLERANCE:
        raise MassNotConserved(
            f"pi_alloc sums to {total!r}, not one; the allocator handed over "
            "something that is not a distribution, and composing it would "
            "propagate the error into every importance ratio"
        )

    # The collapse target is always in the support, even at zero mass, so that
    # refused mass has somewhere defined to land.
    pi_exec: dict[DecisionKey, float] = {collapse_to: 0.0}
    resolutions: list[Resolution] = []

    for key, mass in pi_alloc.items():
        verdict, resolved_to, blocking, certificate_id = resolve_branch(
            key, gate, contexts, at, collapse_to=collapse_to, admissible=admissible
        )
        resolutions.append(
            Resolution(
                key=key,
                mass=float(mass),
                verdict=verdict,
                resolved_to=resolved_to,
                blocking_rule_ids=blocking,
                certificate_id=certificate_id,
            )
        )
        # ACCUMULATE. Several refused branches land on one outcome and their
        # masses add; assigning would keep only the last and lose the rest.
        pi_exec[resolved_to] = pi_exec.get(resolved_to, 0.0) + float(mass)

    composed = ComposedPolicy(
        pi_exec=pi_exec, resolutions=tuple(resolutions), collapse_to=collapse_to
    )
    assert_mass_conserved(pi_alloc, composed)
    return composed


def assert_mass_conserved(pi_alloc: Mapping[DecisionKey, float], composed: ComposedPolicy) -> None:
    """Every branch survives into the composition, and the total is still one.

    Three separate claims, because they fail differently. A branch missing
    from `resolutions` is a dropped decision. A total away from one is lost or
    invented mass. A collapse-target share below the mass directed onto it
    means a veto landed somewhere other than the target.
    """
    missing = set(pi_alloc) - {r.key for r in composed.resolutions}
    if missing:
        raise MassNotConserved(
            f"{len(missing)} allocator branch(es) never resolved: "
            f"{sorted(missing, key=str)}. A branch absent from the composition "
            "has been dropped, and dropping vetoed decisions is selection bias"
        )

    total = float(sum(composed.pi_exec.values()))
    if abs(total - 1.0) > MASS_TOLERANCE:
        raise MassNotConserved(
            f"pi_exec sums to {total!r}, not one; {1.0 - total:+.3e} of probability "
            "mass was lost or invented in the composition"
        )

    directed = composed.veto_mass + sum(
        r.mass for r in composed.resolutions if not r.vetoed and r.key == composed.collapse_to
    )
    carried = composed.pi_exec.get(composed.collapse_to, 0.0)
    if carried + MASS_TOLERANCE < directed:
        raise CollapseTargetRefused(
            f"{composed.collapse_to} carries {carried!r} but {directed!r} was directed "
            "onto it; vetoed mass went somewhere other than the collapse target"
        )


def veto_diagnostics(policies: Sequence[ComposedPolicy]) -> dict[str, float]:
    """CB-VETO's inputs, over a cycle.

    Because `project` and `certify` share one registry and one evaluator, only
    RUNTIME-class rules can fire after allocation, so this should be near
    zero. Above two percent the eligibility projection is broken rather than
    the Gate being strict, and the breaker trips on that reading.
    """
    branches = sum(len(policy.resolutions) for policy in policies)
    vetoed = sum(len(policy.vetoed_keys) for policy in policies)
    mass = sum(policy.veto_mass for policy in policies)
    return {
        "branches": float(branches),
        "vetoed_branches": float(vetoed),
        "veto_rate": (vetoed / branches) if branches else 0.0,
        "mean_veto_mass": (mass / len(policies)) if policies else 0.0,
    }
