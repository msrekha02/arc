"""The scoreboard, and the two things it structurally refuses to do.

REFUSAL ONE: A RECOVERY NUMBER WITHOUT ITS GUARDRAILS. Recovery that generates
complaints and opt-outs is not a win, it is a cost deferred to next quarter,
and a headline reported alone invites exactly that trade. So this is not a
convention here and not a review checklist item: `Headline` cannot be
CONSTRUCTED without a complete `Guardrails`, and `to_dict` re-checks the
payload it just built and raises if a recovery figure appears without all of
them beside it. A later edit that adds a recovery key and forgets the
guardrails fails at runtime rather than shipping a prettier number.

REFUSAL TWO: PREVENTION MERGED INTO RECOVERY. Money that never failed was
never recovered. A T-24h nudge that stops a bounce is the most valuable thing
the system does and it belongs on its own line, because adding it to the
recovery total inflates the headline with money that was never at risk in the
sense the denominator claims. `prevented_paise` is a sibling of the headline
rather than a component of it, and `recovered_paise` is summed from the money
ledger's RECOVERED leg alone, so there is no arithmetic path from one to the
other.

DENOMINATOR DISCIPLINE. `denominator` is a required, free-text statement of
what the rate is over. Recovered divided by all failures contaminates the
denominator with failures the system never attempted, and an unstated
denominator is the first thing an experienced reader attacks.

THE HEADLINE IS INCREMENTAL, NOT TOTAL. It is stated against arm B, the naive
fixed-schedule dunning that is the industry default, with a bootstrap interval.
A total is a count. Only the difference against a comparator is a measurement.

REVERSALS MOVE IT DOWN. `recovered_paise` is read from the money ledger, which
derives every balance by summing legs and stores no running total. A
chargeback posts RECOVERY_REVERSED and the headline falls out of the same sum,
with nothing having to remember to decrement anything. A number that cannot
decrease is not a measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from arc.core.money import Paise, format_inr, paise
from arc.proving_ground.arms import Arm

# Every one of these must be present beside a recovery figure. Named as a
# constant so the serialiser can check the payload it built rather than
# trusting the dataclass that built it.
REQUIRED_GUARDRAIL_KEYS: tuple[str, ...] = (
    "complaint_rate_per_1000",
    "opt_out_rate_per_1000",
    "voluntary_cancel_rate_treated",
    "voluntary_cancel_rate_control",
    "cost_per_rupee_collected",
    "promise_made_rate",
    "promise_kept_rate",
    "right_party_contact_rate",
)

# Keys whose presence in a payload demands the guardrails above.
RECOVERY_KEYS: frozenset[str] = frozenset({"recovered_paise", "incremental_paise", "recovered_inr"})


class GuardrailsMissing(ValueError):
    """A recovery figure was assembled without the metrics that qualify it."""


class PreventionMerged(ValueError):
    """Prevented leakage was counted as recovery."""


@dataclass(frozen=True)
class Guardrails:
    """What the recovery cost, in everything that is not money.

    Counts rather than rates, because a rate with a hidden denominator is how
    a guardrail gets quietly satisfied. The rates are derived here so the
    denominator is visible in the code that computes them.
    """

    contacts: int
    complaints: int
    opt_outs: int

    treated_subjects: int
    treated_cancellations: int
    control_subjects: int
    control_cancellations: int

    promises_made: int
    promises_kept: int
    promises_unresolved: int

    right_party_contacts: int

    spend_paise: Paise
    recovered_paise: Paise

    def __post_init__(self) -> None:
        for name in (
            "contacts",
            "complaints",
            "opt_outs",
            "treated_subjects",
            "treated_cancellations",
            "control_subjects",
            "control_cancellations",
            "promises_made",
            "promises_kept",
            "promises_unresolved",
            "right_party_contacts",
        ):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} is {value}; a count cannot be negative")
        if self.promises_kept > self.promises_made:
            raise ValueError(
                f"{self.promises_kept} promises kept out of {self.promises_made} made; "
                "a promise cannot be kept before it is made"
            )

    @staticmethod
    def _rate(numerator: int, denominator: int, *, scale: float = 1.0) -> float:
        """Zero denominator yields zero, never a division error or a None.

        A rate over nothing is reported as zero and the count beside it is
        zero too, so the pair cannot be mistaken for a real measurement.
        """
        if denominator <= 0:
            return 0.0
        return scale * numerator / denominator

    @property
    def complaint_rate_per_1000(self) -> float:
        return self._rate(self.complaints, self.contacts, scale=1000.0)

    @property
    def opt_out_rate_per_1000(self) -> float:
        return self._rate(self.opt_outs, self.contacts, scale=1000.0)

    @property
    def voluntary_cancel_rate_treated(self) -> float:
        return self._rate(self.treated_cancellations, self.treated_subjects)

    @property
    def voluntary_cancel_rate_control(self) -> float:
        """The sleeping-dog check. Treated above control is value destroyed."""
        return self._rate(self.control_cancellations, self.control_subjects)

    @property
    def cost_per_rupee_collected(self) -> float:
        """The economics that decide digital against human treatment."""
        return self._rate(int(self.spend_paise), int(self.recovered_paise))

    @property
    def promise_made_rate(self) -> float:
        return self._rate(self.promises_made, self.contacts)

    @property
    def promise_kept_rate(self) -> float:
        """Reported beside the made rate on purpose.

        A widening gap between the two means promises are being extracted that
        the customer cannot afford, which reads as success in a kept-count and
        as harm in the relationship.
        """
        return self._rate(self.promises_kept, self.promises_made)

    @property
    def right_party_contact_rate(self) -> float:
        return self._rate(self.right_party_contacts, self.contacts)

    def to_dict(self) -> dict[str, float | int]:
        payload: dict[str, float | int] = {
            "contacts": self.contacts,
            "complaints": self.complaints,
            "opt_outs": self.opt_outs,
            "promises_made": self.promises_made,
            "promises_kept": self.promises_kept,
            "promises_unresolved": self.promises_unresolved,
            "treated_subjects": self.treated_subjects,
            "control_subjects": self.control_subjects,
            "spend_paise": int(self.spend_paise),
        }
        for key in REQUIRED_GUARDRAIL_KEYS:
            payload[key] = float(getattr(self, key))
        return payload


@dataclass(frozen=True)
class Headline:
    """Incremental recovery against a comparator, welded to its guardrails.

    `guardrails` has no default. That is the structural refusal: there is no
    way to build this object without them, so there is no code path that can
    emit the number alone.
    """

    arm: Arm
    comparator: Arm
    recovered_paise: Paise
    comparator_recovered_paise: Paise
    spend_paise: Paise
    denominator: str
    guardrails: Guardrails
    ci_low_paise: Paise | None = None
    ci_high_paise: Paise | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.guardrails, Guardrails):
            raise GuardrailsMissing(
                f"{self.arm} reported recovery with guardrails={self.guardrails!r}. "
                "Recovery that generates complaints and opt-outs is not a win, so "
                "the number does not exist without them"
            )
        if not self.denominator.strip():
            raise ValueError(
                "denominator must be stated; a recovery rate over an unstated "
                "denominator is contaminated by failures never attempted"
            )
        if int(self.guardrails.recovered_paise) != int(self.recovered_paise):
            raise GuardrailsMissing(
                f"guardrails were computed against {self.guardrails.recovered_paise} "
                f"but the headline reports {self.recovered_paise}; cost per rupee "
                "would be quoted against the wrong total"
            )

    @property
    def incremental_paise(self) -> Paise:
        """The measurement. Total is a count; only the difference is a result."""
        return paise(int(self.recovered_paise) - int(self.comparator_recovered_paise))

    @property
    def incremental_per_rupee_spent(self) -> float:
        if int(self.spend_paise) <= 0:
            return 0.0
        return int(self.incremental_paise) / int(self.spend_paise)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "arm": self.arm.value,
            "comparator": self.comparator.value,
            "denominator": self.denominator,
            "recovered_paise": int(self.recovered_paise),
            "recovered_inr": format_inr(self.recovered_paise),
            "comparator_recovered_paise": int(self.comparator_recovered_paise),
            "incremental_paise": int(self.incremental_paise),
            "incremental_per_rupee_spent": self.incremental_per_rupee_spent,
            "spend_paise": int(self.spend_paise),
            "ci_95_paise": (
                None
                if self.ci_low_paise is None or self.ci_high_paise is None
                else [int(self.ci_low_paise), int(self.ci_high_paise)]
            ),
            "guardrails": self.guardrails.to_dict(),
        }
        assert_guardrails_present(payload)
        return payload


@dataclass(frozen=True)
class Diagnostics:
    """Whether the system's own machinery is working. Not customer harm.

    Having these at all is what separates an engineered system from a demo:
    they measure the measurement, and each one has a breaker behind it.
    """

    post_allocation_veto_rate: float = 0.0
    degraded_decision_share: float = 0.0
    cohort_blindspot_share: float = 0.0
    explore_mass_share: float = 0.0
    dr_relative_error: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "post_allocation_veto_rate": self.post_allocation_veto_rate,
            "degraded_decision_share": self.degraded_decision_share,
            "cohort_blindspot_share": self.cohort_blindspot_share,
            "explore_mass_share": self.explore_mass_share,
            "dr_relative_error": self.dr_relative_error,
        }


@dataclass(frozen=True)
class ArmReport:
    """One arm's result. Prevention sits beside recovery, never inside it."""

    headline: Headline
    prevented_paise: Paise = field(default_factory=lambda: paise(0))
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    subjects: int = 0

    @property
    def arm(self) -> Arm:
        return self.headline.arm

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            **self.headline.to_dict(),
            "subjects": self.subjects,
            # A SEPARATE LINE. Never added into recovered_paise, and asserted
            # below rather than left to a reader's discipline.
            "prevented_paise": int(self.prevented_paise),
            "prevented_inr": format_inr(self.prevented_paise),
            "diagnostics": self.diagnostics.to_dict(),
        }
        assert_guardrails_present(payload)
        assert_prevention_separate(payload)
        return payload


@dataclass(frozen=True)
class Scoreboard:
    """Every arm, one denominator, one comparator."""

    reports: tuple[ArmReport, ...]
    comparator: Arm = Arm.NAIVE_DUNNING
    seed: int = 0
    cycles: int = 0

    def __post_init__(self) -> None:
        if not self.reports:
            raise ValueError("a scoreboard with no arms compares nothing")
        arms = [report.arm for report in self.reports]
        if len(set(arms)) != len(arms):
            raise ValueError(f"an arm appears twice: {arms}")

    def by_arm(self, arm: Arm) -> ArmReport:
        for report in self.reports:
            if report.arm is arm:
                return report
        raise KeyError(f"{arm} did not run")

    @property
    def arc(self) -> ArmReport:
        return self.by_arm(Arm.ARC)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "seed": self.seed,
            "cycles": self.cycles,
            "comparator": self.comparator.value,
            "arms": [report.to_dict() for report in self.reports],
        }
        for arm_payload in payload["arms"]:
            assert_guardrails_present(arm_payload)  # type: ignore[arg-type]
            assert_prevention_separate(arm_payload)  # type: ignore[arg-type]
        return payload

    def render(self) -> list[str]:
        """The scoreboard as lines. Guardrails on the same rows as the money."""
        lines = [
            f"seed {self.seed} - {self.cycles} cycles - incremental vs {self.comparator.value}",
            "",
            f"{'arm':<22}{'recovered':>14}{'incremental':>14}{'spend':>12}"
            f"{'per Rs':>9}{'compl/1k':>10}{'optout/1k':>11}{'cancel':>9}",
        ]
        for report in self.reports:
            head, rails = report.headline, report.headline.guardrails
            lines.append(
                f"{report.arm.value:<22}"
                f"{format_inr(head.recovered_paise):>14}"
                f"{format_inr(head.incremental_paise):>14}"
                f"{format_inr(head.spend_paise):>12}"
                f"{head.incremental_per_rupee_spent:>9.2f}"
                f"{rails.complaint_rate_per_1000:>10.2f}"
                f"{rails.opt_out_rate_per_1000:>11.2f}"
                f"{rails.voluntary_cancel_rate_treated:>9.3f}"
            )
        lines.append("")
        lines.append(
            "prevention (separate line, never merged into recovery): "
            + ", ".join(f"{r.arm.value} {format_inr(r.prevented_paise)}" for r in self.reports)
        )
        return lines


def assert_guardrails_present(payload: Mapping[str, object]) -> None:
    """A payload carrying recovery must carry every guardrail.

    Checked on the BUILT PAYLOAD rather than on the object that built it, so
    that a future serialiser which adds a recovery key by another route is
    caught too. This is the structural half of the refusal; the required
    `Guardrails` field is the other half.
    """
    if not RECOVERY_KEYS & set(payload):
        return

    rails = payload.get("guardrails")
    if not isinstance(rails, Mapping):
        raise GuardrailsMissing(
            "payload reports "
            + ", ".join(sorted(RECOVERY_KEYS & set(payload)))
            + " with no guardrails block. A recovery number is not reportable alone"
        )

    absent = [key for key in REQUIRED_GUARDRAIL_KEYS if key not in rails]
    if absent:
        raise GuardrailsMissing(
            "payload reports recovery but its guardrails omit: " + ", ".join(absent)
        )


def assert_prevention_separate(payload: Mapping[str, object]) -> None:
    """Prevention is a sibling of recovery, never a component of it."""
    if "prevented_paise" not in payload:
        return
    if "recovered_paise" not in payload:
        raise PreventionMerged("prevention reported without the recovery line it sits beside")

    prevented = int(payload["prevented_paise"])  # type: ignore[arg-type]
    if prevented < 0:
        raise PreventionMerged(f"prevented_paise is {prevented}; prevention cannot be negative")

    incremental = payload.get("incremental_paise")
    recovered = payload.get("recovered_paise")
    comparator = payload.get("comparator_recovered_paise")
    if (
        isinstance(incremental, int)
        and isinstance(recovered, int)
        and isinstance(comparator, int)
        and incremental != recovered - comparator
    ):
        raise PreventionMerged(
            f"incremental {incremental} is not recovered {recovered} minus comparator "
            f"{comparator}; something else has been folded into the headline"
        )


def guardrails_from_counts(
    *,
    contacts: int,
    complaints: int,
    opt_outs: int,
    treated_subjects: int,
    treated_cancellations: int,
    control_subjects: int,
    control_cancellations: int,
    promises_made: int,
    promises_kept: int,
    promises_unresolved: int,
    right_party_contacts: int,
    spend_paise: Paise,
    recovered_paise: Paise,
) -> Guardrails:
    """Keyword-only construction, so no count is passed to the wrong slot."""
    return Guardrails(
        contacts=contacts,
        complaints=complaints,
        opt_outs=opt_outs,
        treated_subjects=treated_subjects,
        treated_cancellations=treated_cancellations,
        control_subjects=control_subjects,
        control_cancellations=control_cancellations,
        promises_made=promises_made,
        promises_kept=promises_kept,
        promises_unresolved=promises_unresolved,
        right_party_contacts=right_party_contacts,
        spend_paise=spend_paise,
        recovered_paise=recovered_paise,
    )


def blows_guardrails(arm: Guardrails, baseline: Guardrails, *, factor: float = 1.5) -> list[str]:
    """Which guardrails this arm breached against a baseline, by name.

    The threshold matches the circuit breakers at spec 7.6: a rate more than
    `factor` times the comparator's is a trip, not a variation.
    """
    breached: list[str] = []
    checks = (
        ("complaint_rate_per_1000", arm.complaint_rate_per_1000, baseline.complaint_rate_per_1000),
        ("opt_out_rate_per_1000", arm.opt_out_rate_per_1000, baseline.opt_out_rate_per_1000),
        (
            "voluntary_cancel_rate_treated",
            arm.voluntary_cancel_rate_treated,
            baseline.voluntary_cancel_rate_treated,
        ),
    )
    for name, value, reference in checks:
        if reference <= 0.0:
            if value > 0.0:
                breached.append(name)
        elif value > factor * reference:
            breached.append(name)
    return breached


def cheapest_arm(reports: Sequence[ArmReport]) -> Arm:
    """The arm that spent least. Reported so cost is never implicit."""
    return min(reports, key=lambda r: int(r.headline.spend_paise)).arm
