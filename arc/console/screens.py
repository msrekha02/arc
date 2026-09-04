"""The four screens, as view models first and HTML second.

WHY VIEW MODELS AND NOT TEMPLATES. Every claim these screens make is a claim
about money, compliance or causation, and each one has an invariant behind it:
the headline cannot appear without its guardrails, prevention cannot be added
into recovery, a draft rule cannot be shown as law. Those invariants are
assertions over a STRUCTURE. Put them in a template and they become conventions
that hold until somebody adds a column.

    SO EACH SCREEN IS A DATACLASS THAT REFUSES TO BE BUILT WRONG, and rendering
    is a total function from that dataclass. The tests assert on the model
    where the property is structural and on the rendered output where the
    property is about what a reader actually sees.

WHY SERVER-RENDERED HTML WITH NO BUILD STEP. The build document says React and
Tailwind. This repo has no JavaScript toolchain, and a console that cannot be
exercised by the test suite would be the one screen in the system whose claims
are unverified - which is precisely backwards for the screen whose job is to
show that the other claims are true. These are self-contained HTML documents
with inline styles: they open from disk, they render from real data, and every
invariant on them is asserted in `tests/test_console.py`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from arc.console.badges import (
    ForceBadge,
    assert_every_rule_shown,
    assert_no_overstated_force,
    badge_legend,
    badges_for,
    escape,
    honest_mix,
    not_in_force,
)
from arc.core.money import Paise, format_inr
from arc.core.types import CauseLayer
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry
from arc.proving_ground.arms import Arm
from arc.proving_ground.metrics import Scoreboard

# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------
_STYLE = """
:root { color-scheme: light dark; }
body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif;
       background:#fbfbfa; color:#1b1b19; }
main { max-width:1080px; margin:0 auto; padding:28px 20px 64px; }
h1 { font-size:19px; margin:0 0 4px; letter-spacing:-.01em; }
h2 { font-size:14px; margin:28px 0 10px; text-transform:uppercase;
     letter-spacing:.07em; color:#6b6b66; font-weight:600; }
.sub { color:#6b6b66; margin:0 0 20px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid #e8e6e1; }
th { font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:#6b6b66; }
td.n, th.n { text-align:right; }
.badge { display:inline-block; padding:1px 7px; border-radius:9px;
         font-size:11px; font-weight:600; white-space:nowrap; }
.tiles { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 8px; }
.tile { flex:1 1 150px; border:1px solid #e8e6e1; border-radius:8px;
        padding:11px 13px; background:#fff; }
.tile .k { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
           color:#6b6b66; }
.tile .v { font-size:22px; font-weight:650; letter-spacing:-.02em; }
.tile.point { border-color:#8a2f2f; background:#fdf4f4; }
.tile.point .v { color:#8a2f2f; }
.bar { height:9px; border-radius:5px; background:#e8e6e1; overflow:hidden;
       display:flex; }
.bar span { display:block; height:100%; }
.note { color:#6b6b66; font-size:12px; margin:8px 0 0; }
.prose p { margin:0 0 11px; max-width:70ch; }
.prose .step { border-left:3px solid #d8d5cd; padding:1px 0 1px 13px; margin:0 0 13px; }
.spark { display:flex; align-items:flex-end; gap:3px; height:52px; }
.spark i { display:block; width:22px; background:#8a2f2f; border-radius:2px 2px 0 0; }
.spark.arc i { background:#2d5a3d; }
@media (prefers-color-scheme: dark) {
  body { background:#16161a; color:#e8e6e1; }
  .tile,table { background:transparent; }
  .tile { border-color:#31313a; } th,td { border-color:#31313a; }
  .tile.point { border-color:#c46a6a; background:#241a1a; }
  .tile.point .v { color:#e08c8c; }
}
"""


def document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )


def _tile(key: str, value: str, *, point: bool = False) -> str:
    cls = "tile point" if point else "tile"
    return (
        f'<div class="{cls}"><div class="k">{escape(key)}</div>'
        f'<div class="v">{escape(value)}</div></div>'
    )


# ---------------------------------------------------------------------------
# 1. Batch view
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BatchView:
    """Live counters and the diagnosis split.

    `suppressed_by_outage` is the number this screen exists for: claims a
    detected issuer outage took off the contact path entirely. The naive arm
    messaged every one of them.
    """

    seed: int
    claims: int
    subjects: int
    at_risk_paise: Paise

    issuer: int
    merchant: int
    customer: int
    unknown: int

    suppressed_by_outage: int
    self_healing: int
    naive_contacted_same_claims: int
    cohort_blind: int = 0

    def __post_init__(self) -> None:
        split = self.issuer + self.merchant + self.customer + self.unknown
        if split != self.claims:
            raise ValueError(
                f"the diagnosis split covers {split} claims but the batch holds "
                f"{self.claims}; a claim with no layer has not been diagnosed and "
                "must not be quietly dropped off the count"
            )
        if self.suppressed_by_outage > self.issuer:
            raise ValueError(
                f"{self.suppressed_by_outage} claims suppressed by an outage but only "
                f"{self.issuer} were diagnosed issuer-layer; suppression follows the "
                "diagnosis and cannot exceed it"
            )

    @property
    def contact_avoided(self) -> int:
        """Claims ARC did not message that a fixed-schedule dunner would have."""
        return self.suppressed_by_outage + self.self_healing

    def render(self) -> str:
        pct = lambda n: (100.0 * n / self.claims) if self.claims else 0.0  # noqa: E731
        bar = (
            '<div class="bar">'
            f'<span style="width:{pct(self.issuer):.2f}%;background:#8a2f2f"></span>'
            f'<span style="width:{pct(self.merchant):.2f}%;background:#b8862b"></span>'
            f'<span style="width:{pct(self.customer):.2f}%;background:#2d5a3d"></span>'
            f'<span style="width:{pct(self.unknown):.2f}%;background:#9a9a94"></span>'
            "</div>"
        )
        body = (
            f"<h1>Batch &mdash; seed {self.seed}</h1>"
            f'<p class="sub">{self.claims:,} claims across {self.subjects:,} subjects, '
            f"{format_inr(self.at_risk_paise)} at risk</p>"
            "<h2>Counters</h2>"
            '<div class="tiles">'
            + _tile("claims", f"{self.claims:,}")
            + _tile("subjects", f"{self.subjects:,}")
            + _tile("at risk", format_inr(self.at_risk_paise))
            + _tile("suppressed by outage", f"{self.suppressed_by_outage:,}", point=True)
            + "</div>"
            "<h2>Diagnosis split</h2>" + bar + "<table>"
            "<tr><th>layer</th><th class='n'>claims</th><th class='n'>share</th>"
            "<th>what happens</th></tr>"
            f"<tr><td>issuer</td><td class='n'>{self.issuer:,}</td>"
            f"<td class='n'>{pct(self.issuer):.1f}%</td>"
            "<td>suppressed. zero customer contact until the outage clears</td></tr>"
            f"<tr><td>merchant</td><td class='n'>{self.merchant:,}</td>"
            f"<td class='n'>{pct(self.merchant):.1f}%</td>"
            "<td>repaired at the rail. zero customer contact</td></tr>"
            f"<tr><td>customer</td><td class='n'>{self.customer:,}</td>"
            f"<td class='n'>{pct(self.customer):.1f}%</td>"
            "<td>eligible for outreach, subject to the Gate</td></tr>"
            f"<tr><td>unknown</td><td class='n'>{self.unknown:,}</td>"
            f"<td class='n'>{pct(self.unknown):.1f}%</td>"
            "<td>conservative path and review queue. never guessed</td></tr>"
            "</table>"
            f'<p class="note">{self.suppressed_by_outage:,} claims were suppressed by a '
            f"detected issuer outage and received no contact of any kind. The naive "
            f"fixed-schedule arm messaged {self.naive_contacted_same_claims:,} of those "
            f"same claims, because a calendar does not know the issuer is down. "
            f"{self.cohort_blind:,} claims were diagnosed without cohort power and are "
            f"counted as a known blind spot rather than as a clean NORMAL.</p>"
        )
        return document(f"ARC batch - seed {self.seed}", body)


# ---------------------------------------------------------------------------
# 2. Compliance firewall
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuleCounter:
    rule_id: str
    fired: int
    verdict: Verdict


@dataclass(frozen=True)
class FirewallView:
    """Proposed to executed, with per-rule counters and the honest mix."""

    proposed: int
    blocked: int
    deferred: int
    executed: int
    counters: Sequence[RuleCounter]
    registry: RuleRegistry

    def __post_init__(self) -> None:
        accounted = self.blocked + self.deferred + self.executed
        if accounted > self.proposed:
            raise ValueError(
                f"{accounted} outcomes from {self.proposed} proposals; the funnel "
                "invents actions it was never asked about"
            )

    @property
    def badges(self) -> list[ForceBadge]:
        return badges_for(self.registry)

    @property
    def mix(self) -> Mapping[str, int]:
        return honest_mix(self.registry)

    def render(self) -> str:
        mix = self.mix
        fired = {c.rule_id: c for c in self.counters}
        rows = ""
        for badge in self.badges:
            hit = fired.get(badge.rule_id)
            rows += (
                "<tr>"
                f"<td>{escape(badge.rule_id)}</td>"
                f"<td>{badge.html()}</td>"
                f"<td class='n'>{hit.fired if hit else 0}</td>"
                f"<td>{escape(hit.verdict.value) if hit else '-'}</td>"
                "</tr>"
            )
        legend = " ".join(
            f'<span class="badge" style="color:{c[0]};background:{c[1]}">{escape(what)}</span>'
            for tone, what in badge_legend()
            for c in [_legend_colour(tone)]
        )
        pending = "".join(
            f"<li><strong>{escape(b.rule_id)}</strong> &mdash; {b.html()}</li>"
            for b in not_in_force(self.registry)
        )
        body = (
            "<h1>Compliance firewall</h1>"
            '<p class="sub">Every rule evaluates on every call. The full verdict list '
            "is what the audit trail needs, not just the blocker.</p>"
            "<h2>Funnel</h2>"
            '<div class="tiles">'
            + _tile("proposed", f"{self.proposed:,}")
            + _tile("blocked", f"{self.blocked:,}")
            + _tile("deferred", f"{self.deferred:,}")
            + _tile("executed", f"{self.executed:,}")
            + "</div>"
            "<h2>The honest mix</h2>"
            f'<p class="sub">{mix["total"]} rules: {mix["statutory"]} statutory, '
            f"{mix['network_rule']} network, {mix['policy_choice']} our own policy "
            f"choice. {mix['in_force']} in force, {mix['draft']} draft, "
            f"{mix['advisory']} advisory, {mix['contested']} contested. We are "
            f"deliberately stricter than the binding minimum in "
            f"{mix['stricter_than_binding_minimum']} places.</p>"
            f'<p class="note">{legend}</p>'
            "<h2>Not in force, and applied anyway</h2>"
            f"<ul>{pending}</ul>"
            "<h2>Every rule, with its force</h2>"
            "<table><tr><th>rule</th><th>force</th><th class='n'>fired</th>"
            "<th>verdict</th></tr>" + rows + "</table>"
        )
        rendered = document("ARC compliance firewall", body)
        # THE HONESTY AUDIT, ON THE WAY OUT. Checked against what was actually
        # rendered rather than against the model that produced it.
        assert_no_overstated_force(rendered, self.registry)
        assert_every_rule_shown(rendered, self.registry)
        return rendered


def _legend_colour(tone: object) -> tuple[str, str]:
    from arc.console.badges import _TONE_COLOURS

    return _TONE_COLOURS[tone]  # type: ignore[index]


# ---------------------------------------------------------------------------
# 3. Scoreboard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreboardView:
    """Five arms. The headline cannot be shown without its guardrails.

    That is not enforced here by discipline: `Scoreboard.to_dict` refuses to
    serialise a recovery figure without the full guardrail block, and this
    screen renders FROM that payload. There is no path to the number that does
    not pass the refusal.
    """

    scoreboard: Scoreboard
    dr_error_develop: float
    dr_error_judged: float
    judged_seed: int
    decay: Mapping[Arm, Sequence[int]] = field(default_factory=dict)

    def payload(self) -> Mapping[str, object]:
        return self.scoreboard.to_dict()

    def render(self) -> str:
        payload = self.payload()
        arms = payload["arms"]
        rows = ""
        for arm in arms:  # type: ignore[union-attr]
            rails = arm["guardrails"]
            rows += (
                "<tr>"
                f"<td>{escape(arm['arm'])}</td>"
                f"<td class='n'>{format_inr(Paise(arm['recovered_paise']))}</td>"
                f"<td class='n'>{format_inr(Paise(arm['incremental_paise']))}</td>"
                f"<td class='n'>{format_inr(Paise(arm['spend_paise']))}</td>"
                f"<td class='n'>{rails['complaint_rate_per_1000']:.2f}</td>"
                f"<td class='n'>{rails['opt_out_rate_per_1000']:.2f}</td>"
                f"<td class='n'>{rails['voluntary_cancel_rate_treated']:.3f}</td>"
                f"<td class='n'>{rails['cost_per_rupee_collected']:.3f}</td>"
                f"<td class='n'>{rails['promise_kept_rate']:.2f}</td>"
                f"<td class='n'>{format_inr(Paise(arm['prevented_paise']))}</td>"
                "</tr>"
            )

        decay = ""
        for arm, series in self.decay.items():
            if not series:
                continue
            top = max(max(s) for s in self.decay.values() if s) or 1
            bars = "".join(
                f'<i style="height:{max(2, round(100 * v / top))}%" '
                f'title="cycle {i}: {format_inr(Paise(v))}"></i>'
                for i, v in enumerate(series)
            )
            css = "spark arc" if arm is Arm.ARC else "spark"
            decay += (
                f"<div><div class='k'>{escape(arm.value)}</div>"
                f"<div class='{css}'>{bars}</div>"
                f"<div class='note'>{' &rarr; '.join(format_inr(Paise(v)) for v in series)}</div>"
                "</div>"
            )

        body = (
            "<h1>Scoreboard</h1>"
            f'<p class="sub">Incremental against {escape(payload["comparator"])}, '
            f"seed {payload['seed']}, {payload['cycles']} cycles. Denominator: "
            f"{escape(arms[0]['denominator'])}</p>"  # type: ignore[index]
            "<h2>Arms &mdash; recovery and guardrails, one table</h2>"
            "<table><tr><th>arm</th><th class='n'>recovered</th>"
            "<th class='n'>incremental</th><th class='n'>spend</th>"
            "<th class='n'>compl/1k</th><th class='n'>optout/1k</th>"
            "<th class='n'>cancel</th><th class='n'>cost/&#8377;</th>"
            "<th class='n'>ptp kept</th><th class='n'>prevented</th></tr>" + rows + "</table>"
            '<p class="note">Prevention is the last column and is NEVER added into '
            "recovery. Money that never failed was never recovered.</p>"
            "<h2>Estimator error against simulator ground truth</h2>"
            '<div class="tiles">'
            + _tile("develop seed", f"{self.dr_error_develop * 100:.2f}%")
            + _tile(
                f"judged seed {self.judged_seed}",
                f"{self.dr_error_judged * 100:.2f}%",
                point=True,
            )
            + "</div>"
            '<p class="note">Both are shown, and the judged seed is the worse one. '
            "Reporting only the develop figure would be selecting the seed after "
            "seeing the result, which is the thing the three-seed discipline exists "
            "to prevent.</p>"
            "<h2>Recovery per cycle &mdash; what the constraints buy</h2>"
            f'<div class="tiles">{decay}</div>'
            '<p class="note">The unconstrained arm contacts everyone every cycle. '
            "Its recovery decays as the response model's annoyance term bites, which "
            "is why beating it on net value is the result and not an accident.</p>"
        )
        return document("ARC scoreboard", body)


# ---------------------------------------------------------------------------
# 4. Replay
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReplayView:
    """One claim's full decision trace, in prose.

    PROSE, NOT JSON. A JSON dump is not a trace, it is the raw material for
    one, and a reviewer asked to audit a decision from a JSON blob is being
    asked to do the explaining themselves. Every number here is placed in a
    sentence that says what it meant.
    """

    paragraphs: Sequence[str]
    claim_id: str
    subject_token: str

    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def render(self) -> str:
        body = (
            f"<h1>Replay &mdash; claim {escape(self.claim_id[:8])}</h1>"
            f'<p class="sub">subject {escape(self.subject_token)}</p>'
            '<div class="prose">'
            + "".join(f'<p class="step">{escape(p)}</p>' for p in self.paragraphs)
            + "</div>"
        )
        return document(f"ARC replay - {self.claim_id[:8]}", body)


LAYER_ORDER: tuple[CauseLayer, ...] = (
    CauseLayer.ISSUER,
    CauseLayer.MERCHANT,
    CauseLayer.CUSTOMER,
    CauseLayer.UNKNOWN,
)
