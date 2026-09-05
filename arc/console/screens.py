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
from arc.core.reproducibility import JUDGED_DIGEST
from arc.core.types import CauseLayer
from arc.gate.lattice import Verdict
from arc.gate.registry import RuleRegistry
from arc.proving_ground.arms import Arm
from arc.proving_ground.metrics import Scoreboard
from arc.simulator.seeds import JUDGED_SEED

# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------
# ARC console - an instrument panel, not a page.
#
# DARK BY DEFAULT AND NOT BY PREFERENCE. These screens are read off a projector
# in a room with the lights on. `prefers-color-scheme` would hand half of those
# rooms a light theme that washes out, so the theme is fixed and the contrast is
# set for the worst case - which is the LIGHTEST surface, the top of a card's
# gradient, not the page background. Measured against `--surface-2` after this
# pass: 13.70:1 primary text, 7.03:1 dim prose, 4.56:1 mute labels, 8.86:1
# accent figures, 9.47:1 good, 5.97:1 bad. Body text clears AAA and nothing
# clears less than AA.
#
#     TWO GREYS WERE LIGHTENED TO GET THERE, and the reason is worth keeping.
#     A mid-grey that passes on the page background can still fail on a card,
#     because a card is lighter than the page by construction. `--text-mute`
#     at #6B7280 measured 3.42:1 on `--surface-2` and carries every section
#     label, card label and table header on these screens; `--text-dim` at
#     #9BA3AF measured 6.49:1 and carries the prose. Both were raised along
#     the same neutral ramp until the worst surface cleared its floor, which
#     is the smallest change that makes the palette true rather than nearly
#     true.
#
#     A THIRD LIFT, ON THE GREY BAR, FOR THE SAME REASON AT THE NON-TEXT FLOOR.
#     A bar carries its meaning in WHERE IT STOPS, so the colour that has to be
#     findable is the right end of the gradient against the track behind it.
#     #4B5563 on `--surface` measured 2.42:1, under the 3:1 that 1.4.11 asks of
#     a graphic; #586374 measures exactly 3.00:1 and is still a clear step below
#     the accent bar. The figure is printed beside every bar either way, so this
#     was legibility on a projector rather than a conformance failure - but a
#     bar nobody can find is not a chart.
#
# ELEVATION BY TONE, NOT BY BORDER. Three surfaces, each a step lighter, and a
# card is a 180deg gradient between the top two rather than a flat fill - so a
# card separates from the page even where its hairline is invisible on a
# projector. Heavy borders read as boxes; tone reads as depth.
#
# ONE ACCENT, SPENT SPARINGLY. Amber marks the numbers that carry a claim - the
# suppressed-by-outage tile, ARC's own series, the graded figure - and nothing
# else. The compliance badges are the one deliberate exception: four statuses
# must be told apart across a room, so they get four hues. `ForceBadge.html`
# writes a light-theme palette into a style attribute, which is why those four
# rules carry `!important`; without it the badges render as pale chips on a dark
# screen. The same applies to the diagnosis bar, where the renderer writes both
# a width and a colour and only the colour is overridden.
#
# NUMBERS ARE MONOSPACED, PROSE IS NOT. Tabular figures in a mono face keep
# digits in columns down a table; prose in a mono face is a chore to read.
#
# NO COMMENTS BELOW THIS LINE. The emitted stylesheet is comment-free by
# request, so every explanation lives here instead.
_STYLE = """
:root{
--sp-1:4px;--sp-2:8px;--sp-3:12px;--sp-4:16px;--sp-5:20px;--sp-6:24px;
--sp-8:32px;--sp-10:40px;--sp-12:48px;--sp-16:64px;
--fs-label:10px;--fs-section:11px;--fs-3:12px;--fs-table:13px;--fs-body:15px;
--fs-card:26px;--fs-headline:40px;--fs-title:38px;
--lh-tight:1.05;--lh-snug:1.4;--lh-base:1.6;
--tr-section:0.14em;--tr-label:0.12em;--tr-3:0.04em;--tr-neg:-0.02em;
--r-1:4px;--r-2:6px;--r-3:10px;--r-pill:999px;
--bg:#0A0C11;
--surface:#12151C;
--surface-2:#1A1F29;
--border:rgba(255,255,255,0.07);
--border-hi:rgba(255,255,255,0.14);
--rule:rgba(255,255,255,0.04);
--row-hover:rgba(255,255,255,0.02);
--text:#E8EAED;
--text-dim:#A1AAB6;
--text-mute:#7E8797;
--accent:#F0B429;
--accent-deep:#D19A1F;
--accent-dim:#8A6516;
--good:#4ADE80;
--bad:#F87171;
--card:linear-gradient(180deg,var(--surface-2),var(--surface));
--shadow:0 1px 2px rgba(0,0,0,0.5),0 8px 24px rgba(0,0,0,0.25);
--shadow-lift:0 2px 4px rgba(0,0,0,0.5),0 14px 34px rgba(0,0,0,0.35);
--glow:0 0 24px rgba(240,180,41,0.15);
--dur:160ms;
--ease:cubic-bezier(0.2,0.7,0.3,1);
--ring:0 0 0 2px var(--bg),0 0 0 4px var(--accent);
--measure:1280px;
--gutter:32px;
--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
--badge-law-fg:#a3ccff;--badge-law-bg:#122539;--badge-law-line:#2f4d75;
--badge-net-fg:#e8c48f;--badge-net-bg:#2b2215;--badge-net-line:#5a4326;
--badge-ours-fg:#95d6b6;--badge-ours-bg:#0f281e;--badge-ours-line:#2d5443;
--badge-prov-fg:#f0a8a8;--badge-prov-bg:#2d1719;--badge-prov-line:#602e32;
--bar-grey-a:#444D5C;--bar-grey-b:#586374;
color-scheme:dark;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--text);
font:var(--fs-body)/var(--lh-base) var(--sans);-webkit-font-smoothing:antialiased;
text-rendering:optimizeLegibility;}
svg.defs{position:absolute;width:0;height:0;overflow:hidden;}
.topbar{position:sticky;top:0;z-index:20;height:56px;background:var(--surface);
border-bottom:1px solid var(--border);}
.topbar-in{max-width:var(--measure);margin:0 auto;height:56px;
padding:0 var(--gutter);display:flex;align-items:center;
justify-content:space-between;gap:var(--sp-4);}
.brand{display:flex;align-items:baseline;gap:var(--sp-3);min-width:0;}
.brand .mark{font:600 var(--fs-body)/1 var(--sans);color:var(--text);
letter-spacing:var(--tr-3);}
.brand .full{font:600 var(--fs-section)/1 var(--sans);color:var(--text-mute);
text-transform:uppercase;letter-spacing:var(--tr-label);white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;}
.stamp{font:var(--fs-section)/1 var(--mono);color:var(--text-dim);
font-variant-numeric:tabular-nums;background:var(--surface-2);
border:1px solid var(--border);border-radius:var(--r-pill);
padding:6px var(--sp-3);white-space:nowrap;}
main{max-width:var(--measure);margin:0 auto;padding:0 var(--gutter) var(--sp-16);}
main>*:last-child{margin-bottom:0;}
h1{font:700 var(--fs-title)/1.12 var(--sans);letter-spacing:var(--tr-neg);
margin:var(--sp-10) 0 var(--sp-3);color:var(--text);}
h2{font:600 var(--fs-section)/1 var(--sans);text-transform:uppercase;
letter-spacing:var(--tr-section);color:var(--text-mute);
margin:var(--sp-12) 0 var(--sp-4);padding:0 0 var(--sp-3);
border-bottom:1px solid var(--border);}
.sub{color:var(--text-dim);font-size:var(--fs-body);line-height:var(--lh-base);
margin:0 0 var(--sp-8);max-width:80ch;}
strong{color:var(--text);font-weight:600;}
code{font:var(--fs-3) var(--mono);color:var(--text);background:var(--surface-2);
padding:2px var(--sp-1);border-radius:var(--r-1);
border:1px solid var(--border);}
a{color:var(--accent);text-decoration:none;
transition:color var(--dur) var(--ease),opacity var(--dur) var(--ease);}
a:hover{text-decoration:underline;}
a:focus-visible,[tabindex]:focus-visible{outline:none;box-shadow:var(--ring);
border-radius:var(--r-1);}
ul{margin:0 0 var(--sp-4);padding-left:var(--sp-5);}
li{margin:0 0 var(--sp-2);color:var(--text-dim);}
.k{font:600 var(--fs-label)/1 var(--sans);text-transform:uppercase;
letter-spacing:var(--tr-label);color:var(--text-mute);}
.tile,.cards a,.versus .side,.compare .col,.headline .big{
background:var(--card);border:1px solid var(--border);
border-radius:var(--r-3);box-shadow:var(--shadow);
transition:border-color var(--dur) var(--ease),
box-shadow var(--dur) var(--ease),transform var(--dur) var(--ease);}
.tile:hover,.versus .side:hover,.compare .col:hover{border-color:var(--border-hi);}
table{border-collapse:collapse;width:100%;font-size:var(--fs-table);
font-variant-numeric:tabular-nums;margin:0 0 var(--sp-4);}
th{font:600 var(--fs-label)/1 var(--sans);text-transform:uppercase;
letter-spacing:var(--tr-label);color:var(--text-mute);text-align:left;
padding:var(--sp-3);border-bottom:1px solid var(--border-hi);}
td{padding:var(--sp-3);border-bottom:1px solid var(--rule);color:var(--text-dim);
transition:background var(--dur) var(--ease);}
td:first-child{color:var(--text);}
td.n,th.n{text-align:right;font-family:var(--mono);
font-variant-numeric:tabular-nums;}
td.n{color:var(--text);}
tbody tr:hover td,table tr:hover td{background:var(--row-hover);}
td.good{color:var(--good);}
td.bad{color:var(--bad);}
tr.arm-arc td:first-child{color:var(--accent);font-weight:600;}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
gap:var(--sp-4);margin:0 0 var(--sp-4);}
.tile{padding:var(--sp-5) var(--sp-6);}
.tile:hover{box-shadow:var(--shadow-lift);transform:translateY(-1px);}
.tile .v{margin-top:var(--sp-3);
font:600 var(--fs-card)/var(--lh-tight) var(--mono);
font-variant-numeric:tabular-nums;letter-spacing:var(--tr-neg);color:var(--text);}
.tile.point{border-color:var(--accent-dim);box-shadow:var(--shadow),var(--glow);}
.tile.point:hover{border-color:var(--accent);}
.tile.point .v{color:var(--accent);}
@property --n{syntax:"<integer>";initial-value:0;inherits:false;}
@keyframes settle{from{opacity:0;transform:translateY(var(--sp-2));
filter:blur(6px);}to{opacity:1;transform:none;filter:blur(0);}}
@keyframes tick{from{--n:0;}}
@keyframes hand_off{0%,68%{opacity:1;}100%{opacity:0;}}
.tile{--d:0s;}
.tile:nth-child(1){--d:0.05s;}
.tile:nth-child(2){--d:0.13s;}
.tile:nth-child(3){--d:0.21s;}
.tile:nth-child(4){--d:0.29s;}
.tile:nth-child(5){--d:0.37s;}
.tile:nth-child(6){--d:0.45s;}
.tile:nth-child(7){--d:0.53s;}
.tile:nth-child(8){--d:0.61s;}
.tile .v{animation:settle 0.55s var(--ease) both var(--d);}
.tile .v.count{position:relative;}
.tile .v.count::after{content:counter(c);counter-reset:c var(--n);
position:absolute;inset:0;background:var(--surface);
animation:tick 0.9s var(--ease) both var(--d),
hand_off 0.9s linear both var(--d);}
.badge{display:inline-block;padding:3px var(--sp-2);border-radius:var(--r-1);
font:600 var(--fs-label)/1.5 var(--mono);letter-spacing:var(--tr-3);
white-space:nowrap;border:1px solid transparent;}
.badge-law{color:var(--badge-law-fg) !important;
background:var(--badge-law-bg) !important;border-color:var(--badge-law-line);}
.badge-network{color:var(--badge-net-fg) !important;
background:var(--badge-net-bg) !important;border-color:var(--badge-net-line);}
.badge-ours{color:var(--badge-ours-fg) !important;
background:var(--badge-ours-bg) !important;border-color:var(--badge-ours-line);}
.badge-provisional{color:var(--badge-prov-fg) !important;
background:var(--badge-prov-bg) !important;border-color:var(--badge-prov-line);}
.bar{height:10px;border-radius:var(--r-2);background:var(--surface);
border:1px solid var(--border);overflow:hidden;display:flex;}
.bar span{display:block;height:100%;}
.bar span:nth-child(1){
background:linear-gradient(90deg,var(--accent-deep),var(--accent)) !important;}
.bar span:nth-child(2){background:linear-gradient(90deg,#5B6675,#748193) !important;}
.bar span:nth-child(3){background:linear-gradient(90deg,#3A424F,#4B5563) !important;}
.bar span:nth-child(4){background:linear-gradient(90deg,#232A34,#2E3742) !important;}
#g-accent stop:first-child{stop-color:var(--accent-deep);}
#g-accent stop:last-child{stop-color:var(--accent);}
#g-grey stop:first-child{stop-color:var(--bar-grey-a);}
#g-grey stop:last-child{stop-color:var(--bar-grey-b);}
.hero,.arms{overflow:visible;}
.hero .track,.arms .track{fill:var(--surface);}
.hero .track,.hero .fill{rx:6px;height:36px;}
.arms .track,.arms .rec,.arms .inc{rx:6px;height:24px;}
.hero .fill,.arms .rec{fill:url(#g-grey);}
.hero .win .fill,.hero.win .fill,.arms .inc{fill:url(#g-accent);}
.hero .win .fill,.arms .inc{filter:drop-shadow(0 0 12px rgba(240,180,41,0.30));}
.hero>g.hbar:last-of-type .fill{filter:drop-shadow(-3px 0 0 var(--bad));}
.decay{display:block;width:100%;height:auto;margin:0 0 var(--sp-4);}
.decay .grid{stroke:var(--border);stroke-width:1;}
.decay .tick{fill:var(--text-mute);font:600 10px var(--sans);
text-transform:uppercase;letter-spacing:var(--tr-label);}
.decay .series .stroke{fill:none;stroke:var(--bar-grey-b);stroke-width:2.5;
stroke-linejoin:round;stroke-linecap:round;}
.decay .series .dot{fill:var(--bar-grey-b);}
.decay .series .endlabel{fill:var(--text-dim);font:600 12px var(--sans);
text-transform:uppercase;letter-spacing:var(--tr-label);}
.decay .series .endvalue{fill:var(--text);font:600 14px var(--mono);
font-variant-numeric:tabular-nums;}
.decay .series .startvalue{fill:var(--text-mute);font:600 11px var(--mono);
font-variant-numeric:tabular-nums;}
.decay .series.arc .stroke{stroke:var(--accent);stroke-width:3;}
.decay .series.arc .dot{fill:var(--accent);}
.decay .series.arc .endlabel{fill:var(--accent);}
.decay .series.arc .endvalue{fill:var(--accent);}
.legend{display:grid;gap:var(--sp-2);margin:0 0 var(--sp-3);}
.legend .leg{display:grid;grid-template-columns:auto 14ch 1fr auto;
gap:var(--sp-3);align-items:baseline;padding:var(--sp-3) 0;
border-top:1px solid var(--rule);font:var(--fs-table) var(--mono);
font-variant-numeric:tabular-nums;color:var(--text-dim);}
.legend .swatch{width:14px;height:3px;border-radius:var(--r-1);
background:var(--bar-grey-b);align-self:center;}
.legend .arc .swatch{background:linear-gradient(90deg,var(--accent-deep),var(--accent));}
.legend .name{color:var(--text);text-transform:uppercase;
letter-spacing:var(--tr-label);font-weight:600;font-size:var(--fs-label);
font-family:var(--sans);}
.legend .arc .name{color:var(--accent);}
.legend .share{color:var(--text-mute);}
.arms{display:block;width:100%;margin:var(--sp-2) 0 var(--sp-4);}
.arms .lab{fill:var(--text-dim);font:600 9px var(--sans);
text-transform:uppercase;letter-spacing:var(--tr-label);}
.arms .val{fill:var(--text);font:600 var(--fs-label) var(--mono);
font-variant-numeric:tabular-nums;}
.arms .row:hover .rec{fill:var(--text-mute);}
.headline{display:grid;grid-template-columns:2fr 1fr;gap:var(--sp-4);
margin:0 0 var(--sp-4);align-items:stretch;}
.headline .big{border-color:var(--accent-dim);
box-shadow:var(--shadow),var(--glow);padding:var(--sp-6) var(--sp-8);
display:flex;flex-direction:column;justify-content:center;}
.headline .big:hover{border-color:var(--accent);}
.headline .big .v{font:700 var(--fs-headline)/var(--lh-tight) var(--mono);
letter-spacing:var(--tr-neg);color:var(--accent);margin-top:var(--sp-3);
font-variant-numeric:tabular-nums;}
.headline .big .ci{color:var(--text-dim);font:var(--fs-table) var(--mono);
font-variant-numeric:tabular-nums;margin-top:var(--sp-3);}
.headline .big .den{color:var(--text-mute);
font:var(--fs-3)/var(--lh-snug) var(--sans);margin-top:var(--sp-3);
max-width:56ch;}
.headline .side{display:grid;gap:var(--sp-4);align-content:stretch;}
.headline .side .tile{display:flex;flex-direction:column;justify-content:center;}
.versus{display:grid;grid-template-columns:1fr auto 1fr;gap:var(--sp-4);
align-items:stretch;margin:0 0 var(--sp-4);}
.versus .side{padding:var(--sp-5) var(--sp-6);}
.versus .side.win{border-color:var(--accent-dim);
box-shadow:var(--shadow),var(--glow);}
.versus .side .v{margin-top:var(--sp-3);
font:600 var(--fs-card)/var(--lh-tight) var(--mono);
font-variant-numeric:tabular-nums;color:var(--text);}
.versus .side.win .v{color:var(--accent);}
.versus .side.lose .v{color:var(--bad);}
.versus .gap{display:flex;align-items:center;justify-content:center;
font:600 var(--fs-card) var(--mono);color:var(--accent);padding:0 var(--sp-3);
white-space:nowrap;}
.hero-row{display:grid;grid-template-columns:minmax(0,2.1fr) minmax(240px,1fr);
gap:var(--sp-6);align-items:start;margin:0 0 var(--sp-4);}
.hero-side{display:grid;gap:var(--sp-4);align-content:start;}
.hero{display:block;width:100%;height:auto;margin:0 0 var(--sp-4);}
.hero .lab{fill:var(--text-dim);font:600 12px var(--sans);
text-transform:uppercase;letter-spacing:var(--tr-label);}
.hero .val{fill:var(--text);font:600 15px var(--mono);
font-variant-numeric:tabular-nums;}
.hero .win .lab{fill:var(--accent);}
.hero .win .val{fill:var(--accent);}
.compare{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
gap:var(--sp-4);margin:0 0 var(--sp-4);}
.compare .col{padding:var(--sp-5) var(--sp-6);display:flex;
flex-direction:column;justify-content:center;}
.compare .col.win{border-color:var(--accent-dim);
box-shadow:var(--shadow),var(--glow);}
.compare .col .v{margin-top:var(--sp-3);
font:600 var(--fs-card)/var(--lh-tight) var(--mono);
font-variant-numeric:tabular-nums;letter-spacing:var(--tr-neg);color:var(--text);}
.compare .col.win .v{color:var(--accent);font-size:var(--fs-headline);
font-weight:700;}
.finding{color:var(--text);font:600 var(--fs-body)/var(--lh-snug) var(--sans);
margin:0 0 var(--sp-3);padding:var(--sp-4) 0 0;max-width:74ch;
border-top:1px solid var(--border);}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(232px,1fr));
gap:var(--sp-4);margin:0 0 var(--sp-3);}
.cards a{position:relative;padding:var(--sp-5) var(--sp-6);color:var(--text);
text-decoration:none;display:flex;flex-direction:column;gap:var(--sp-2);
overflow:hidden;}
.cards a::before{content:"";position:absolute;inset:0 0 auto 0;height:2px;
background:linear-gradient(90deg,var(--accent-deep),var(--accent));opacity:0;
transition:opacity var(--dur) var(--ease);}
.cards a:hover{border-color:var(--border-hi);box-shadow:var(--shadow-lift);
transform:translateY(-2px);text-decoration:none;}
.cards a:hover::before{opacity:1;}
.cards a:focus-visible{box-shadow:var(--ring);border-radius:var(--r-3);}
.cards a .t{font:600 var(--fs-body)/1.2 var(--sans);}
.cards a .d{color:var(--text-dim);font-size:var(--fs-table);
line-height:var(--lh-snug);margin-top:0;}
.facts{list-style:none;margin:0 0 var(--sp-3);padding:0;
border-top:1px solid var(--border);}
.facts li{display:grid;grid-template-columns:12ch 1fr;gap:var(--sp-5);
align-items:baseline;padding:var(--sp-4) 0;
border-bottom:1px solid var(--rule);color:var(--text-dim);margin:0;
font-size:var(--fs-body);transition:background var(--dur) var(--ease);}
.facts li:hover{background:var(--row-hover);}
.facts li .n{font:600 var(--fs-body)/1 var(--mono);color:var(--accent);
font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right;}
.tl{margin:0;padding:0 0 0 var(--sp-6);border-left:1px solid var(--border);
list-style:none;}
.tl>li{position:relative;margin:0 0 var(--sp-6);padding:0;color:var(--text);}
.tl>li:last-child{margin-bottom:0;}
.tl>li::before{content:"";position:absolute;left:-29px;top:5px;width:9px;
height:9px;border-radius:var(--r-pill);background:var(--bg);
border:1px solid var(--text-mute);
transition:background var(--dur) var(--ease),
box-shadow var(--dur) var(--ease);}
.tl>li.hit::before{background:var(--accent);border-color:var(--accent);
box-shadow:var(--glow);}
.tl .stage{font:600 var(--fs-label)/1 var(--sans);text-transform:uppercase;
letter-spacing:var(--tr-section);color:var(--text-mute);margin:0 0 var(--sp-3);}
.tl dl{display:grid;grid-template-columns:max-content 1fr;
gap:var(--sp-1) var(--sp-5);margin:0 0 var(--sp-3);}
.tl dt{font:600 var(--fs-label)/1.5 var(--sans);text-transform:uppercase;
letter-spacing:var(--tr-label);color:var(--text-mute);}
.tl dd{margin:0;font:var(--fs-table)/1.5 var(--mono);
font-variant-numeric:tabular-nums;color:var(--text);}
.tl .line{color:var(--text-dim);font-size:var(--fs-table);
line-height:var(--lh-base);margin:0;max-width:74ch;}
.tl table{margin:0 0 var(--sp-3);}
.note{color:var(--text-dim);font-size:var(--fs-table);
line-height:var(--lh-base);margin:var(--sp-3) 0 0;max-width:80ch;}
.endpoint{color:var(--text);font:600 var(--fs-table)/var(--lh-snug) var(--mono);
font-variant-numeric:tabular-nums;margin:var(--sp-3) 0 0;}
.prose p{margin:0 0 var(--sp-4);max-width:74ch;color:var(--text-dim);}
.prose .step{border-left:2px solid var(--border-hi);padding:0 0 0 var(--sp-5);
margin:0 0 var(--sp-5);}
@media (max-width:900px){
.headline{grid-template-columns:1fr;}
.versus{grid-template-columns:1fr;}
.versus .gap{justify-content:flex-start;}
.hero-row{grid-template-columns:1fr;}
}
@media (max-width:600px){
:root{--gutter:16px;--fs-title:28px;--fs-headline:30px;}
.brand .full{display:none;}
}
@media (prefers-reduced-motion:reduce){
*{animation:none !important;transition:none !important;}
.tile:hover,.cards a:hover{transform:none;}
.tile .v.count::after{display:none;}
}
"""


# The two bar gradients, defined once per document and referenced from the
# stylesheet by `fill:url(#g-accent)`.
#
# WHY THIS MARKUP EXISTS AT ALL. The bars on the landing page and the scoreboard
# are SVG `<rect>` elements, and `fill` takes a paint value - a colour or a
# reference to a paint server. It does not take a CSS gradient, so there is no
# arrangement of the stylesheet alone that puts a gradient on those bars. This
# is the paint server; the stylesheet still decides which bar gets which.
_BAR_GRADIENTS = (
    '<svg class="defs" aria-hidden="true" focusable="false"><defs>'
    '<linearGradient id="g-accent" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#D19A1F"/><stop offset="1" stop-color="#F0B429"/>'
    "</linearGradient>"
    '<linearGradient id="g-grey" x1="0" y1="0" x2="1" y2="0">'
    '<stop offset="0" stop-color="#444D5C"/><stop offset="1" stop-color="#586374"/>'
    "</linearGradient>"
    "</defs></svg>"
)


def run_stamp(seed: int | None) -> str:
    """The provenance pill for the top bar: which run these figures came from.

    THE DIGEST IS ONLY SHOWN FOR THE RUN IT DESCRIBES. `JUDGED_DIGEST` is the
    hash of the judged seed's output, and `python -m arc.console.build` defaults
    to the DEVELOP seed - so stamping it unconditionally would put a seed-3
    claim on the header of every page of a seed-1 console, on all five screens
    at once. A seed the reader can see is worth more than a hash they cannot
    check, so a non-judged run says which seed it was and stops there.
    """
    if seed is None:
        return ""
    if seed == JUDGED_SEED:
        return f"seed {seed} &middot; {escape(JUDGED_DIGEST[:8])}"
    return f"seed {seed}"


def document(title: str, body: str, *, stamp: str = "") -> str:
    """One screen, as a self-contained document.

    THE TOP BAR IS ON EVERY SCREEN AND CARRIES THE PROVENANCE. Four screens that
    each open from disk have no shared navigation and no address bar worth
    reading, so without it a reader three clicks in has nothing telling them
    which run they are looking at. Putting the seed and the digest in a fixed
    position makes the reproducibility claim answerable at a glance instead of
    only on the landing page.
    """
    pill = f'<div class="stamp">{stamp}</div>' if stamp else ""
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body>{_BAR_GRADIENTS}"
        '<header class="topbar"><div class="topbar-in">'
        '<div class="brand"><span class="mark">ARC</span>'
        '<span class="full">autonomous revenue continuity</span></div>'
        f"{pill}</div></header>"
        f"<main>{body}</main></body></html>"
    )


def _tile(key: str, value: str, *, point: bool = False, count: int | None = None) -> str:
    """One readout tile. `count` makes the figure count up when the page opens.

    Pass it only for whole numbers. `value` stays the rendered text and is what
    the reader is left with; `count` is the integer the overlay animates to
    before handing back to it. Money is never counted up - the overlay cannot
    render paise or digit grouping, and a figure in rupees that is briefly
    wrong is worse than one that simply appears.
    """
    cls = "tile point" if point else "tile"
    if count is None:
        inner = f'<div class="v">{escape(value)}</div>'
    else:
        inner = f'<div class="v count" style="--n:{int(count)}">{escape(value)}</div>'
    return f'<div class="{cls}"><div class="k">{escape(key)}</div>{inner}</div>'


def _decay_chart(series: Sequence[tuple[str, Sequence[int], bool]]) -> str:
    """Both arms on ONE axis, because the crossover is the finding.

    WHAT WAS WRONG BEFORE. Each arm was drawn in its own box. The scale was
    shared arithmetically, but two charts side by side with
    `preserveAspectRatio="none"` stretch to their own widths, so the reader had
    no way to see that the lines cross. They do cross, and that is the entire
    argument for the constraints: the unconstrained arm starts ahead and burns
    the population down, and by the last cycle the constrained arm it started
    behind is collecting more than it is.

    ONE AXIS, ZERO-BASED. A shared maximum is not enough - the baseline has to
    be zero too, or a truncated axis exaggerates whichever line happens to sit
    near the bottom. Endpoints are labelled on the plot so the crossing does
    not need a legend to be read.
    """
    live = [(name, list(values), accent) for name, values, accent in series if len(values) > 1]
    if not live:
        return ""
    top = max(max(v) for _, v, _ in live) or 1
    cycles = max(len(v) for _, v, _ in live)

    width, height = 760.0, 300.0
    pad_l, pad_r, pad_t, pad_b = 12.0, 208.0, 28.0, 34.0
    span_x = width - pad_l - pad_r
    span_y = height - pad_t - pad_b
    step = span_x / (cycles - 1)
    floor = height - pad_b

    def at(index: int, value: int) -> tuple[float, float]:
        return pad_l + index * step, floor - span_y * (value / top)

    grid = "".join(
        f'<line class="grid" x1="{pad_l:.1f}" y1="{floor - span_y * frac:.1f}" '
        f'x2="{pad_l + span_x:.1f}" y2="{floor - span_y * frac:.1f}"/>'
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    ticks = "".join(
        f'<text class="tick" x="{pad_l + i * step:.1f}" y="{height - 10:.1f}" '
        f'text-anchor="middle">cycle {i + 1}</text>'
        for i in range(cycles)
    )

    body = ""
    for name, values, accent in live:
        points = [at(i, v) for i, v in enumerate(values)]
        line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
        css = "series arc" if accent else "series"
        dots = "".join(
            f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="3.5"/>' for x, y in points
        )
        end_x, end_y = points[-1]
        body += (
            f'<g class="{css}">'
            f'<path class="stroke" d="{line}"/>{dots}'
            f'<text class="endlabel" x="{end_x + 12:.1f}" y="{end_y - 4:.1f}">'
            f"{escape(name)}</text>"
            f'<text class="endvalue" x="{end_x + 12:.1f}" y="{end_y + 13:.1f}">'
            f"{escape(format_inr(Paise(values[-1])))}</text>"
            f'<text class="startvalue" x="{points[0][0] + 8:.1f}" '
            f'y="{points[0][1] - 10:.1f}">{escape(format_inr(Paise(values[0])))}</text>'
            "</g>"
        )

    return (
        f'<svg class="decay" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="recovery per cycle, both arms on one axis">'
        f'<g class="axis">{grid}{ticks}</g>{body}</svg>'
    )


def _arm_bars(arms: Sequence[Mapping[str, object]]) -> str:
    """Five arms compared on recovery, with the incremental part called out."""
    rows = [
        (str(a["arm"]), int(a["recovered_paise"]), max(int(a["incremental_paise"]), 0))
        for a in arms
    ]
    if not rows:
        return ""
    top = max(recovered for _, recovered, _ in rows) or 1
    label_w, right_w, row_h, gap = 132.0, 96.0, 16.0, 12.0
    width = 720.0
    span = width - label_w - right_w
    height = len(rows) * (row_h + gap)

    body = ""
    for index, (name, recovered, incremental) in enumerate(rows):
        y = index * (row_h + gap)
        mid = y + row_h / 2 + 3.5
        body += (
            f'<g class="row">'
            f'<text class="lab" x="0" y="{mid:.1f}">{escape(name)}</text>'
            f'<rect class="track" x="{label_w}" y="{y:.1f}" width="{span:.1f}" '
            f'height="{row_h}" rx="2"/>'
            f'<rect class="rec" x="{label_w}" y="{y:.1f}" '
            f'width="{span * recovered / top:.1f}" height="{row_h}" rx="2"/>'
            f'<rect class="inc" x="{label_w}" y="{y:.1f}" '
            f'width="{span * incremental / top:.1f}" height="{row_h}" rx="2"/>'
            f'<text class="val" x="{width:.0f}" y="{mid:.1f}" text-anchor="end">'
            f"{escape(format_inr(Paise(recovered)))}</text>"
            f"</g>"
        )
    return (
        f'<svg class="arms" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="recovery by arm">{body}</svg>'
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
            + _tile("claims", f"{self.claims:,}", count=self.claims)
            + _tile("subjects", f"{self.subjects:,}", count=self.subjects)
            + _tile("at risk", format_inr(self.at_risk_paise))
            + _tile(
                "suppressed by outage",
                f"{self.suppressed_by_outage:,}",
                point=True,
                count=self.suppressed_by_outage,
            )
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
        return document(f"ARC batch - seed {self.seed}", body, stamp=run_stamp(self.seed))


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
    declined: int = 0
    # Carried for the top bar's provenance pill only. Optional, so a test that
    # builds a funnel to assert on the funnel does not have to name a run.
    seed: int | None = None

    def __post_init__(self) -> None:
        accounted = self.blocked + self.deferred + self.declined + self.executed
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

    def _outside_registry(self) -> str:
        """Refusers that are not compliance rules, counted where they are visible.

        THE SCREEN WAS KEEPING A COUNT IT NEVER SHOWED. The per-rule table
        iterates the REGISTRY, so a refusal recorded against anything else -
        the allocator's admission step, in this batch every single one of them
        - was tallied and then rendered nowhere. A funnel that reports 609
        refusals above a table of thirty-three zeroes invites the reader to
        conclude the table is broken. It is not: those refusals were never
        compliance decisions, and saying so is the honest fix rather than
        quietly folding them into a rule that did not make them.
        """
        known = {badge.rule_id for badge in self.badges}
        outside = [c for c in self.counters if c.rule_id not in known]
        if not outside:
            return ""
        rows = "".join(
            f"<tr><td>{escape(c.rule_id)}</td>"
            f"<td>the allocator&rsquo;s admission step, a budget limit and "
            f"not a compliance rule</td>"
            f"<td class='n'>{c.fired:,}</td>"
            f"<td>{escape(c.verdict.value)}</td></tr>"
            for c in outside
        )
        return (
            "<h2>Refused, but not by a rule</h2>"
            "<table><tr><th>source</th><th>what it is</th><th class='n'>fired</th>"
            "<th>verdict</th></tr>" + rows + "</table>"
            '<p class="note">Budget headroom is not compliance. These refusals are '
            "counted here rather than attributed to a rule that did not make them.</p>"
        )

    def _funnel_note(self) -> str:
        """Say what the categories mean, and never show a bare zero.

        A ZERO IS A CLAIM AND HAS TO BE EXPLAINED. `blocked` at zero does not
        mean the Gate was absent; it means no compliance rule refused a branch
        that had already survived the Gate's eligibility projection, because
        projection prunes ineligible actions before they are ever scored. Left
        as a bare 0 the reader is invited to conclude the firewall did nothing.
        """
        total = self.blocked + self.deferred + self.declined + self.executed
        parts = [
            f"{total:,} of {self.proposed:,} proposals are accounted for: every "
            "proposal is either refused, declined by the policy, or executed."
        ]
        if self.deferred and not self.blocked:
            parts.append(
                f"All {self.deferred:,} refusals this batch were DEFER, from the "
                "allocator's admission step running out of budget headroom mid-cycle. "
                "A deferred action is one that did not happen this cycle and may "
                "happen in a later one under a fresh decision."
            )
            parts.append(
                "Blocked is zero because no compliance rule refused an action that "
                "had already survived the Gate's eligibility projection - projection "
                "prunes ineligible actions before they are scored, so the temporal "
                "rules do their work earlier than this funnel can see."
            )
        elif not self.deferred and not self.blocked:
            parts.append(
                "Nothing was refused in this batch. That is a fact about this "
                "population, not evidence that the rules were not consulted."
            )
        return " ".join(parts)

    def render(self) -> str:
        mix = self.mix
        fired = {c.rule_id: c for c in self.counters}
        rows = ""
        # BY WORK DONE, NOT BY NAME. Registry order put four never-fired rules
        # at the top and buried the one that did every refusal in this batch.
        # Ties fall back to the id so the order is still deterministic.
        ordered = sorted(
            self.badges,
            key=lambda b: (-(fired[b.rule_id].fired if b.rule_id in fired else 0), b.rule_id),
        )
        for badge in ordered:
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
            "<h2>Funnel &mdash; every proposal accounted for</h2>"
            '<div class="tiles">'
            + _tile("proposed", f"{self.proposed:,}", count=self.proposed)
            + _tile("blocked", f"{self.blocked:,}", count=self.blocked)
            + _tile("deferred", f"{self.deferred:,}", point=self.deferred > 0, count=self.deferred)
            + _tile("declined", f"{self.declined:,}", count=self.declined)
            + _tile("executed", f"{self.executed:,}", count=self.executed)
            + "</div>"
            + f'<p class="note">{self._funnel_note()}</p>'
            "<h2>The honest mix</h2>"
            f'<p class="sub">{mix["total"]} rules: {mix["statutory"]} statutory, '
            f"{mix['network_rule']} network, {mix['policy_choice']} our own policy "
            f"choice. {mix['in_force']} in force, {mix['draft']} draft, "
            f"{mix['advisory']} advisory, {mix['contested']} contested. We are "
            f"deliberately stricter than the binding minimum in "
            f"{mix['stricter_than_binding_minimum']} places.</p>"
            f'<p class="note">{legend}</p>'
            "<h2>Not in force, and applied anyway</h2>"
            f"<ul>{pending}</ul>" + self._outside_registry() + "<h2>Every rule, with its force</h2>"
            "<table><tr><th>rule</th><th>force</th><th class='n'>fired</th>"
            "<th>verdict</th></tr>" + rows + "</table>"
        )
        rendered = document("ARC compliance firewall", body, stamp=run_stamp(self.seed))
        # THE HONESTY AUDIT, ON THE WAY OUT. Checked against what was actually
        # rendered rather than against the model that produced it.
        assert_no_overstated_force(rendered, self.registry)
        assert_every_rule_shown(rendered, self.registry)
        return rendered


# What "spend" is, said once and rendered on every screen that divides by it.
# A ratio with an unstated denominator invites the reader to supply their own,
# and the one they supply will include compute and salaries.
_SPEND_DENOMINATOR = (
    "Spend is marginal channel cost only: messaging, retries and voice minutes. "
    "Compute, human-tier time and amortised build are not in the denominator."
)


def _harm_tone(
    rails: Mapping[str, object],
    baseline: Mapping[str, object] | None,
    key: str,
) -> str:
    """Colour a guardrail cell green or red AGAINST THE COMPARATOR, or not at all.

    THE COMPARISON IS COMPUTED, NEVER ASSERTED BY THE STYLESHEET. A rule that
    painted ARC's complaint rate green would be making a claim about the data
    from inside the CSS, and it would go on making it on the run where ARC
    loses. So the arithmetic happens here, against the same comparator the
    headline is stated against, and a cell that is not better or worse than the
    comparator gets no colour at all.

    The comparator's own row is never coloured - an arm cannot beat itself, and
    a green baseline would read as an endorsement of the arm this whole
    scoreboard exists to beat.
    """
    if baseline is None or rails is baseline:
        return ""
    observed = float(rails[key])  # type: ignore[arg-type]
    reference = float(baseline[key])  # type: ignore[arg-type]
    if observed < reference:
        return " good"
    if observed > reference:
        return " bad"
    return ""


def _multiple(numerator: float, denominator: float) -> int:
    """How many times bigger, from the figures AS PRINTED, rounded down.

    TWO DELIBERATE CHOICES. The multiple is computed from the values rounded
    the way the screen shows them, so a reader who divides 0.027 by 0.003 gets
    the number beside them; taking it from the raw floats gave 9.65 and printed
    10 next to two figures that divide to 9. And it rounds DOWN, because on a
    screen whose subject is honesty the tolerable error is the one that
    understates our own result.
    """
    if denominator <= 0:
        return 0
    shown_num, shown_den = round(numerator, 3), round(denominator, 3)
    if shown_den <= 0:
        return 0
    return int(shown_num / shown_den)


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

    def _decay_section(self) -> str:
        """Recovery per cycle, with each curve labelled by where it ends.

        WHY THIS SITS DIRECTLY UNDER THE HEADLINE. It is the only thing on the
        scoreboard that shows change over TIME, and it is the argument for the
        constraints: the unconstrained arm contacts everyone every cycle and
        burns the population down doing it. At the bottom of the page nobody
        reached it inside a five-minute demo.

        THE LABEL IS THE POINT, NOT THE LINE. A sparkline shows a shape; the
        reader needs the shape's consequence. So each series states where it
        finishes as a share of where it started, which is the comparison the
        curves are drawn to support.
        """
        if not self.decay:
            return ""
        ordered = sorted(self.decay.items(), key=lambda kv: kv[0] is Arm.ARC)
        chart = _decay_chart(
            [(arm.value.replace("_", " "), list(values), arm is Arm.ARC) for arm, values in ordered]
        )
        legs = ""
        for arm, values in ordered:
            if not values:
                continue
            first, last = values[0], values[-1]
            share = (100.0 * last / first) if first else 0.0
            css = "leg arc" if arm is Arm.ARC else "leg"
            legs += (
                f"<div class='{css}'><span class='swatch'></span>"
                f"<span class='name'>{escape(arm.value.replace('_', ' '))}</span>"
                f"<span class='track'>"
                f"{' &rarr; '.join(format_inr(Paise(v)) for v in values)}</span>"
                f"<span class='share'>ends at {share:.0f}% of cycle 1</span></div>"
            )
        out = chart + f"<div class='legend'>{legs}</div>"
        return out

    def _headline(self, arms: Sequence[Mapping[str, object]]) -> str:
        """The graded number, its interval, and the error on the held-out seed.

        THE SCREEN THAT IS GRADED HAS TO SHOW THE GRADED NUMBER. It was in the
        table as one cell among ninety. Incremental per rupee spent is the
        measurement; the interval is what says whether it is real; and the
        estimator error is what says whether the measurement can be trusted at
        all. The judged-seed error is the worse of the two and is the one in
        the accent - reporting the better number would be choosing the seed
        after seeing the result.
        """
        by_arm = {str(a["arm"]): a for a in arms}
        arc = by_arm.get("arc")
        greedy = by_arm.get("greedy_unconstrained")
        if arc is None:
            return ""

        per_rupee = float(arc["incremental_per_rupee_spent"])  # type: ignore[arg-type]
        interval = arc.get("ci_95_paise")
        ci = (
            f"{format_inr(Paise(int(arc['incremental_paise'])))} incremental. "  # type: ignore[index]
            f"Bootstrap 95% CI on recovered rupees "
            f"{format_inr(Paise(int(interval[0])))} to "  # type: ignore[index]
            f"{format_inr(Paise(int(interval[1])))}, subjects resampled as clusters"
            if interval
            else "no interval computed"
        )
        head = (
            '<div class="headline">'
            '<div class="big"><div class="k">incremental recovered per rupee spent, '
            "against naive dunning</div>"
            f'<div class="v">{per_rupee:.2f}x</div>'
            f'<div class="ci">{escape(ci)}</div>'
            f'<div class="den">{escape(_SPEND_DENOMINATOR)}</div></div>'
            '<div class="side">'
            + _tile("DR error, develop seed", f"{self.dr_error_develop * 100:.2f}%")
            + _tile(
                f"DR error, judged seed {self.judged_seed}",
                f"{self.dr_error_judged * 100:.2f}%",
                point=True,
            )
            + "</div></div>"
            '<p class="note">Both estimator errors are shown and the judged seed is '
            "the worse one. Reporting only the develop figure would be selecting the "
            "seed after seeing the result, which is the thing the three-seed "
            "discipline exists to prevent.</p>"
        )

        if greedy is None:
            return head
        arc_cost = float(arc["guardrails"]["cost_per_rupee_collected"])  # type: ignore[index]
        greedy_cost = float(greedy["guardrails"]["cost_per_rupee_collected"])  # type: ignore[index]
        ratio = _multiple(greedy_cost, arc_cost)
        return head + (
            "<h2>Cost per rupee collected &mdash; the efficiency gap</h2>"
            '<div class="versus">'
            '<div class="side win"><div class="k">arc</div>'
            f'<div class="v">{arc_cost:.3f}</div></div>'
            f'<div class="gap">{ratio:d}x cheaper</div>'
            '<div class="side"><div class="k">greedy unconstrained</div>'
            f'<div class="v">{greedy_cost:.3f}</div></div>'
            "</div>"
            '<p class="note">Spending more to collect the same rupee is not a '
            "strategy. This was a cell in the middle of the table; it is the second "
            "result on the screen.</p>"
        )

    def render(self) -> str:
        payload = self.payload()
        arms = payload["arms"]
        comparator = str(payload["comparator"])
        baseline = next(
            (a["guardrails"] for a in arms if a["arm"] == comparator),  # type: ignore[union-attr,index]
            None,
        )
        rows = ""
        for arm in arms:  # type: ignore[union-attr]
            rails = arm["guardrails"]
            rows += (
                f"<tr class='arm arm-{escape(str(arm['arm']))}'>"
                f"<td>{escape(arm['arm'])}</td>"
                f"<td class='n'>{format_inr(Paise(arm['recovered_paise']))}</td>"
                f"<td class='n'>{format_inr(Paise(arm['incremental_paise']))}</td>"
                f"<td class='n'>{format_inr(Paise(arm['spend_paise']))}</td>"
                f"<td class='n{_harm_tone(rails, baseline, 'complaint_rate_per_1000')}'>"
                f"{rails['complaint_rate_per_1000']:.2f}</td>"
                f"<td class='n{_harm_tone(rails, baseline, 'opt_out_rate_per_1000')}'>"
                f"{rails['opt_out_rate_per_1000']:.2f}</td>"
                f"<td class='n'>{rails['voluntary_cancel_rate_treated']:.3f}</td>"
                f"<td class='n'>{rails['cost_per_rupee_collected']:.3f}</td>"
                f"<td class='n'>{rails['promise_kept_rate']:.2f}</td>"
                f"<td class='n'>{format_inr(Paise(arm['prevented_paise']))}</td>"
                "</tr>"
            )

        decay = self._decay_section()

        body = (
            "<h1>Scoreboard</h1>"
            f'<p class="sub">Incremental against {escape(payload["comparator"])}, '
            f"seed {payload['seed']}, {payload['cycles']} cycles. Denominator: "
            f"{escape(arms[0]['denominator'])}</p>"  # type: ignore[index]
            + self._headline(arms)  # type: ignore[arg-type]
            + "<h2>Recovery per cycle &mdash; what the constraints buy</h2>"
            + f'<div class="tiles">{decay}</div>'
            + '<p class="note">The unconstrained arm contacts everyone every cycle '
            "and burns the population down doing it: its recovery decays as the "
            "response model's annoyance term bites, and by the fourth cycle it is "
            "collecting less than the constrained arm it started ahead of. That "
            "crossover is the argument for the constraints, and it is why beating "
            "it on net value is a result rather than an accident.</p>"
            + "<h2>Arms &mdash; recovery and guardrails, one table</h2>"
            "<table><tr><th>arm</th><th class='n'>recovered</th>"
            "<th class='n'>incremental</th><th class='n'>spend</th>"
            "<th class='n'>compl/1k</th><th class='n'>optout/1k</th>"
            "<th class='n'>cancel</th><th class='n'>cost/&#8377;</th>"
            "<th class='n'>ptp kept</th><th class='n'>prevented</th></tr>" + rows + "</table>"
            '<p class="note">Prevention is the last column and is NEVER added into '
            "recovery. Money that never failed was never recovered.</p>"
            "<h2>Recovery by arm &mdash; the incremental part in the accent</h2>"
            + _arm_bars(arms)  # type: ignore[arg-type]
            + '<p class="note">Full bar is what the arm recovered; the accent segment '
            "is what it recovered above the comparator. The same two figures as the "
            "first two columns of the table, drawn so the gap is visible from the "
            "back of a room.</p>"
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
        )
        return document(
            "ARC scoreboard",
            body,
            stamp=run_stamp(int(payload["seed"])),  # type: ignore[call-overload]
        )


# ---------------------------------------------------------------------------
# 4. Replay
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One step of a decision, as a label, its numbers, and one line.

    THE NUMBERS ARE ROWS, NOT SENTENCES. A reviewer reading start to finish is
    served by prose; a reader ten feet from a projector needs to find the
    propensity without parsing a clause. Both come from the same trace, so the
    screen and the audit text cannot disagree.
    """

    label: str
    rows: Sequence[tuple[str, str]] = ()
    prose: str = ""
    table: Sequence[Sequence[str]] = ()

    def html(self, *, hit: bool = False) -> str:
        rows = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in self.rows)
        table = ""
        if self.table:
            head, *body = self.table
            cells = "".join(
                f"<th class='n'>{escape(c)}</th>" if i else f"<th>{escape(c)}</th>"
                for i, c in enumerate(head)
            )
            table = "<table><tr>" + cells + "</tr>"
            for line in body:
                table += (
                    "<tr>"
                    + "".join(
                        f"<td class='n'>{escape(c)}</td>" if i else f"<td>{escape(c)}</td>"
                        for i, c in enumerate(line)
                    )
                    + "</tr>"
                )
            table += "</table>"
        prose = f'<p class="line">{escape(self.prose)}</p>' if self.prose else ""
        return (
            f'<li class="{"hit" if hit else ""}">'
            f'<div class="stage">{escape(self.label)}</div>'
            f"{f'<dl>{rows}</dl>' if rows else ''}{table}{prose}</li>"
        )


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
    stages: Sequence[Stage] = ()
    seed: int | None = None

    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def render(self) -> str:
        """The timeline. `text()` remains the prose an auditor reads."""
        stages = "".join(
            stage.html(hit=stage.label in ("Gate verdict", "Decision")) for stage in self.stages
        )
        body = (
            f"<h1>Replay &mdash; claim {escape(self.claim_id[:8])}</h1>"
            f'<p class="sub">subject {escape(self.subject_token)} &mdash; one decision, '
            f"end to end, in the order it happened</p>"
            f'<ol class="tl">{stages}</ol>'
        )
        return document(f"ARC replay - {self.claim_id[:8]}", body, stamp=run_stamp(self.seed))


LAYER_ORDER: tuple[CauseLayer, ...] = (
    CauseLayer.ISSUER,
    CauseLayer.MERCHANT,
    CauseLayer.CUSTOMER,
    CauseLayer.UNKNOWN,
)
