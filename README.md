# ARC — Autonomous Revenue Continuity

Detect revenue at risk, diagnose it, choose a bounded intervention, execute it
compliantly, and prove what was recovered.

Four leak surfaces — failed autopay mandates, declined cards, abandoned
checkouts and overdue B2B invoices — normalise onto one claim type, so a single
contact budget can be shared across all of them. A pure, versioned compliance
Gate authorises every action with a certificate that expires. A doubly-robust
off-policy estimator says what the policy was worth, and reports its own error
against ground truth.

## Documents

| Document | Read it for |
|---|---|
| [`architecture.md`](architecture.md) | The design — every layer, every boundary, and why each one is where it is. |
| [`CLAUDE.md`](CLAUDE.md) | The working conventions that hold in every session. |

## Quick start

```bash
make up        # Postgres 16 via docker compose, waits for healthy
make migrate   # apply migrations/*.sql in order, once each
make test      # pytest
make lint      # ruff check + ruff format --check
```

Requires Python 3.11 (managed by [uv](https://docs.astral.sh/uv/)), Docker, and
GNU make. Everything runs CPU-only.

## Seeing it run

```bash
make demo                 # deterministic replay of the judged run, SEED=3
make demo-live            # real-time and jittered, for the "watch it react" beat
make demo-adversarial     # 20 attacks, each through the real code path
make demo-digest          # just the reproducibility digest
make console              # build the four HTML screens into console/
make validate             # simulator distributions against published anchors
```

`make demo` reads no clock and prints no wall time, so three consecutive runs
produce byte-identical output. The digest at the bottom is a SHA-256 over the
figures the demo prints, and it is pinned in
[`arc/core/reproducibility.py`](arc/core/reproducibility.py) so a change to any
headline number fails a test rather than passing quietly.

`make console` writes four self-contained HTML documents you can open from disk:
the batch, the compliance firewall, the scoreboard and a single decision
replayed end to end. Every figure on them comes from a real run, not a fixture.

## What the current build does

Seed 3, 1,200 claims across 879 subjects, four cycles.

| | |
|---|---|
| Recovered by ARC | ₹7,36,526.52 |
| Incremental against naive dunning | ₹2,78,592.08 |
| Complaints per 1,000 contacts | 4.62 |
| Opt-outs per 1,000 contacts | 13.86 |
| Claims suppressed by a detected issuer outage | 40, contacted zero times |
| Estimator error against ground truth, judged seed | 8.59% |

The industry-default comparator recovered less than doing nothing on this
population. The unconstrained arm recovered more gross rupees than ARC and blew
every guardrail doing it, which is what makes beating it on net value a result
rather than an accident.

Guardrails sit on the same rows as the money because the metrics object refuses
to serialise a recovery figure without them.

## How it is put together

Nine layers, each a package:

```
L0/L1  arc/ingest/         trust boundary, then redaction boundary
L2     arc/sentinel/       cause attribution: whose fault, and can it be fixed silently
L3     arc/forecaster/     bounce, uplift, promise-kept — three models, three techniques
L4     arc/allocator/      portfolio optimisation with priced budgets
L5     arc/gate/           pure compliance evaluation, 33 rules as data
L6     arc/conductor/      exactly-once state transition, breakers, kill switch
L7     arc/channels/       effect the world, decide nothing
L8     arc/proving_ground/ arms, off-policy estimation, the scoreboard
```

Supporting: `arc/core/` (money, time, ids, types), `arc/ledger/` (three stores
plus the PII write-guard), `arc/events/`, `arc/inngest_fns/`, `arc/simulator/`,
`arc/llm_service/`, `arc/voice/`, `arc/console/`, `arc/demo/`.

Read [`architecture.md`](architecture.md) for what each one does and why.

## What is enforced rather than documented

Convention fails under deadline pressure, so the load-bearing rules are tests
that fail the build:

- 22 import bans, AST-walked, resolving relative and dynamic imports
- one clock — nothing but `TimeAuthority` may read wall time
- no compliance rule inside the Allocator, no decision logic inside a channel or
  an adapter, checked by identifier **and** by string literal
- the Sentinel's four checks run in a declared order that nothing may bypass
- the simulator's latent state is unreachable from what the agent observes, by
  attribute, `__dict__`, dataclass introspection, pickle or graph walk
- no rule may be rendered as carrying more force than its source instrument,
  audited against the rendered HTML

Each scanner is itself tested against a planted violation, so it is never
trusted on an empty tree.

`make test` currently gives **512 passed, 1 skipped**. The skip is an import ban
naming a package that does not exist yet; it skips visibly rather than reading
as green.

## Money and time

Money is integer paise (`arc/core/money.py`). No `float`, no `Decimal`, no
`NUMERIC` column, anywhere. Formatting to rupees happens only at the
presentation boundary.

All timestamps are timezone-aware UTC and all rolling windows are half-open.
Every function that needs "now" receives it as a parameter, which is what makes
the Gate pure and replay honest.
