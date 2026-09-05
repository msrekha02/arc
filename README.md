# ARC — Autonomous Revenue Continuity

**Detect revenue at risk, diagnose why it failed, choose a bounded intervention,
execute it compliantly, and prove what was recovered.**

Recurring-revenue businesses lose money to failed payments they never look at
twice. The usual answer is a dunning schedule: retry on day 1, message on day 3,
call on day 7. That approach is wrong in three expensive ways at once, and this
system is built around not being wrong in those ways.

```
Seed 3 · 1,200 claims · 879 subjects · 4 cycles

  ARC recovered            ₹7,36,526.52     complaints/1k   4.62
  naive dunning            ₹4,57,934.44     complaints/1k   9.57
  doing nothing            ₹5,24,475.34     complaints/1k   0.00

  The industry default recovered LESS than doing nothing.
```

---

## Table of contents

- [The problem](#the-problem)
- [How it works](#how-it-works)
- [System architecture](#system-architecture)
- [Key architectural decisions](#key-architectural-decisions)
- [Results](#results)
- [Installation](#installation)
- [Running it](#running-it)
- [Folder structure](#folder-structure)
- [What CI enforces](#what-ci-enforces)
- [Further reading](#further-reading)

---

## The problem

Four different surfaces leak the same thing — money that should have arrived and
did not:

| Surface | Rail | What failed |
|---|---|---|
| Autopay mandate | UPI Autopay, eNACH | A scheduled debit bounced |
| Card charge | Card | The issuer declined |
| Checkout | Card | The customer left before paying |
| B2B invoice | Invoice | It aged past its due date |

A fixed dunning schedule fails on all three of the hard parts.

**It cannot tell an outage from a delinquency.** When an issuer goes down, four
hundred debits fail within the same hour. Each one, viewed alone, looks like an
ordinary customer failure. A schedule dutifully duns four hundred people for
their bank's incident.

**It contacts people whom contact makes worse.** Some customers pay less the
more you chase them. That segment exists, it is invisible in aggregate metrics,
and every message sent into it destroys value while the recovery dashboard goes
up.

**It cannot prove what it was worth.** "We recovered ₹7 lakh" is a count, not a
measurement. Without a comparator, a stated denominator, and a way to ask what a
*different* policy would have recovered, the number is unfalsifiable.

---

## How it works

One claim, end to end. Every step below is a real module, and the ordering of
the steps is load-bearing rather than incidental.

```mermaid
flowchart TD
    W["Webhook bytes<br/>4 gateway dialects"] --> V{"Signature valid?<br/>checked on RAW bytes"}
    V -->|no| REJ["Refused, never parsed"]
    V -->|yes| ARC["Archive the raw delivery<br/>BEFORE parsing"]
    ARC --> P["Parse"]
    P --> D{"Seen this event_id<br/>in 30 days?"}
    D -->|yes| DROP["Deduplicated"]
    D -->|no| ORD["Sort by EVENT time<br/>fold per account"]
    ORD --> N["REDACTION BOUNDARY"]

    N --> C["Claim<br/>pseudonymous, structured<br/>→ hash-chained ledger"]
    N --> S["SubjectRecord<br/>name, number, narration<br/>→ encrypted, erasable"]

    C --> DIAG["Sentinel: diagnose the cause"]
    DIAG --> FC["Forecaster: bounce, uplift, promise-kept"]
    FC --> AL["Allocator: price budgets, score portfolio"]
    AL --> G{"Gate: certify"}
    G -->|"refused"| NOTHING["do_nothing<br/>probability mass collapses here"]
    G -->|"ALLOW, certificate issued"| TX["ONE transaction:<br/>FSM + reserve + ledger + outbox"]
    TX --> WK{"Worker: certificate<br/>still valid?"}
    WK -->|expired| REQ["Cancel, release budget,<br/>requeue for a fresh decision"]
    WK -->|valid| CH["Channel effector → provider"]
    CH --> M["Outcome measured<br/>off-policy estimate<br/>scoreboard"]
```

### Step 1 — get it in without trusting it

A webhook is attacker-controlled input to a system that moves money. So the
signature is verified **on the raw bytes, before anything is deserialised**, and
the raw delivery is **archived before it is parsed** — because a parser bug is
discovered later, and the deliveries that most need replaying are exactly the
ones a parse-then-archive order would have thrown away.

Events are then sorted by **event time, not arrival time**. A payment capture can
arrive before the failure it supersedes; processing by arrival creates a claim
for money that was already collected, which is worse than missing one, because it
gets diagnosed, funded, and messaged to somebody who already paid.

### Step 2 — split what can be erased from what cannot

This is the single most consequential boundary in the system.

```mermaid
flowchart LR
    E["Incoming event"] --> N["Normaliser"]
    N --> L["Claim<br/><br/>pseudonymous token<br/>closed vocabulary<br/>structured fields only"]
    N --> R["SubjectRecord<br/><br/>name, phone, email<br/>bank narration<br/>raw payload"]
    L --> LED[("Decision Ledger<br/>SHA-256 hash chain<br/>append-only<br/>NEVER erasable")]
    R --> SUB[("Subject Store<br/>AES-GCM per subject<br/>key destroyed on request")]
    LED -.->|"carries only a pointer<br/>and a digest"| SUB
```

An immutable audit chain and a right to erasure are directly contradictory — so
they are **not the same store**. The chain covers pseudonymous tokens and
structured fields only. Names, numbers and bank narrations live encrypted under a
per-subject key. Erasure destroys the key; the row survives, unreadable, and the
chain stays verifiable because it never covered plaintext.

A PII write-guard sits in front of the ledger and **raises** on any name, phone,
email, PAN, Aadhaar, IFSC or card number. No bypass flag, no
redact-and-continue. It also never puts what it found into its own error message,
because a guard that leaks the value it caught has moved the leak into the log.

### Step 3 — ask the expensive question first

This is how the outage problem is solved.

```mermaid
flowchart TD
    START["A debit failed"] --> Q1{"1. COHORT<br/>Is this systemic?<br/>EWMA z-score per issuer cell"}
    Q1 -->|"DEGRADED"| ISSUER["ISSUER layer<br/>→ SUPPRESSED<br/>zero contact"]
    Q1 -->|"no signal"| Q2{"2. MANDATE HEALTH<br/>Is our own setup broken?"}
    Q2 -->|"orphaned, expired,<br/>cap exceeded"| MERCH["MERCHANT layer<br/>→ SELF_HEALING<br/>repaired at the rail"]
    Q2 -->|"healthy"| Q3{"3. CODE MAP<br/>Deterministic lookup"}
    Q3 -->|"known code"| CUST["CUSTOMER layer<br/>→ PLANNED<br/>outreach on the table"]
    Q3 -->|"unmatched"| Q4{"4. LLM RESIDUE<br/>free text only<br/>confidence capped at 0.70"}
    Q4 -->|"no confident answer"| UNK["UNKNOWN<br/>→ conservative path<br/>+ review queue"]

    Q1 -.->|"not enough sample"| POWER["INSUFFICIENT_POWER<br/>never coerced to NORMAL<br/>caps downstream confidence at 0.75"]
```

**The order is the design.** Running the cheap decline-code lookup first turns a
four-hundred-account issuer outage into four hundred delinquent customers. The
expensive systemic check has to come first precisely because it is the one that
prevents mass harm.

The order is enforced structurally: the checks live in a tuple, the diagnose
function iterates it, and no check is ever called by name from the body. A test
walks the AST and fails the build if that stops being true — because a reordering
that looked like a tidy-up would otherwise be invisible in code review.

When the cohort detector does not have enough sample, it says
`INSUFFICIENT_POWER` — a distinct third answer that is **never** collapsed into
`NORMAL`. A detector that quietly answered NORMAL on thin issuers would restore
code-map-first behaviour for most of the traffic with nobody noticing. In the
current run, 586 of 1,200 claims are diagnosed without cohort power, and that is
reported on the dashboard as a measured blind spot rather than hidden.

**The layer matters more than the label.** Issuer-layer causes require zero
customer contact. Merchant-layer causes are our own fault and get repaired
silently at the rail. Only customer-layer causes justify contacting anybody.

### Step 4 — decide over the portfolio, not per account

Three models feed one optimisation:

| Model | Question | Technique |
|---|---|---|
| A | Will tomorrow's debit bounce? | LightGBM + **isotonic calibration** |
| B | Does contacting this person help or hurt? | **X-learner** uplift, signed output |
| C | Will this promise be kept? | Discrete-time hazard, **censored**, IPW-corrected |

Model B is the one that matters. Its output is **signed**: a negative estimate
means contacting this account *reduces* recovery. The sleeping-dog rule then
falls out of the arithmetic rather than being written by hand as a threshold.

The allocation is a multi-dimensional knapsack, relaxed with Lagrange
multipliers so the problem separates and every subject independently takes its
best action by adjusted value. That turns a joint optimisation over the whole
portfolio into one cheap choice per subject.

The multipliers are **shadow prices**, and they are the explainability artifact:
`lambda_retry = 4,631` means the marginal retry was worth that much recovery
given up elsewhere. "This action lost because the budget was worth more
somewhere else" is a real reason derived from the optimisation, not a
rationalisation written afterwards.

Then the crucial part:

```mermaid
flowchart LR
    subgraph BAD["What most systems do"]
        A1["Best action<br/>by score"] --> A2["Take it"]
        A2 --> A3["One action gets probability 1.0<br/>every other gets 0<br/><br/>Every counterfactual weight<br/>is a division by zero.<br/>The result is UNDEFINED."]
    end
    subgraph GOOD["What ARC does"]
        B1["Softmax over<br/>adjusted values"] --> B2["+ 5% epsilon floor<br/>spread uniformly"]
        B2 --> B3["Sample, and LOG<br/>the exact probability"]
        B3 --> B4["The sampling probability<br/>is a recorded fact<br/><br/>Off-policy evaluation works<br/>by construction."]
    end
```

An argmax would be the obvious implementation and would quietly make the whole
system unmeasurable. The epsilon floor costs a few percent of the budget and buys
the ability to answer "what would a different policy have recovered?" — which is
the entire basis of the headline number.

### Step 5 — authorise it, with an expiry

Every action passes a **Gate**: a pure function over 33 versioned rules, with no
I/O, no clock and no model calls anywhere on the evaluation path.

```mermaid
flowchart TD
    R["33 rules, authored as YAML<br/>class · basis · status · on_violation"] --> EV["ONE evaluator"]
    EV --> T1["Touchpoint 1 — project<br/>Allocator prunes candidates<br/>ADVISORY, cannot authorise"]
    EV --> T2["Touchpoint 2 — certify<br/>Decision commits<br/>BINDING, issues certificate"]
    EV --> T3["Touchpoint 3 — dispatch<br/>Worker re-checks the window<br/>expired → cancel + requeue"]
    EV --> T4["Touchpoint 4 — wake<br/>Durable function re-certifies<br/>NO fast path, nothing carried forward"]

    T2 --> CERT["Certificate<br/><br/>valid_from / valid_until<br/>full verdict list, all 33<br/>pinned registry version<br/>derived, not random, id"]
```

Four properties make this work:

- **Every rule is evaluated on every call.** No short-circuit on the first block,
  because an audit trail that records only the refusal cannot show what else was
  considered.
- **Verdicts form a total order.** `BLOCK_PERMANENT > BLOCK > DEFER > ALLOW`, most
  restrictive wins, so 33 independent answers collapse into one decision with no
  adjudication step that could be wrong.
- **Certificates expire, and the window edges are walked back** to the last minute
  the Gate would still say ALLOW. An approval issued at 18:58 for a voice call
  does not stay valid at 19:02.
- **`project` and `certify` share one evaluator**, filtered by decidability class.
  Two rule sets would drift apart silently, and the drift would be invisible until
  it produced an action nobody authorised.

The registry also separates **basis** (statutory / network rule / policy choice)
from **status** (in force / draft / advisory / contested), and one function
renders both. A rule resting on a draft consultation is displayed as *"draft, not
in force; we apply it anyway"* — never as statutory. A test re-reads the rendered
HTML and fails if the word "statutory" appears on a rule that is not binding law.

### Step 6 — make it happen exactly once

```mermaid
flowchart TD
    subgraph ONE["ONE Postgres transaction — all four or none"]
        direction TB
        F1["1 · FSM transition<br/>PLANNED → IN_TREATMENT"]
        F2["2 · Hard budget reservation<br/>conditional UPDATE, cap enforced by CHECK"]
        F3["3 · Ledger append<br/>hash-chained under advisory lock"]
        F4["4 · Outbox insert<br/>ON CONFLICT idempotency_key DO NOTHING"]
        F1 --> F2 --> F3 --> F4
    end
    ONE --> POLL["Workers poll with<br/>FOR UPDATE SKIP LOCKED"]
    POLL --> GATE3{"Certificate<br/>still valid?"}
    GATE3 -->|no| CANCEL["Cancel · release budget ·<br/>log ABANDONED_UNEXECUTED ·<br/>requeue to the Allocator"]
    GATE3 -->|yes| SEND["Provider call<br/>with the idempotency key"]
```

The transaction *is* the guarantee. If those four writes were separate, a crash
between them leaves a claim reading `IN_TREATMENT` forever with no outbox row and
nothing leased — a silent, permanent leak of one customer's recovery that shows
up in no counter.

The idempotency key is `sha256(claim : action : cycle : certificate)`. **The
attempt counter is deliberately not in it**: a dispatch retry must reuse the key
so the provider deduplicates it, while a genuine re-decision must produce a new
one. Putting `attempts` in the key would make every retry look like a fresh
instruction to charge somebody.

What is guaranteed, stated precisely:

| Property | Guaranteed? |
|---|---|
| Exactly-once **state transition** | Yes — by the transaction |
| At-least-once **dispatch** | Yes — by lease and retry |
| Effectively-once **effect** | Yes — by the stable idempotency key |
| Exactly-once **delivery** | **No, and not claimed.** It is impossible. |

### Step 7 — the claim's life, in full

```mermaid
stateDiagram-v2
    [*] --> DETECTED
    DETECTED --> DIAGNOSED

    DIAGNOSED --> SUPPRESSED: issuer outage
    DIAGNOSED --> SELF_HEALING: our own fault
    DIAGNOSED --> PLANNED: customer layer
    DIAGNOSED --> WRITTEN_OFF

    SUPPRESSED --> PLANNED
    SUPPRESSED --> DIAGNOSED
    SELF_HEALING --> RECOVERED
    SELF_HEALING --> PLANNED

    PLANNED --> IN_TREATMENT
    PLANNED --> WRITTEN_OFF

    IN_TREATMENT --> PROMISED
    IN_TREATMENT --> ESCALATED
    IN_TREATMENT --> DISPUTED
    IN_TREATMENT --> RECOVERED
    IN_TREATMENT --> FORBORNE
    IN_TREATMENT --> WRITTEN_OFF

    PROMISED --> RECOVERED
    PROMISED --> IN_TREATMENT
    PROMISED --> FORBORNE
    PROMISED --> WRITTEN_OFF

    ESCALATED --> RECOVERED
    ESCALATED --> FORBORNE
    ESCALATED --> WRITTEN_OFF

    DISPUTED --> IN_TREATMENT
    DISPUTED --> WRITTEN_OFF

    RECOVERED --> REVERSED: chargeback
    REVERSED --> IN_TREATMENT
    REVERSED --> WRITTEN_OFF

    FORBORNE --> [*]
    WRITTEN_OFF --> [*]
```

`FORBORNE` is the hardship path and it is **absorbing** — zero outgoing edges,
including to `WRITTEN_OFF`. No expected-value argument can reopen it, because
there is no edge for an argument to travel along. The property is enforced twice:
in the Python transition table, and again by a Postgres trigger.

### Step 8 — prove what it was worth

Five arms run over the same population. Only ARC's logs can be evaluated
off-policy, because only ARC is stochastic.

```mermaid
flowchart LR
    LOGS["ARC's logged decisions<br/>each with the exact<br/>probability it was sampled with"] --> DR["Doubly-robust estimator"]
    QHAT["Outcome model<br/>shrunk cell means"] --> DR
    DR --> EST["Estimated policy value"]
    TRUTH[("Simulator ground truth<br/>every counterfactual retained")] --> ERR["Estimator's OWN error<br/>reported, not just its output"]
    EST --> ERR
```

The estimator is consistent if **either** the outcome model or the propensity is
correct. Standard practice has to *estimate* the propensity and inherits the
mis-specification as bias. Here the propensity is not estimated — the Allocator
drew with it and wrote it down — so one leg is correct **by construction** rather
than by hope.

Two things the scoreboard structurally refuses to do:

1. **Report recovery without guardrails.** The headline object cannot be
   constructed without a complete guardrail set, and the serialiser re-checks the
   payload it just built.
2. **Merge prevention into recovery.** Money that never failed was never
   recovered. Prevention is a sibling line, and the recovery total is summed from
   the money ledger's RECOVERED leg alone, so there is no arithmetic path from one
   to the other.

---

## System architecture

Nine layers. Each is a Python package, each has one job, and the arrows that do
*not* exist are as deliberate as the ones that do.

```mermaid
flowchart TB
    GW["Payment gateways<br/>UPI · NACH · Card · Billing"]

    subgraph PIPE["The decision pipeline"]
        direction TB
        L0["<b>L0 · Ingest adapters</b><br/>verify → archive → parse → dedupe"]
        L1["<b>L1 · Normaliser</b><br/>THE REDACTION BOUNDARY"]
        L2["<b>L2 · Sentinel</b><br/>cause attribution, 4 ordered checks"]
        L3["<b>L3 · Forecaster</b><br/>bounce · uplift · promise-kept"]
        L4["<b>L4 · Allocator</b><br/>Lagrangian knapsack, stochastic policy"]
        L6["<b>L6 · Conductor</b><br/>one transaction · outbox · breakers"]
        L7["<b>L7 · Channels</b><br/>effect the world, decide nothing"]
        L0 --> L1 --> L2 --> L3 --> L4 --> L6 --> L7
    end

    subgraph GATE["L5 · Gate — PURE"]
        G["33 rules as data<br/>no I/O · no clock · no models"]
    end

    subgraph STORE["Stores"]
        direction TB
        DL[("Decision Ledger<br/>hash-chained<br/>append-only")]
        SS[("Subject Store<br/>encrypted<br/>erasable")]
        ML[("Money Ledger<br/>double-entry<br/>sums to zero")]
    end

    subgraph MEAS["L8 · Proving Ground"]
        PG["arms · composed policy<br/>DR estimator · scoreboard"]
    end

    SIM["Simulator<br/>FROZEN before any policy existed"]

    GW --> L0
    G -.->|"project()"| L4
    G -.->|"certify()"| L6
    G -.->|"re-check"| L7
    L1 --> SS
    L6 --> DL
    L6 --> ML
    L7 --> PG
    SIM -.->|"only the harness<br/>may read truth"| PG
```

Supporting packages: `arc/core` (money, time, ids, types), `arc/ledger` (the
three stores plus the write-guard), `arc/events` (event vocabulary, durable-run
lifecycle), `arc/inngest_fns` (durable functions), `arc/llm_service` (the
confined model boundary), `arc/voice`, `arc/console`, `arc/demo`.

### The arrows that do not exist

CI fails the build on any of these imports:

```
arc/gate/        ⇸  arc/llm_service/    a model answer must never reach a rule
arc/gate/        ⇸  arc/ledger/         the Gate is pure; no I/O on the eval path
arc/allocator/   ⇸  arc/llm_service/
arc/inngest_fns/ ⇸  arc/channels/       a durable step may never touch the world
arc/sentinel/    ⇸  arc/channels/
arc/simulator/   ⇸  arc/allocator/ arc/forecaster/ arc/gate/
                                        the world must not know about the policy
arc/*/           ⇸  arc/simulator/      the policy must not read the answer key
```

The last two are the **anti-circularity guard**, and it runs both ways. Only the
evaluation harness may read simulator ground truth. A forecaster that could see
the answer key would produce a headline number that measures nothing.

---

## Key architectural decisions

The eight choices that shape everything else. In each case the rejected
alternative is the one that would feel natural to write.

### 1. Money is integer paise. Nowhere is there a float.

`Paise = NewType("Paise", int)`. The constructor rejects `float`, `Decimal` and
even `bool` — because `bool` subclasses `int`, so `True` would silently become
one paise. Every monetary column is `BIGINT`; there is no `NUMERIC` and no
`DOUBLE PRECISION` in the schema. Formatting to rupees happens only at the
presentation boundary, and nothing parses it back.

*Why:* float arithmetic on money produces silent, compounding errors, and the
headline number of this system is a sum of money. An error that rounds the wrong
way on 50,000 claims is invisible in code review and fatal under audit.

### 2. Exactly one thing in the repository may read a clock.

`TimeAuthority.now()`. Everything else receives the moment as a parameter — in
particular the Gate, which takes `at: datetime` and never looks it up. A test
walks the AST of every file and fails the build on any other caller, including
aliased and indirect reads.

*Why:* it is what makes the Gate a pure function, makes replay possible, and makes
tests deterministic. A rule that reads a clock is a rule whose verdict cannot be
reproduced six months later during an audit.

### 3. The policy is stochastic, and the propensity is logged.

The optimiser produces a deterministic best action per subject. Taking it would be
the obvious move. Instead it becomes a softmax with a 5% epsilon floor, and the
exact sampling probability is recorded on every decision.

*Why:* off-policy evaluation only has an answer where the logging policy had a
chance of taking the other action. A deterministic policy makes every
counterfactual importance weight a division by zero — not noisy, **undefined**.
The floor costs a few percent of budget and is the entire reason the headline
number exists.

*Consequence:* the uplift model's propensity weights are a recorded fact rather
than a fitted model, so one leg of the doubly-robust estimator is correct by
construction rather than by hope.

### 4. Compliance is data, evaluated by one pure function, at four moments.

33 rules in YAML. One evaluator, filtered by decidability class. No second rule
set for the planning path, no fast path on re-certification, no `force`
parameter, no grace period on an expired certificate.

*Why:* two rule sets drift apart silently and the drift is invisible until it
produces an action nobody authorised. And "it is only four minutes past" is exactly
the reasoning a certificate window exists to refuse — re-deciding costs one cycle,
executing stale authorisation costs the measurement.

### 5. The audit chain and the erasable data are different stores.

The hash chain covers pseudonymous tokens and structured fields. Personal data
lives encrypted under a per-subject key, and erasure destroys the key.

*Why:* an append-only chain and a right to erasure are directly contradictory —
unless the chain never covered anything erasable in the first place. The erasure
sweep also purges the raw webhook archive, which is the copy everybody forgets:
L0 deliberately stores the original bytes before parsing, so shredding the subject
store while leaving the archive intact erases nothing.

### 6. Postgres transactional outbox, not a message broker.

The FSM transition, the budget reservation, the ledger append and the intent to
dispatch commit together or not at all. Workers poll with `FOR UPDATE SKIP
LOCKED`.

*Why:* a broker cannot enrol in a Postgres transaction. Publishing after the
commit reopens exactly the window the outbox closes — the process dies between
the two and the effect is lost, or it publishes then fails to commit and the
effect happens twice. No Redis, no Kafka, no second datastore to keep consistent
with the first.

### 7. Budgets are reserved, never checked.

A reservation is one conditional `UPDATE` against a row holding both the cap and
the amount already taken, with a database `CHECK` enforcing `reserved <= cap`.

*Why:* check-then-act reads, decides, then writes — and between the read and the
write another worker does the same, and both proceed. Under twenty workers that
race fires constantly, and the budget it overruns is a network attempt cap whose
penalty is a real fine.

*Related:* cooldowns are **hard limits and live in the Gate**; contact *volume* is
a **budget and lives in the allocator**. A 48-hour voice cooldown must never be
purchasable at any expected value. Which of this week's contact slots to spend on
whom is exactly what the optimiser should decide.

### 8. The simulator was frozen before any policy code existed.

Every behavioural constant in `arc/simulator/` was written first and has not been
touched since the freeze tag. The simulator cannot import the allocator, the
forecaster or the gate.

*Why:* this is the circularity attack. A world whose constants were adjusted until
the policy looked good measures nothing. The import ban and the commit ordering
are the defence, and both are checkable by someone who does not trust us.

The world hides structure the system has to *discover*: two issuer outages, a
festival week, ~3% of mandates silently orphaning after a card reissue, salary
clustering with per-month jitter, 5% wrong decline codes. None of it appears in
what the agent is allowed to observe. A test tries to reach the hidden state by
attribute, by `__dict__`, by dataclass introspection, by pickle and by walking the
object graph, and asserts every route fails.

---

## Results

Seed 3, the judged seed. 1,200 claims across 879 subjects, ₹16,76,415.92 at risk,
four cycles.

| Arm | Recovered | Incremental | Spend | Compl/1k | Opt-out/1k |
|---|---:|---:|---:|---:|---:|
| null | ₹5,24,475.34 | ₹66,540.90 | ₹0.00 | 0.00 | 0.00 |
| naive_dunning | ₹4,57,934.44 | — | ₹1,406.97 | 9.57 | 16.58 |
| gateway_default | ₹6,88,239.73 | ₹2,30,305.29 | ₹231.60 | 0.00 | 0.00 |
| greedy_unconstrained | ₹6,49,511.48 | ₹1,91,577.04 | ₹17,252.70 | 11.97 | 30.78 |
| **arc** | **₹7,36,526.52** | **₹2,78,592.08** | **₹2,026.73** | **4.62** | **13.86** |

Three readings that matter more than the top line:

**The industry default lost money.** Naive fixed-schedule dunning recovered less
than doing nothing on this population, while generating the second-highest
complaint rate. That is the comparator the headline is stated against.

**The unconstrained arm burns the population down.** It recovers more in cycle 1
and collapses to 15% of that by cycle 4, while ARC holds 42%. Beating a weak
baseline proves nothing; beating this one on net value, at one-eighth the spend
and half the complaints, is the result worth having.

```
recovery per cycle          c1            c2            c3            c4
greedy_unconstrained  ₹3,55,127.98  ₹1,67,641.61    ₹73,147.24    ₹53,594.65
arc                   ₹2,81,479.08  ₹1,92,174.99  ₹1,44,666.47  ₹1,18,205.98
```

**The blind spots are on the dashboard.** 40 claims were suppressed by a detected
issuer outage and contacted zero times. 586 were diagnosed *without* cohort power,
and a circuit breaker watches that share rather than letting it pass as a clean
read. The estimator's error is 2.00% on the develop seed and 8.59% on the judged
seed — both shown, worse one included, because reporting only the better figure
would be choosing the seed after seeing the result.

---

## Installation

**Requirements**

| | |
|---|---|
| Python | 3.11 (pinned; `.python-version` is present) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Database | Postgres 16, via Docker |
| Build tool | GNU make |
| Hardware | CPU only — no GPU anywhere |

**Setup**

```bash
git clone <repo-url> arc
cd arc

# 1. Dependencies, from the lockfile
uv sync --locked

# 2. Postgres 16, waits until healthy
make up

# 3. Schema — migrations/*.sql applied in order, once each
make migrate

# 4. Verify
make test
```

`make test` takes about ten minutes — most of it is model training — and should
report **512 passed, 1 skipped**. The skip is an import ban naming a package that
does not exist yet; it skips visibly rather than reading as green.

The database connection defaults to `postgresql://arc:arc@localhost:5432/arc`
and is overridden with the `DATABASE_URL` environment variable.

**Note:** the demo and the adversarial suite need no database at all. If you only
want to see it run, steps 1 and 4 are optional.

---

## Running it

```bash
make demo                 # the judged run — deterministic, ~25s
make demo-live            # the same, with pauses to narrate into
make demo-adversarial     # 20 attacks through the real code path, ~10s
make demo-digest          # just the reproducibility hash
make console              # build five HTML screens into console/
make validate             # simulator distributions vs published anchors
make lint                 # ruff check + ruff format --check
```

Override the run with `make demo SEED=1 SIZE=600 CYCLES=2`. Seed 1 is the develop
seed; seed 3 is the judged one.

### Reproducibility

`make demo` reads no clock and prints no wall-clock time, so three consecutive
runs produce byte-identical output ending in:

```
digest 5c60e67cf45646afd4e5ff094a1890b98f26e3be1e31a42df0c621a1ae916bef
```

That digest is a SHA-256 over the figures the demo **prints** — not over the
result object, which holds floats that could differ in a last bit without
changing anything a reader sees. It is pinned in `arc/core/reproducibility.py`
and asserted against the real command, so any change to a headline number fails a
test rather than passing quietly.

### The adversarial suite

```
  20 of 20 attacks refused.
```

Every attack runs through the real component, never a mock, and every line names
**which rule refused it** — a refusal nobody can attribute is indistinguishable
from a bug that happened to help.

```
a voice call at 19:01 local            REFUSED    TIME-WINDOW
contact a FORBORNE subject             REFUSED    ABS-FORBORNE
retry a lost-or-stolen card            REFUSED    NET-CAT1
smuggle a name into the ledger         REFUSED    PII write-guard (aadhaar, name_token)
execute an expired certificate         REFUSED    TIME-CERT-WINDOW
reopen a FORBORNE subject              REFUSED    FSM (FORBORNE has no outgoing edge)
render a draft rule as statutory       REFUSED    GI-9 (force always through force_label)
prompt injection in a customer reply   REFUSED    redactor/fence (reply marked as data)
...
```

It closes with the one case that must **succeed** rather than be refused: the
entire pipeline running with `LLM_ENABLED=false`. That is not a fallback path, it
is the default configuration, exercised on every run.

### The console

`make console` writes five self-contained HTML files that open from disk with no
server: an index, the batch, the compliance firewall, the scoreboard, and one
decision replayed end to end in prose. Every figure comes from the same real run
the terminal demo prints — no fixtures, because a console rendered from a fixture
cannot go wrong in the same way the system does.

---

## Folder structure

```
arc/
├── arc/                        the Python package
│   ├── core/                   money · time_authority · ids · types
│   │                           the only monetary type, the only clock
│   ├── ledger/                 decision_ledger · subject_store
│   │                           money_ledger · pii_guard
│   ├── gate/                   L5 — evaluator · registry · lattice · checks
│   │   └── rules/              33 rules, six YAML files, readable by a
│   │                           compliance reviewer without reading Python
│   ├── ingest/                 L0/L1 — pipeline · normaliser · dedupe
│   │   │                       ordering · archive · breaker
│   │   └── adapters/           four gateway dialects, translation only
│   ├── sentinel/               L2 — diagnose · cohort · mandate_health
│   │                           code_map
│   ├── forecaster/             L3 — bounce · uplift · ptp · calibration
│   │                           features · estimates · service
│   ├── allocator/              L4 — candidates · budgets · lagrangian
│   │                           policy · cycle
│   ├── conductor/              L6 — commit · outbox · worker · reservations
│   │                           fsm · breakers · kill_switch · erasure
│   ├── channels/               L7 — contracts · effectors · provider
│   │                           registry
│   ├── proving_ground/         L8 — arms · composed · dr_estimator
│   │                           metrics · policies · harness
│   ├── events/                 leaf package — names · bus · runs
│   ├── inngest_fns/            durable functions — runtime · gated_enqueue
│   │                           salary_retry · ptp_tracker
│   ├── simulator/              FROZEN — world · response_model · seeds
│   │                           wire_fake · codes · validate
│   ├── llm_service/            contracts · redactor · validator · client
│   ├── voice/                  the bounded call state machine
│   ├── console/                build · screens · replay · badges
│   └── demo/                   harness · attacks · run
│
├── migrations/                 plain SQL, applied in filename order
│   ├── 001_core.sql            domain enums, claims, absorbing-state trigger
│   ├── 002_ledger.sql          the three stores
│   ├── 003_ingest.sql          raw archive, dedupe, subject arms
│   ├── 004_conductor.sql       outbox, budgets, reservations, attempts
│   └── 005_control.sql         durable runs, kill switch, breakers, erasure
│
├── tests/                      513 tests across 18 files
├── scripts/migrate.py          no ORM — the outbox needs the index details
├── console/                    generated HTML output
├── architecture.md             the full design document
├── docker-compose.yml          Postgres 16, locale C, UTC
├── Makefile
└── pyproject.toml              uv, Python 3.11
```

Subpackages are created by the milestone that needs them, never in advance —
stubs cause implementations to be written against interfaces that do not exist
yet.

---

## What CI enforces

Convention fails under deadline pressure. A build that fails is evidence; a
comment is a promise. Every rule below is a test that fails the build.

| Guard | How |
|---|---|
| 22 import bans | AST walk, resolving relative *and* dynamic imports |
| One clock | AST scan for `datetime.now`, aliased and indirect |
| Gate purity | No I/O, no clock, no model call on the evaluation path |
| No compliance rule in the Allocator | AST walk of the package |
| No decision logic in channels or adapters | AST walk, by identifier **and** by string literal — `payload["amount_paise"] > 100000` reaches a domain concept just as well as an attribute does |
| Sentinel check order | The checks live in a tuple; nothing may call one by name |
| Simulator observability boundary | Attribute, `__dict__`, dataclass, pickle and object-graph walks all asserted to fail |
| No overstated regulatory force | Re-reads the **rendered HTML** for the word "statutory" |
| No global RNG | Every generator seeded and injected |
| Append-only ledger | Postgres trigger **and** the hash chain |
| Absorbing states | Postgres trigger **and** the Python transition table |
| Every durable function subscribes to every stop | Asserted over the function set |
| No config can disable a voice rule | Reads the config class's own fields |

Each scanner is itself tested against a planted violation, so it is never trusted
on an empty tree. CI runs lint, migrate and the full suite against a real
Postgres 16 service on every push.

---

## Further reading

**[`architecture.md`](architecture.md)** — the full design document. Every layer,
every boundary, the reasoning behind each decision, and a Known Limitations
section that states what this system does *not* do:

- the arm comparison is a paired design, not a between-subjects randomisation,
  and is only possible because the world is synthetic
- prevention is an expectation, not a realised figure
- the cohort detector has a measured blind spot with a breaker on it
- the spend denominator is marginal channel cost only
- no LLM provider is wired; the deterministic path is the supported configuration
- exactly-once delivery is not provided and is not claimed

Where to start in the code:

| If you want | Read |
|---|---|
| The domain vocabulary | `arc/core/types.py` |
| Why erasure and an audit chain coexist | `arc/ledger/pii_guard.py` |
| The compliance model | `arc/gate/evaluator.py`, then `arc/gate/rules/*.yaml` |
| The single most consequential ordering | `arc/sentinel/diagnose.py` |
| Why the policy is stochastic | `arc/allocator/policy.py` |
| What "exactly once" actually means | `arc/conductor/commit.py` |
| Why the headline is defensible | `arc/proving_ground/dr_estimator.py` |
| Whether any of it is true | `tests/`, and `make demo-adversarial` |
