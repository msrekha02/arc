# ARC — architecture

**Autonomous Revenue Continuity.** ARC detects revenue at risk, diagnoses why it
failed, chooses a bounded intervention, executes it compliantly, and proves what
was recovered.

This document describes the system as it exists in this repository. Where a claim
is enforced by a test rather than by convention, the test is named. Where
something is a known limitation, it is stated as one.

Companion documents: `CLAUDE.md` holds the working conventions; `README.md` holds
the quick start.

---

## 1. The shape of the problem

Four leak surfaces produce the same underlying fact — money that should have
arrived and did not:

| Surface | Rail | Claim type |
|---|---|---|
| Failed autopay mandate | UPI Autopay, eNACH | `mandate_failure` |
| Declined card | Card | `card_decline` |
| Abandoned checkout | Card | `checkout_abandon` |
| Overdue B2B invoice | Invoice | `invoice_overdue` |

All four normalise onto one `Claim` type. That is what lets a single contact
budget be shared across them, and it is why the budget can be enforced at all: a
person with three claims on three rails competes with themselves in one softmax
rather than receiving three independent messages.

The hard parts are not detection. They are: not dunning four hundred people for
their bank's outage; not contacting the segment for whom contact destroys value;
not executing an authorisation that has gone stale; and producing a recovery
number that survives an adversarial reading.

---

## 2. The layer stack

Nine layers, each a package. The number in the package docstring is the
authority; this table is a summary.

| Layer | Package | Responsibility |
|---|---|---|
| L0 | `arc/ingest/adapters/` | Trust boundary. Verify signatures on raw bytes, archive, translate. |
| L1 | `arc/ingest/normaliser.py` | Redaction boundary. Split ledgerable from erasable. |
| L2 | `arc/sentinel/` | Cause attribution. Whose fault, and can it be fixed without a human. |
| L3 | `arc/forecaster/` | Three models: bounce, uplift, promise-kept. |
| L4 | `arc/allocator/` | Portfolio optimisation. Which subject gets which action. |
| L5 | `arc/gate/` | Compliance evaluation. Pure, versioned, certificate-issuing. |
| L6 | `arc/conductor/` | Exactly-once state transition, dispatch, breakers, kill switch. |
| L7 | `arc/channels/` | Effect the world. Decide nothing. |
| L8 | `arc/proving_ground/` | Measurement: arms, off-policy estimation, the scoreboard. |

Supporting packages that are not layers: `arc/core/` (money, time, ids, types),
`arc/ledger/` (three stores plus the PII write-guard), `arc/events/` (event
vocabulary and durable-run lifecycle), `arc/inngest_fns/` (durable functions),
`arc/simulator/` (the frozen world), `arc/llm_service/` (the confined model
boundary), `arc/voice/` (the bounded call state machine), `arc/console/` and
`arc/demo/` (rendering and sequencing).

### The flow of one claim

```
webhook bytes
  -> L0  verify signature on RAW bytes
  -> L0  archive the delivery BEFORE parsing
  -> L0  parse, then dedupe on (source, event_id)
  -> L0  sort by EVENT time, fold per account
  -> L1  normalise: Claim (pseudonymous) + SubjectRecord (encrypted)
  -> L1  assign experiment arm per SUBJECT
  -> L2  diagnose: cohort -> mandate -> code map -> LLM residue
  -> L4  candidate pool, priced by L3 uplift, pruned by L5 project()
  -> L4  Lagrangian relaxation, softmax sample, log the propensity
  -> L5  certify() -> Certificate with a validity window
  -> L6  ONE transaction: FSM + reservation + ledger + outbox
  -> L6  worker claims the row, re-checks the certificate, dispatches
  -> L7  effector calls the provider with the idempotency key
  -> L8  outcome measured, arm compared, headline computed
```

---

## 3. Global invariants

Nine invariants are asserted in code rather than documented in prose. Violation
halts rather than continues.

| # | Invariant | Where it is enforced |
|---|---|---|
| GI-1 | No effect without a valid certificate authorising it | `conductor/worker.py` dispatch boundary; `demo/attacks.py` proves it |
| GI-2 | Money is integer paise, everywhere | `core/money.py`, `_reject_float` in `core/types.py`, `BIGINT` columns |
| GI-3 | Budgets are reserved, never checked | `conductor/reservations.py`, `budget_caps.reserved <= cap` CHECK |
| GI-4 | The ledger holds nothing that erasure must destroy | `ledger/pii_guard.py`, `ledger/subject_store.py` |
| GI-5 | Unknown fails closed | `sentinel/code_map.py`, `channels/effectors.py`, adapter admission |
| GI-6 | One rule evaluator; the Allocator holds no compliance rule of its own | `gate/evaluator.py`; `test_allocator_contains_no_compliance_rule_of_its_own` |
| GI-7 | *Not referenced anywhere in the current tree* | — |
| GI-8 | Randomisation is at the subject level | `proving_ground/arms.py`; the `subject_arms` schema has nowhere to put a claim id |
| GI-9 | No rule is represented as carrying more force than its instrument | `Rule.force_label()`, `console/badges.py` |

GI-7 has no reference in any source file, test, migration or rule. Either it was
never assigned or its identifier was lost. This is recorded rather than
back-filled with a guess.

---

## 4. Core domain

### 4.1 Money

```python
Paise = NewType("Paise", int)
```

`arc/core/money.py` is the only monetary type. `paise()` rejects `float`,
`Decimal` and `bool` — `bool` explicitly, because it subclasses `int` and `True`
would otherwise silently become one paise. `format_inr` renders Indian digit
grouping and is a presentation boundary: nothing parses its output back.

The rule holds through the schema. Every monetary column is `BIGINT`; there is no
`NUMERIC` and no `DOUBLE PRECISION` on a money column in any migration.

Floats do appear, legitimately, in three places: shadow prices (a ratio, not an
amount), expected values (an expectation of money, not money) and model
confidences. None of them holds a quantity anyone is owed.

### 4.2 Time

`TimeAuthority.now()` is the only wall-clock read in the repository, and
`test_no_direct_datetime_now` walks the AST of every file to keep it that way,
including aliased and indirect reads.

Everything that needs "now" receives it as a parameter. All timestamps crossing a
module boundary are timezone-aware UTC; `ensure_utc` rejects naive datetimes
rather than assuming they are UTC. All rolling windows are half-open, `[t - d, t)`,
so an event at exactly `t` belongs to the next window.

`FrozenTimeAuthority` lives in `core`, not in `tests`, because replay is a product
feature and needs a real clock substitute on the production path.

`TimezoneBasis` records *which source* decided a subject's timezone — declared,
billing address or telecom circle. The three disagree, and picking silently
produces out-of-hours contact with a clean audit log.

### 4.3 The claim and its state machine

`Claim` carries `amount_paise` (what failed) and `ltv_remaining_paise` (what is at
risk). The objective is weighted by the second: aggressively recovering a small
charge at the cost of a high-value customer is a loss.

Its `subject_token` must be a derived `sub_<32 hex>` pseudonym; the constructor
refuses a raw identifier, and so does a database CHECK constraint.
`evidence_structured` accepts only scalars and flat sequences of scalars, capped
at 128 characters — a shape rule that makes smuggling awkward, not the defence
itself.

Thirteen claim states, with the transition table in `LEGAL_TRANSITIONS`:

```
DETECTED -> DIAGNOSED -> { SUPPRESSED, SELF_HEALING, PLANNED, WRITTEN_OFF }
PLANNED  -> IN_TREATMENT -> { PROMISED, ESCALATED, DISPUTED,
                              RECOVERED, WRITTEN_OFF, FORBORNE }
RECOVERED -> REVERSED -> { IN_TREATMENT, WRITTEN_OFF }
FORBORNE, WRITTEN_OFF   absorbing, zero outgoing edges
```

`FORBORNE` is the hardship path and is absorbing: no expected-value argument
reopens it. The absorbing property is duplicated into a Postgres trigger — the one
property whose violation would be silent and irreversible.

### 4.4 The action space

`ActionType` is **closed at thirteen members** and never extended at runtime. An
open action space cannot be bounded, gated, costed or optimised over. The thirteen
split into silent rail actions (`retry`, `card_updater`, `mandate_re_register`,
`rail_fallback`), digital contact (`whatsapp_utility`, `sms`, `email`,
`payment_link`), assisted (`voice_call`, `instalment_offer`), escalated
(`human_handoff`, `statutory_notice`) and `do_nothing`.

`do_nothing` is a first-class member, not an absence. Without it the knapsack has
no decline-to-act option and every subject receives the least-bad available
action.

---

## 5. The three stores

The system's central structural decision is that an immutable hash chain and a
right to erasure are directly contradictory, so they are **not the same store**.

### 5.1 Decision ledger — immutable, hash-chained, pseudonymous

```
h_n = SHA256(h_{n-1} || canonical_json(entry_n))
```

`canonical_json` produces byte-stable JSON: sorted keys, no whitespace, ASCII
escaped. The bytes hashed are the bytes stored, so verification never has to
reproduce a serialisation decision. `seq` is inside the hashed body, so reordering
breaks the hash.

Appends serialise on a Postgres transaction-scoped advisory lock, so a crashed
appender releases it. `find_breaks` checks three things independently —
recomputed body hash, backward link, sequence contiguity — because an attacker who
repairs one still trips another. A trigger refuses `UPDATE` and `DELETE` outright;
the trigger and the chain both exist because the trigger can be disabled by
whoever owns the table and the chain cannot.

A generated `body JSONB` column gives a queryable projection derived from the
hashed bytes, so it cannot drift from what was signed.

### 5.2 Subject store — mutable, encrypted, erasable

Per-subject envelope encryption with AES-GCM, the subject token as associated
data. Erasure destroys the key; the row survives, unreadable. The `subject_keys`
row is kept after shredding so a later `put()` cannot silently re-key an erased
subject — that path raises `SubjectErased`.

This is the one random source in the repository that is **not** seeded and
injected. A deterministic data key would make ciphertext reproducible from the
seed, which is precisely what crypto-shredding must prevent. The determinism
convention exists so simulation and policy sampling replay identically; it was
never about key material.

### 5.3 Money ledger — double-entry

Every movement writes two legs sharing a `group_id` and summing to zero, so
`SELECT sum(delta_paise)` over the whole table is always zero and checkable in one
query. Money enters through an `EXTERNAL` counter-account, which is what makes the
opening balance balance.

Balances are derived by summing legs. **Nothing stores a running total**, because
a stored total is a second source of truth that drifts. A chargeback posts
`RECOVERED -> REVERSED` and the headline falls out of the same sum with nothing
having to remember to decrement. A number that cannot decrease is not a
measurement.

### 5.4 The PII write-guard

Every string reaching the ledger passes `PIIGuard.scan` first. A hit raises and the
write does not happen. There is no bypass flag, no redact-and-continue, no
warn-and-log — a guard with a bypass is not a guard. There is deliberately no
non-raising inspection method, because a second way in is the shape a bypass
takes.

It detects email, Indian mobile, Aadhaar, PAN, IFSC, card PAN (Luhn-validated),
bank account and name runs. System identifiers are masked first so a subject
token's incidental digit run is not reported as a bank account. Detectors run in
precedence order with span claiming, so a ten-digit mobile reports as a phone and
not as an account.

The name-token heuristic applies only to values containing whitespace — rule ids,
enum values and hashes never contain a space, so they sit outside the heuristic
entirely rather than relying on a vocabulary to spare them.

**The guard never puts what it found into its own exception message.** It reports
kind, path, length and a truncated fingerprint. A guard that leaks the value it
caught has moved the leak into the log.

The guard runs *outside* the transaction on purpose: a refused write must not
leave a rolled-back transaction a caller could mistake for a transient failure and
retry.

---

## 6. L5 — the Gate

The Gate is a **pure function**. No I/O, no clock, no model calls, no database.
Everything a rule could need is a field on `GateContext`; if a rule wants
something not in that object, the answer is to widen the context, never to let the
Gate fetch it. A rule that reads a database is a rule whose verdict cannot be
reproduced six months later during a replay.

### 6.1 Rules as data

33 rules across six YAML files, evaluated by 20 named check functions. Each rule
declares four things that are easy to conflate:

- **class** — when it can be decided
- **basis** — statutory, network rule, policy choice, or heuristic
- **status** — in force, draft, advisory, or contested
- **on_violation** — the remedy, and for `DEFER`, how to compute when

`basis` and `status` are separate deliberately. A rule can be our own policy
informed by an advisory report, or a network rule whose published number is
contested. Collapsing the two is how a system ends up claiming regulatory force it
does not have.

`Rule.force_label()` is the single function every renderer goes through, and it
withholds the basis word whenever the instrument is draft or advisory. A draft
rule renders as `"draft, not in force; we apply it anyway"`.

The registry version is a SHA-256 over the rules' canonical content (`rr-<12 hex>`),
so pinning a version into a certificate pins the exact text that was evaluated. A
replay cannot silently re-decide under today's rules. An empty registry raises: it
would allow everything.

| File | Rules | Kind |
|---|---|---|
| `01_absolute.yaml` | 7 | Third party, employer, opt-out, consent, forborne, disclosure, minor |
| `02_freeze.yaml` | 7 | Promise-to-pay, dispute, hardship, complaint, payment pending, issuer, erasure |
| `03_network.yaml` | 6 | Category-1 declines, do-not-retry advice, attempt budgets, mandate cap, pre-debit notice |
| `04_time.yaml` | 4 | Contact window, quiet hours, blocked days, certificate window |
| `05_cooldown.yaml` | 6 | Per-channel and cross-channel minimum gaps |
| `06_frequency.yaml` | 3 | 24-hour, 7-day, 30-day contact caps |

### 6.2 Decidability classes

| Class | Decidable | Consulted by |
|---|---|---|
| `INVARIANT` | From subject state at plan time | `project` and `certify` |
| `TEMPORAL` | As a prediction at planned execution time | `project` and `certify` |
| `RESERVED` | Because the resource is reserved at plan time | `project` and `certify` |
| `RUNTIME` | Only at execution — certificate expiry, kill-switch state | `certify` only |

There is **one evaluator**, filtered by class. There is no second rule set and no
fast path for `project`, because two rule sets drift apart silently and the drift
is invisible until it produces an action nobody authorised.

### 6.3 The verdict lattice

```
BLOCK_PERMANENT > BLOCK > DEFER > ALLOW
```

Most restrictive wins. The ordering is total, which is what lets 33 independent
verdicts collapse into one decision without an adjudication step that could be
wrong.

Every rule evaluates on every call — **no short-circuit on the first BLOCK**. The
certificate carries the full verdict list, because an audit trail that records only
the refusal cannot show what else was considered.

When the outcome is `DEFER`, the wait is the **latest** of every deferring rule's
next-eligible time. Waiting for the earliest would wake into a rule still
violated. A `DEFER` that carries no timestamp is downgraded to `BLOCK`: a DEFER
nobody can sleep on is a BLOCK wearing the wrong label, and `step.sleepUntil()`
consumes the timestamp directly.

A check that raises produces `BLOCK`, not a pass. Failing closed is the default
everywhere.

### 6.4 Certificates

`certify()` issues a `Certificate` with a validity window, the full verdict list,
the blocking rule ids and the pinned registry version.

The window edges are **walked back to the last moment the Gate itself would still
say ALLOW**, by bounded binary search to minute resolution. A flat ±15 minutes is
not enough: an ALLOW issued at 18:58 for a voice call would otherwise stay valid
until 19:13, and the dispatcher would place a 19:02 call under an authorisation
that was honest when it was written.

The certificate id is **derived, not random** — a UUIDv5 over the registry
version, claim, action, moment, decision and every rule verdict. Identical inputs
give an identical certificate, which makes the Gate observably pure and makes the
dispatch idempotency key stable across retries for free.

### 6.5 Four touchpoints

| # | Where | Call | Binding |
|---|---|---|---|
| 1 | Allocator builds candidates | `project()` | No — advisory mask, prunes only |
| 2 | Decision commits | `certify()` | Yes — issues the certificate |
| 3 | Worker dispatches | Certificate re-check | Yes — expired means cancel and requeue |
| 4 | Durable function wakes | Full `certify()` again | Yes — nothing is carried forward |

Touchpoint 4 has **no fast path**. There is no "if nothing changed, skip the gate"
branch, because "nothing changed" is a claim about the world that only the Gate is
entitled to make.

---

## 7. L0 and L1 — ingest

### 7.1 The order is the design

```
admit -> verify -> ARCHIVE -> parse -> dedupe
      -> order by event time -> fold per account
      -> normalise -> assign arms -> persist -> ledger
```

Three orderings are load-bearing.

**Verify before parse, on the raw bytes.** An unverified webhook is
attacker-controlled input to a money-moving system. Deserialising first and
verifying the reconstructed object hands untrusted input to a parser and compares
a re-serialisation that may not be byte-identical to what was signed.

**Archive before parse.** A parser bug is discovered later, and the only recovery
is replaying the original bytes through a fixed parser. Parse-then-archive loses
exactly the deliveries that most need replaying.

**Order by event time, not arrival.** A capture can arrive before the failure it
supersedes. Processing by arrival creates a claim for money already collected —
worse than a missing claim, because it gets diagnosed, funded and messaged to
somebody who paid.

Dedupe is on `(source, event_id)` over a rolling 30-day window, and the **primary
key is the check**. A check-then-insert races two workers into both believing they
were first, and the cost is a second claim, a second diagnosis and a second message
to the same person.

`raw_events` records *deliveries*, not events, with no unique constraint on
`(source, event_id)`: duplicates are the traffic the table exists to record.

### 7.2 The redaction boundary

`normalise()` splits every arrival in two:

| Half | Contents | Rules |
|---|---|---|
| `Claim` | Pseudonymous, structured, closed vocabulary | Ledgerable, hash-chained, **never erasable** |
| `SubjectRecord` | Name, number, bank narration, raw payload | Encrypted per subject, **destroyed on request** |

A claim carries a pointer and a digest across that line and nothing else. There is
deliberately no method on `SubjectRecord` that returns a ledger-safe projection —
a convenience like that is how a name reaches the chain.

The boundary is here and not only in front of the LLM, because a bank narration
flows downstream attached to the claim. Scrub only the model's input and the
narration still reaches the hash chain, where it can never be removed.

`normalise` performs no I/O. It is a pure function of the event and its context,
which is what lets the whole boundary be tested without a database.

Two failure modes fail closed rather than defaulting: `UnresolvableIdentity` (a
made-up token would be randomised into an arm of its own and contaminate the
design) and `MissingValueEstimate` (the allocator's objective is weighted by
exactly that number).

### 7.3 Adapters

Four dialects — card gateway, NACH, UPI Autopay, billing — each translating and
nothing else. No adapter branches on claim state, cause or amount;
`tests/test_ingest.py` walks their ASTs and fails the build if one starts to.
Gateway-specific shape is a fact about a vendor; policy is a fact about the
system. Once one leaks into the other, a compliance rule becomes untestable
without a webhook fixture.

A per-source circuit breaker (20 failures, five-minute cooldown) keeps one
misbehaving gateway from stalling the others. A tripped source raises on admission
and the refusal is counted — never a silent drop.

---

## 8. L2 — the Sentinel

Four checks, in one order, first confident hit wins:

```
1. COHORT         is this systemic?          -> ISSUER    layer
2. MANDATE HEALTH is our own setup broken?   -> MERCHANT  layer
3. CODE MAP       deterministic lookup       -> CUSTOMER  layer
4. LLM RESIDUE    free text only, capped     -> any
```

**The order is the design.** Running the cheap code map first would attribute a
four-hundred-account issuer outage to four hundred delinquent customers and dun
every one of them. The ordering is worth more than any single check's accuracy,
because the failure it prevents is not a wrong label on one claim but the same
wrong label on a whole cohort.

The order is enforced structurally: `ORDERED_CHECKS` is a tuple, `diagnose`
iterates it, and no check is called by name from the body.
`tests/test_sentinel.py` walks the AST and fails the build if that stops being
true, because a reordering that looked like a tidy-up would otherwise be invisible
in review.

**The layer matters more than the label.** `ISSUER` requires zero customer contact
and routes to `SUPPRESSED`. `MERCHANT` is our own fault, repaired at the rail, and
routes to `SELF_HEALING`. Only `CUSTOMER` justifies outreach. `UNKNOWN` fails
closed onto the conservative path rather than onto silence: a claim nobody could
diagnose is still money owed.

### 8.1 The cohort detector, and INSUFFICIENT_POWER

An EWMA z-score on decline rate per cell against a seasonal baseline built from
the cell's own history. Three sigma, with a floor under the standard deviation and
a minimum of 12 attempts before a cell's rate means anything.

Below that floor the answer is `INSUFFICIENT_POWER`, and it is **never coerced to
NORMAL**. For most issuer-instrument combinations most of the time there is no
power; a detector that quietly answered NORMAL there would restore
code-map-first behaviour for the majority of traffic without anybody noticing.

Three mechanisms handle the thin case: a **back-off ladder** that climbs until a
level has sample and records which level answered; **shrinkage** toward the parent
cell (`w = n/(n+20)`), so eleven transactions cannot fire on one unlucky bucket;
and an **independent downtime signal** from the gateway, which needs none of our
sample and is the primary detector for a thin issuer.

When the cohort has no power, customer-layer attribution is capped at 0.75
confidence, below the 0.80 threshold for money-moving actions. A claim diagnosed
without cohort power spends its first cycle on conservative actions instead of
presenting a debit on the strength of a guess.

### 8.2 Mandate health

Groups on the `mnd_` pseudonym, never the raw UMRN — the history refuses anything
that is not a pseudonym, so a caller who happens to have the real identifier
cannot use it by accident.

It detects cap exceeded, expiry, orphaning after a card reissue (two consecutive
failures with no success between), missing pre-debit notice and a wrong debit
date. Confidences are high where the evidence is arithmetic and lower where it is
inferential.

### 8.3 Code map

Deterministic lookup, run **third**. An unmatched code returns `UNKNOWN` at zero
confidence and goes to a review queue — never guessed at, never falling through to
a permissive default.

A code the table cannot *safely* name is also `UNKNOWN`. There is no cause label
for a risk-driven or technical decline, and forcing one onto `HARD_DECLINE` would
permanently block retries on a transaction merely refused once. Those codes are
listed and mapped to `UNKNOWN` on purpose — an explicit, reviewable "no safe
label" rather than an omission that reads as an oversight.

### 8.4 The LLM confidence cap

An LLM-derived cause is capped at 0.70 confidence, and the cap is enforced in
`Cause.__post_init__` and by a database CHECK — **not** in the Sentinel. A third
copy would be a third place for the number to drift. What the Sentinel does is let
the refusal happen and treat it as a finding that failed validation, which falls
through to `UNKNOWN` and a review queue rather than being coerced into range.

---

## 9. L3 — the Forecaster

Three statistical problems, three techniques. Using one technique for all three is
the common error: they differ in what is observable.

### 9.1 Model A — P(bounce) at T-24h

LightGBM plus isotonic calibration. The only model with no intervention in it, so
it is an ordinary binary classification problem. It powers prevention: a bounce
predicted 24 hours ahead can be intervened on inside the pre-debit notification
window, turning a future failure into a non-event.

**Calibration is not optional.** A GBDT trained on binary logloss emits a score,
not a probability, and that score feeds an expected-value product multiplied by a
rupee amount at L4. An uncalibrated 0.8 that is truly 0.55 does not announce
itself; it quietly misprices every allocation downstream. `BounceModel` cannot
emit a probability without the calibrator — the predict path raises rather than
falling back to the raw score.

Isotonic rather than Platt: Platt assumes the distortion is a sigmoid, and a
GBDT's is not.

The fit is a **three-way split on disjoint accounts** — train, calibrate,
evaluate. Calibrating on the training split fits the isotonic map to scores the
trees have memorised, producing a beautiful reliability curve and a model still
wrong in production.

### 9.2 Model B — uplift, as an X-learner

Estimates `tau(x, a) = E[Y | X, A=a] - E[Y | X, A=do_nothing]` for every action.
The label is never observed per unit.

- **S-learner** systematically under-detects: nothing forces the tree to split on
  the treatment column, and when ability-to-pay and issuer health dominate the
  loss it will not, so the effect collapses toward zero and the sleeping dogs
  vanish.
- **T-learner** has variance set by the smaller arm, and the arms are wildly
  imbalanced by design — voice is rare because it is expensive.
- **X-learner** imputes the effect on each side using the model fitted on the
  other, so the rare arm borrows the abundant arm's outcome surface, then blends
  by propensity.

**The propensity is known, not estimated.** The Allocator logged the exact
sampling probability, so the weight is a recorded fact. `require_propensities`
refuses to run without it and there is **no estimation fallback** — a fallback
would be used, and the moment it is used the guarantee is gone with no visible
symptom.

Signed output is the product, not a side effect. `tau < 0` means contacting this
account reduces recovery, so the sleeping-dog rule falls out of the model rather
than being written by hand.

### 9.3 Model C — P(promise kept)

Two problems at once, and using one technique for both is the common error.

**Censoring.** A promise dated the 20th is neither kept nor broken on the 18th.
Coding unresolved as broken biases the model pessimistic in a specific way: the
promises in flight at any analysis moment are disproportionately recent, and recent
promises are disproportionately about to be kept. The bias is not noise, it is a
systematic pull toward "nobody pays", concentrated on exactly the population being
scored. A discrete-time hazard model expands each promise into one row per day
actually observed; a censored promise contributes zeros for days observed and then
stops, never getting a zero for days nobody watched.

**Selection.** Kept-versus-broken is only observed for promises that were made, by
people who were contacted and engaged. Corrected with inverse-probability weighting
on P(promise made).

### 9.4 Features and staleness

Extraction is defined against `ObservableLike`, a structural protocol the
simulator's `ObservableState` happens to satisfy, so the forecaster trains on a
simulated population and serves on a real one unchanged. This package may not
import `arc.simulator` and may not touch the ground-truth surface by name; both
bans are enforced in CI. A model that reads the answer key measures nothing.

**Per-family staleness, not one global TTL.** Issuer health goes stale in minutes,
account attributes in weeks. Past its TTL a family is not quietly extrapolated: the
estimate falls back to a segment prior, sets `degraded`, and widens its interval.

`Calibrated` enforces three rules in `__post_init__`: a degraded estimate cannot
claim to be a model output; a cold-start estimate (below 50 observations) cannot be
a confident point estimate; an interval always contains its point.

### 9.5 What the models actually score

From the current `make test` run. Every figure carries the acceptance floor the
gate asserts against, so a model that degrades fails the build rather than
quietly getting worse.

| Model | Metric | Value | Floor / ceiling |
|---|---|---|---|
| A | PR-AUC | 0.3994 | 1.35× prevalence baseline of 0.2509 |
| A | Expected calibration error | 0.0136 | ceiling 0.05 |
| A | Brier | 0.1786 | — |
| B | Decile correlation vs ground truth | +0.9746 | floor 0.75 |
| B | Bottom-decile enrichment | 2.267× | floor 1.35× |
| B | Ranking AUC | 0.7260 | floor 0.57 |
| C | Broken rate, naive coding | 0.7789 | — |
| C | Broken rate, over resolved only | 0.5401 | — |

Model B found 747 of 6,000 planted sleeping dogs (12.4%). The two Model C rates are
the censoring bias made visible: coding unresolved promises as broken reports 78%
broken where the resolved population says 54%.

---

## 10. L4 — the Allocator

The unit of decision is the batch. **One action per subject per cycle** — three
claims against one person compete in the same softmax and two lose. That is where
the shared contact budget is actually enforced.

### 10.1 Candidates

Three exclusions, each doing different work.

**Control subjects are removed from the pool entirely**, not merely left untreated.
A control subject still contending for a shared budget would be affecting treated
subjects' allocation, and that is itself a treatment effect.

**Eligibility comes from `gate.project()`, evaluated here and pruned before the
optimisation** — not filtered afterwards, because an action vetoed after sampling
contaminates the logged propensity: the probability written down would be the
probability of an action that could never have run.

**`do_nothing` is always present**, once per subject, at value zero and cost zero.
It sits at subject level rather than per claim because doing nothing is not a fact
about a claim, and three copies would triple its softmax mass for no reason.

There is no compliance rule in this package.
`test_allocator_contains_no_compliance_rule_of_its_own` walks the AST and fails
the build if one appears.

The value function:

```
v_ia = tau * amount_paise * ltv_weight(claim) - direct_cost - annoyance_cost
```

`tau` is signed, so an action that hurts can never outscore `do_nothing` at zero.
The sleeping-dog rule is arithmetic here, not a threshold somewhere else.

### 10.2 Budgets versus hard limits

**Cooldowns are hard limits and live in the Gate; contact volume is a budget and
lives here.** A 48-hour voice cooldown must never be purchasable at any expected
value. Which of this week's three contact slots to spend on which subject is
exactly what the knapsack should decide. Nothing in `budgets.py` can refuse an
action; it can only make one expensive.

Six priced dimensions in a fixed order — contact, voice, rupee, retry, human,
concession — plus `explore`, which is **recorded and never priced**. Pricing the
epsilon spend would make the optimiser trade away the overlap that off-policy
evaluation depends on.

Costs per action range from `email` at ₹0.04 and `sms` at ₹0.18 to `human_handoff`
at ₹9.00 plus eight agent minutes and `statutory_notice` at ₹45.

### 10.3 The optimisation

A multi-dimensional multi-choice knapsack, relaxed:

```
maximise    sum_i sum_a  v_ia * x_ia
subject to  sum_i sum_a  c^k_ia * x_ia <= B_k   for each budget k
            sum_a x_ia <= 1                      for each subject i
```

NP-hard, and the exact solution is not worth having: the LP relaxation's
integrality gap vanishes at portfolio scale.

Relaxing the budget constraints into the objective with multipliers prices them
rather than enforcing them, and the inner maximisation **separates** — every
subject independently takes its best action by adjusted value. Fifty thousand
subjects become fifty thousand independent one-line choices, solved in seconds on
one core. The dual is convex, so each multiplier is found by bisection in
coordinate ascent.

**Lambda is a shadow price and it is the explainability artifact.**
`lambda_voice = 340` means the marginal voice minute is worth ₹340 of foregone
recovery elsewhere. `lambda_k = 0` means budget k is not binding. "Voice lost here
because voice was worth more elsewhere" is a real reason derived from the
optimisation rather than a rationalisation written afterwards.

**Stop-EV falls out rather than being configured.** A subject whose best adjusted
value is not positive takes `do_nothing`. There is no threshold to tune.

**Infeasibility shrinks the treated set and never relaxes a budget.** Constraint
relaxation under pressure is how compliance systems fail, so the caps are frozen
and the treated set moves, with the drops logged.

### 10.4 The stochastic policy

The dual gives every subject a deterministic best action. Taking it would be the
obvious thing to do and would quietly make the system unmeasurable.

Off-policy evaluation only has an answer where the logged policy had a chance of
taking the other action. A deterministic policy assigns probability one to one
action and zero to the rest, so every counterfactual importance weight is a
division by zero and the estimate is **undefined — not noisy, undefined**.

So the argmax becomes a softmax with a 5% epsilon floor spread uniformly across the
eligible set. Three things this buys: `pi(a|s)` is known exactly rather than
estimated; overlap is guaranteed and importance ratios cannot blow up; and
exploration is priced separately as a reported line item rather than folded into
the objective.

Temperature is a **fraction of the subject's own value spread**, not an absolute
number of paise — a portfolio of ₹200 subscriptions and one of ₹2 lakh invoices
differ by three orders of magnitude.

### 10.5 Both times are pinned

`decision_time` is when the portfolio was scored. `planned_execution_time` is when
the action is meant to run, and it is the moment the Gate was asked about. If the
Gate were consulted at wall-clock gating time, a clock tick between allocation and
certification could veto an action whose probability had already been written down.

The Allocator proposes; it does not act. `certify` is invoked because the
certificate is part of the decision record, but the FSM transition, the
reservation, the ledger append and the outbox insert are one transaction owned by
the Conductor.

---

## 11. L6 — the Conductor

> Postgres owns state. Inngest owns time. Neither owns the other's job.

### 11.1 What is guaranteed, precisely

| Property | Guaranteed | By |
|---|---|---|
| exactly-once **state transition** | yes | the Postgres transaction |
| at-least-once **dispatch** | yes | lease and retry |
| effectively-once **effect** | yes | a stable idempotency key the provider honours |
| exactly-once **delivery** | **no, and not claimed** | impossible |

Claiming the last one is the tell that somebody has not thought about it.

### 11.2 The one transaction

`commit_decision` performs four writes or none:

```
FSM transition
hard budget reservation
ledger append
outbox insert  ON CONFLICT (idempotency_key) DO NOTHING
```

The ordering inside is deliberate. The reservation goes before the ledger append
because a refused budget should not leave a decision recorded as taken. The outbox
insert goes last because it is the only write idempotent on its own.

If these were separate: the FSM moves, the process dies, the claim reads
`IN_TREATMENT` forever, no outbox row exists, nothing is leased so no reaper looks
at it. A silent, permanent leak of one customer's recovery that shows up in no
counter.

### 11.3 The outbox, not a broker

A broker cannot enrol in a Postgres transaction. Publishing after the commit
reopens exactly the window the table closes: the process dies between the two and
the effect is lost, or it publishes and fails to commit and the effect happens
twice.

`FOR UPDATE SKIP LOCKED` lets N workers poll one table, each skipping rows another
holds rather than queueing behind them. No advisory locks, no worker-id
partitioning, no Redis, no second datastore.

The lease is the safety net, not the mechanism: `SKIP LOCKED` prevents concurrent
claiming, the lease recovers a row whose worker died holding it. Two minutes,
comfortably longer than the slowest channel call.

### 11.4 The idempotency key

```
sha256(claim_id : action_type : cycle_id : certificate_id)
```

**The attempt counter is not in it, and that is the whole design.** A dispatch
retry of the same decision must reuse the key so the provider deduplicates it; a
genuine re-decision after a wake must produce a new one so it is allowed through.
Putting `attempts` in the key would make every retry look like a new instruction
to charge somebody.

### 11.5 Two-tier reservations

Budgets are **reserved, never checked**. Checking reads, decides, then writes;
between read and write another worker does the same and both proceed. Under twenty
workers that race fires constantly, and the budget it overruns is a network attempt
cap whose penalty is a real fine. A reservation is one conditional `UPDATE` against
a row holding both the cap and the amount taken, with a `reserved <= cap` CHECK.
There is no window to lose.

| Horizon | Tier | Effect |
|---|---|---|
| ≤ 15 minutes | HARD | Budget locked at plan time |
| > 15 minutes | SOFT | Recorded as pipeline demand; converts at wake |

A retry scheduled for payday three days out would otherwise hold a network attempt
for three days. It is also why a FREEZE can safely release reservations —
long-horizon work never held one.

Release happens on both paths: on a terminal outcome, and on expiry. A leaked
reservation is a starvation bug that stays invisible until the portfolio quietly
stops treating anybody.

**The gateway retries on its own schedule**, and those attempts count against the
network cap whether or not we issued them. Both initiators are recorded in
`network_attempts` and the cap is read from the sum. A counter that only knows
about our own retries is wrong in the unsafe direction: the cap can be exceeded
without ARC ever having sent the excess, and the penalty lands on us regardless.

### 11.6 Gate touchpoint 3

The first thing the worker does, before any provider is reached. If the certificate
window has closed, the row is cancelled, an `ABANDONED_UNEXECUTED` entry is
appended, the budget is released, and the claim goes back to the Allocator.

There is no grace parameter and no `force` argument. "It is only four minutes past"
is exactly the reasoning the window exists to refuse. Re-deciding costs one cycle;
executing stale authorisation costs the measurement.

The worker is deliberately dumb about policy. It does not choose actions, does not
escalate, does not decide what to do after a failure beyond retry-or-die.
Escalation authority belongs to the Allocator.

### 11.7 Ten circuit breakers

Seven measure harm to people; three watch the system's own machinery.

| Breaker | Metric | Threshold |
|---|---|---|
| CB-COMPLAINT | complaints per 1,000 contacts | 1.5× trailing median |
| CB-OPTOUT | opt-outs per 1,000 contacts | 1.5× trailing median |
| CB-CANCEL | treated over control cancel rate | 1.5× trailing median |
| CB-VOLUME | outbound volume | 3.0× trailing median |
| CB-CHANNEL-FAIL | dispatch failure rate | 10% |
| CB-SENTIMENT | negative sentiment on calls | 1.5× trailing median |
| CB-RESUME-RAMP | admission over the ramp cap | 1.0 |
| **CB-VETO** | post-allocation veto rate | 2% |
| **CB-DEGRADED** | decisions on stale features | 20% |
| **CB-COHORT-BLIND** | diagnosed without cohort power | 40% |

The three self-monitoring ones are what separate an engineered system from a demo.
CB-VETO above 2% does not mean the Gate is strict — `project` and `certify` share
one registry, so only RUNTIME rules can fire after allocation, and a higher rate
means the eligibility projection is broken. CB-COHORT-BLIND is the honest one: the
Sentinel's back-off cannot always find power, and this is where that miss
**surfaces** rather than passing as a clean NORMAL. An unmeasured blind spot is a
defect; a measured one is a known limitation.

**Every breaker trips to SHADOW, not to OFF.** Shadow keeps L0–L5 running and the
ledger filling, so the diagnosis of whatever tripped it is being recorded while it
is tripped. A system that goes dark when it detects a problem destroys the evidence
about the problem.

Thresholds are ratios against a trailing median or absolute shares with a stated
denominator. A complaint rate is meaningless without the volume it came from.

### 11.8 Four modes

| Mode | L0–L4 | L5 | L6 | L7 |
|---|---|---|---|---|
| NORMAL | run | certify | reserve + dispatch | execute |
| SHADOW | run | certify | record intent only | idle |
| DRAIN | admission stopped | — | finish in-flight | in-flight |
| FREEZE | run | certify | RELEASE + mark HELD | idle |

SHADOW takes no reservation and writes no outbox row. A shadow mode that queued
work would accumulate a backlog for its whole duration, and the moment it switched
off that backlog would go out at once — to people whose circumstances had moved on
by however long the shadow lasted.

**Held work is invalidated on resume and never executed.** The world changed during
the freeze — that is what freezing is for — and the certificates expired anyway.
The tempting bug is a resume path that dispatches: it looks efficient, and it is
the thundering herd and the stale-authorisation bug at once. `resume` has no branch
that can execute; held items are requeued through the same path everything else
uses, so there is no special-case resume code to get wrong.

The admission ramp is 5 / 25 / 60 / 100 percent of the trailing median over four
cycles. Coming back at full volume sends a burst that is itself a volume surge,
which trips CB-VOLUME, which freezes the system again.

The mode lives in a one-row table (`only_row BOOLEAN PRIMARY KEY CHECK (only_row)`),
because a table that permits two modes will eventually hold two.

### 11.9 Erasure

Erasure has to reach every store:

| Store | Action | Why it is easy to forget |
|---|---|---|
| Subject store | Crypto-shred the key | The obvious one |
| **Raw archive** | Purge rows by subject token | **The one that gets forgotten.** A complete copy of everything the subject store holds, in the gateway's own format |
| Outbox | Cancel scheduled work | A message going out after erasure is the erasure failing visibly |
| Durable runs | Cancel via the existing event | One cancellation path, not two |
| Decision ledger | **Not erased** | The point of the whole design |

The ledger survives because it never contained personal data: the chain covers
pseudonymous tokens and structured fields, the write-guard fails any append that
would change that, and raw text lives on the other side of the redaction boundary.

**Order matters.** Stop the future first — cancel scheduled work and running
functions — so nothing in flight reads a subject mid-shred. Then shred, sweep the
archive, and record.

`requested_by` must be a role or a pseudonymous operator reference. An erasure
recording the requester's own personal data would create a new erasure obligation
in the one store that cannot honour one. This is not merely documented: the
write-guard refuses the append, so an email there fails the whole transaction.

---

## 12. L7 — channels, and the voice machine

### 12.1 Channels decide nothing

A channel receives a payload and an idempotency key, hands them to a provider, and
reports one of seven structured outcomes. It does not read claim state, cause,
amount, arm or confidence. `tests/test_channels.py` walks the AST and fails the
build if a branch condition mentions a domain concept **by identifier or by string
literal**, because `payload["amount_paise"] > 100000` reaches the same place
through a subscript.

The effector is the last code before the world. Policy living there executes
without having passed the Gate, cannot be replayed from the ledger, and cannot be
tested without a provider fixture.

Every channel is the same class. The provider's vocabulary is **mapped, not
branched on** — `STATUS_TO_OUTCOME` is a dictionary, so there is no conditional for
the AST scan to object to and no place for a rule to grow. An unrecognised status
raises rather than falling back to `failed`: folding a new vendor state into the
residual bucket is how it disappears from the guardrail metrics for a year.

The seven outcomes are closed. `WRONG_NUMBER` is not `BOUNCED` — a bounce means
nobody received it, a wrong number means a stranger did, which is a third-party
disclosure risk and triggers suppression rather than a retry. `OPTED_OUT` is not
`FAILED` — an opt-out is a permanent consent change the Gate must honour forever
after. Failure modes are training data, not noise.

The certificate is **not** re-checked in the effector. GI-1 is asserted at the
Conductor's dispatch boundary, the last code that runs before this one. Re-checking
here would put compliance semantics into the one layer this design keeps free of
them.

### 12.2 The fake provider

Failure selection is **hashed from the idempotency key, not sampled**. Under twenty
concurrent workers a seeded draw per call is not reproducible: which key gets the
failure depends on the order workers happen to reach the provider. Deriving it from
the key makes the outcome a property of the message rather than of the schedule.

Every invocation is recorded, not only every effect. If two workers dispatch one
row the key appears twice in `invocations` even though `effects` shows one —
counting only effects would hide the Conductor's failure behind the provider's
competence.

### 12.3 Voice

Voice is a separate package, not an effector, and the reason is structural. A voice
conversation branches on claim state, verification and confidence **by
construction** — the rules fire mid-call, on a turn the Gate never saw because the
turn had not happened when the certificate was issued. Putting it in `channels/`
would make the AST scanner right to complain, and the only ways to quiet it would
be to exempt the file or weaken the forbidden-name list for everybody.

Six non-removable rules:

| Rule | Requirement |
|---|---|
| VOX-DISCLOSE | AI identity disclosed in the **first** utterance |
| VOX-VERIFY | Identity verified before any account detail is spoken |
| VOX-RECORD | Recording disclosed with prior intimation |
| VOX-CLI | Transactional number series, never promotional |
| VOX-DISTRESS | Distress → immediate human handoff, automation ends |
| VOX-WRONG-PARTY | Wrong party → disclose nothing, suppress, no redial |

"Non-removable" means **no flag exists**, not that a flag defaults to on. There is
nowhere in `VoiceConfig` to put them, and `assert_no_configuration_can_disable`
reads the class to prove it. A configuration option to disable AI disclosure is one
a campaign manager will eventually find, under pressure, on a bad quarter — and the
fact that it defaulted to on will not be in the transcript.

The stops fire on recognition, not on a confidence threshold. Confidence gates
whether a promise is *recorded*, because a wrongly recorded promise freezes a claim
and misleads a model. It does not gate whether the machine stops: the cost of
stopping a call that was fine is a call, and the cost of continuing one that was
not is a person in distress being argued with by software.

The LLM is confined to two jobs inside the machine — choosing which allowed
utterance to speak, and classifying what it heard into a closed intent set. A model
returning nonsense produces a call that says something bland and hangs up, rather
than one that promises a discount nobody authorised.

---

## 13. Durable functions and events

### 13.1 The split

`arc/events/` is a **leaf package** — it imports `arc.core` and nothing else in the
system, so anything may depend on it and it depends on nothing that could depend
back. That is what removed a `conductor -> inngest_fns -> conductor` cycle which
held only because `runtime.py` happened to import nothing from the Conductor.

`arc/events/` owns the vocabulary, the log and run lifecycle (state).
`arc/inngest_fns/runtime.py` owns step semantics (execution).

### 13.2 Memoised steps

The primary key on `(run_id, step_id)` is the whole guarantee. A durable function
is **replayed from the top** on every resumption, not resumed mid-line. Without
memoisation, waking from a sleep would re-run `gated_enqueue`, issue a second
certificate for the same wake, and put a second row in the outbox under a different
key.

Cancellation is evaluated at step boundaries, which is where it matters. A sleeping
run is by definition between steps, so a `cancelOn` event landing during a sleep is
seen the instant the sleep is asked to end. Nothing polls.

### 13.3 The six behavioural stops

`CANCEL_ON` is one constant carried by every durable function: `claim.recovered`,
`claim.disputed`, `subject.hardship`, `subject.erasure`, `consent.withdrawn`,
`system.freeze`. `assert_cancels_on_every_stop` makes adding a function without the
full set fail rather than ship.

Match keys differ by event and that is the point. `claim.recovered` matches a
claim; `subject.hardship` matches a **subject**, because hardship is a property of
a person and must stop every claim they hold.

The alternative is the bug. A function that checks for hardship when it wakes has
already decided to wake, and between the signal and the wake it is a scheduled
contact to somebody who has told you they are in distress.

### 13.4 gated_enqueue — the only path to an effect

Files under `inngest_fns/` may not import from `channels/`; CI enforces it.

1. Re-fetch claim state **fresh from the database**, not from the event that
   started the run three days ago.
2. Gate touchpoint 4 — full re-certification against the state and moment of the
   **wake**.
3. On ALLOW, harden the soft reservation and insert the outbox row in one
   transaction.

**BLOCK is not retryable. DEFER is, three times.** A DEFER carries a computable
next-eligible timestamp. A BLOCK does not and cannot — a freeze with no known end
resolves to BLOCK precisely *because* there is no timestamp to sleep until.
Treating it as retryable would mean sleeping on a duration invented by the
scheduler, which is a policy decision the Gate declined to make.

The defer loop is bounded at three because each hop reuses a decision made for a
different moment. One or two hops is a cooldown expiring. A fourth would mean the
world has moved far enough that the decision itself is stale.

### 13.5 The promise tracker does not escalate

A broken promise is not an instruction to escalate; it is a **feature**, and the
Allocator re-scores the claim with it alongside everything else it knows.

The off-policy consequence is the sharp one: an action chosen inside a tracker has
no logged propensity, because no distribution was sampled. It would appear in the
logs as an action that happened with probability nothing, and every importance
ratio touching it would be a division by zero. **One escalation decided in the
wrong place makes the batch it belongs to unmeasurable.**

Censoring is not breakage. A promise whose date has not arrived is `UNRESOLVED`, a
third answer, and Model C is fitted on exactly these records.

---

## 14. The simulator

Frozen at `simulator-frozen-v1`. Every behavioural constant was written before any
policy code existed and none is touched after the tag. A world tuned after seeing
policy results measures the policy against itself.

### 14.1 Two boundaries

**The observability boundary.** `World.observe()` returns an `ObservableState` and
nothing else. `LatentState` is not reachable from it — not by attribute, not
through `__dict__` (`slots=True` removes it), not by dataclass introspection, not
through pickled bytes, not by walking the object graph. `tests/test_simulator.py`
tries all of those routes and asserts each fails.

**The anti-circularity guard.** The simulator does not import `arc.allocator`,
`arc.forecaster` or `arc.gate` — the world does not know about the policy measured
against it. The ban runs both ways: policy packages may not import `arc.simulator`,
and only `arc/proving_ground/` may read ground truth.

### 14.2 Structure the agent must discover

None of this appears in `ObservableState`:

- Two issuer outages: two hours on a large private issuer across the morning
  presentation peak, and forty minutes on a smaller PSU issuer — deliberately thin,
  so the Sentinel has to climb its back-off ladder rather than answer from one
  bucket.
- A festival week suppressing payment activity (ability multiplier 0.72).
- ~3% of mandates silently orphaning after a card reissue. The merchant's own
  records still say `active`, and `mandate_status` keeps saying so.
- Salary clustering on the 1st and the last working day, jittered per month.
- 5% wrong or remapped decline codes, 3% stale phone numbers.

An outage is visible only as a burst of correlated declines, which is exactly what
the Sentinel's cohort detector has to discover.

### 14.3 The response model

```
P(pay) = sigmoid( b0
                + b1 * ability_to_pay(t)
                + b2 * responsiveness[channel]
                + b3 * timing_fit(t, salary_day)
                + b4 * issuer_health(issuer, t)
                - b5 * annoyance_sensitivity * contacts_7d
                - b6 * friction(action)
                + b7 * amount_affordability(amount, income) )
```

Two terms carry the result. `b5 > 0` is the **sleeping-dog term**: because
`contacts_7d` counts the contact being considered, an account with high annoyance
sensitivity and low responsiveness has a genuinely negative treatment effect. A
policy that contacts everyone loses to one that does not.

`b3` is the **payday term**. `salary_day` is latent and jittered, so a
fixed-calendar policy cannot align with it; it is inferable from
`prior_payment_timestamps`, which are observable. That gap between the naive arm
and ARC is structural, not tuned.

Every constant names the published figure it is anchored to. Issuer identifiers are
generic — attaching a real bank's name to a synthetic outage would be an assertion
about that bank the simulator has no evidence for.

### 14.4 Seeds and streams

Three seeds, announced in advance: `DEVELOP = 1` to develop against, `TUNE = 2` to
tune against, `JUDGED = 3` to run once, live. A system tuned on the seed it is
evaluated on has been fitted to the evaluation set.

Six independent streams — population, failures, outcome, promise, wire, validate —
keyed by a hash of the stream *name* rather than its enum position, so inserting a
member does not renumber the others and silently change every generated batch.

`stable_hash` exists because Python's `hash()` is salted per process, and anything
reproducing across runs must not depend on it.

### 14.5 The wire fake

A separate component from the world because they fail differently: the world is
wrong when its behavioural constants are wrong, the fake when its payload shape
drifts from the gateway's. Merging them would couple the L0 adapter to simulator
internals.

It emits signed, gateway-shaped webhooks with 2% duplicate delivery, 3% late
delivery, three dialects for the same underlying fact, and free text containing
real-looking names.

`python -m arc.simulator.validate` prints generated distributions against the
published figures they were calibrated to, with deltas, and exits non-zero on
drift. It keeps two tables apart: external anchors, and structure this build
deliberately planted — quoting our own constant back as though it were external
evidence would be worthless.

---

## 15. L8 — the Proving Ground

### 15.1 Arms, assigned per subject

Five arms: `null`, `naive_dunning` (the comparator), `gateway_default` (the
incumbent), `greedy_unconstrained`, `arc`.

The unit of randomisation is the unit of interference. A person with one claim in
control and another in treatment breaks SUTVA twice: portfolio allocation can
starve the control claim of budget the treated claim consumed, and a WhatsApp about
invoice A reminds them about invoice B. Neither effect is small and neither is
detectable after the fact — the estimate is simply wrong, with no symptom.

The `subject_arms` table is keyed by `subject_token` and **has nowhere to put a
claim id** — the schema is the cheapest place to make claim-level randomisation
impossible rather than merely discouraged. `ArmRegistry` makes the first answer the
only answer, because a subject's claim count grows and a stratum recomputed later
would move them between arms.

Stratified on claim-count bucket, value decile and rail.

### 15.2 The composed policy

What reached the world is not always what the Allocator drew, because the Gate can
refuse. The behaviour policy is the composition:

```
pi_exec(a|s) = sum_a' pi_alloc(a'|s) * 1[ gate.certify(a', s) resolves to a ]
```

The Gate is deterministic given state and time, so the indicator is a function
rather than an expectation and the composition is exact — no sampling, no
estimation.

**Vetoed branches collapse onto `do_nothing`; they are never dropped.** Dropping
and renormalising looks harmless and is selection bias of the worst kind: the Gate
refuses precisely the subjects whose state makes them refusable — inside a
cooldown, mid-freeze, out of hours — and those subjects differ systematically in
their recovery. There is no symptom. `veto_mass` exists so the quantity a dropping
estimator would have discarded is a number on the record rather than an absence.

`do_nothing` resolves to itself without a Gate call, so the collapse target is a
fixed point. If it could be refused, refused mass would have nowhere to go and the
distribution would not sum to one. Multiple branches resolving to one outcome
**add**; assigning rather than accumulating would silently keep only the last.

### 15.3 The doubly-robust estimator

```
V_DR(pi) = (1/n) sum_i [ sum_a pi(a|s_i) qhat(s_i, a)
                       + (pi(a_i|s_i) / pi_b(a_i|s_i)) * (r_i - qhat(s_i, a_i)) ]
```

The estimate stays consistent if **either** the outcome model or the propensity is
correct. Standard practice must estimate the propensity and inherits the
mis-specification as bias, leaving the position "hopefully one of two fitted models
is right."

**Our position is structural.** `pi_b` is not estimated: the Allocator drew with it
and logged it, the epsilon floor bounds it away from zero, and
`composed_propensity` folds the Gate in in closed form. One leg is correct **by
construction**.

That is why the outcome model is shrunk cell means and not something more
expressive. Fitted on the same logs it corrects, an expressive model would
interpolate the noise it is meant to average out, and its residuals would shrink
toward zero exactly where the importance weights are largest. A better `qhat`
reduces variance; it does not buy consistency, which is already paid for.

Importance ratios are clipped and **the clipped share is reported**, because
clipping trades bias for variance and a reader is entitled to know how much was
traded.

**The bootstrap resamples subjects, not rows.** The subject is the unit of
independence; resampling rows would treat one subject's four cycles as four
independent observations and produce an interval roughly half as wide as the truth
deserves.

Because the simulator retains full counterfactuals, the estimator's **own error**
is reported rather than only its output. That is the difference between a
measurement and a claim.

### 15.4 The harness and the paired design

In production the arms would be a randomised split, and `arms.py` implements
exactly that. A simulator has more than one world: `World.fork()` returns a fresh
interaction history over the same population, so every arm runs over every subject
without arm B's contacts raising arm E's annoyance.

**The honest statement is stated rather than discovered:** the randomisation
machinery is real and is what would run against a real population; the reported
numbers come from the stronger paired design the simulator permits. Both are
carried — the stratified assignment is computed on every row.

**Only ARC's logs are evaluated off-policy.** The other four arms are
deterministic, so they have no propensity to condition on and off-policy evaluation
of them is undefined. They are run directly against the world and measured
on-policy.

Ground truth is read **before the world moves**, immediately before each action is
applied, at exactly the state the decision was made in. Reading it afterwards would
compare the estimate against a different question.

**What is held constant.** Every arm sees the same batch, the same claims and the
**same** uplift estimates; only the decision rule differs. `SharedUplift` conditions
only on `ObservableState` — the facts a merchant has in its own database. It knows
nothing about annoyance, contact history, payday timing or churn intent, all four
of which the world implements and never discloses. Any advantage ARC shows comes
from the decision machinery, not from being told more.

The Gate applies to four arms, not five. Greedy is unconstrained — that is what the
word means — and it can only show what the constraints were buying by not having
them. It is a measurement instrument and never touches a real channel.

**Greedy is the arm that matters.** It will recover more *gross* rupees: it
contacts everyone, escalates immediately, and pays whatever a contact costs.
Beating a weak baseline proves nothing. Beating this one on *net* value while it
blows the guardrails and spends several times the money is the result worth
presenting.

### 15.5 The scoreboard's two refusals

**A recovery number cannot be constructed without its guardrails.** `Headline`
cannot be built without a complete `Guardrails`, and `to_dict` re-checks the
payload it just built and raises if a recovery figure appears without all of them.
A later edit that adds a recovery key and forgets the guardrails fails at runtime
rather than shipping a prettier number.

**Prevention is never merged into recovery.** Money that never failed was never
recovered. `prevented_paise` is a sibling of the headline, not a component, and
`recovered_paise` is summed from the money ledger's RECOVERED leg alone, so there
is no arithmetic path from one to the other.

`denominator` is a required free-text statement of what the rate is over. An
unstated denominator is the first thing an experienced reader attacks.

The headline is **incremental against arm B**, with a bootstrap interval. A total
is a count; only the difference against a comparator is a measurement.

Guardrails are stored as **counts**, with rates derived in the same code, so the
denominator is visible rather than hidden. The promise made-rate and kept-rate are
reported side by side on purpose: a widening gap means promises are being extracted
that the customer cannot afford, which reads as success in a kept-count and as harm
in the relationship.

---

## 16. The LLM boundary

The model is confined by **structure, not by prompt text**. A prompt asking a model
not to compute an amount is a request; a type system in which no LLM output can
carry an amount is a constraint.

Four sanctioned tasks: free-text cause classification (enum plus confidence, capped
at 0.70), message generation (validated against a template), in-call conversation
(utterance plus intent from a closed set), and PTP extraction.

It may never compute or alter an amount, choose an action, schedule a time,
evaluate a compliance rule, write to the ledger, read latent state, see another
subject's data, determine a cause layer without a deterministic corroborator,
decide an escalation tier, or produce free-form output reaching a customer
unvalidated.

Amounts, actions and times are enforced by the types having nowhere to put them.
The ledger, latent state and Gate are enforced by import bans: `arc/gate/` and
`arc/allocator/` may not import `arc.llm_service` at all, so an LLM answer cannot
reach a rule evaluation however it is dressed up. A third ban names `arc/money/`,
which does not exist as a package, so it currently skips rather than passing.

**Validation order:** schema → groundedness → safety → fallback → log.

Groundedness is **string equality against the source record**, deliberately nothing
cleverer. A validator parsing the amount and comparing numerically would accept
"twelve hundred and ninety nine rupees" for a record saying "Rs 1,299.00", which is
fine right up until it accepts one saying twelve thousand.

**Reject, never coerce.** A model returning an enum value one character off is not
nearly right; it is a model whose output nobody has checked. Coercing means the
first genuinely wrong answer arrives looking like all the ones quietly corrected.

`LLM_ENABLED` defaults to false, and that is **the supported configuration**, not a
stub. The canned template is what every message is with the model off, so the
fallback is exercised on every run rather than only when something breaks. The
adversarial demo runs the whole pipeline that way.

The redactor does two jobs, and the second is the forgotten one: scrub PII before it
reaches a model, and **delimit untrusted content and label it as data**. A customer
replying to a WhatsApp message is an attacker-controlled input channel to a system
that moves money; prompt injection there is not hypothetical, it is the obvious
thing to try.

---

## 17. Console and demo

Four screens, built from a real run with **no fixtures** — the same objects the
acceptance gates assert on. A console rendered from a fixture is a screenshot, and
a screenshot cannot go wrong in the same way the system does.

| Screen | Shows |
|---|---|
| Batch | Counters, the diagnosis split, claims a detected outage took off the contact path |
| Compliance firewall | Proposed to executed with every category counted, per-rule fired counts, the honest mix |
| Scoreboard | Five arms, guardrails beside the money, the estimator's own error |
| Replay | One decision end to end, including the propensity it was drawn with |

Each screen is a **dataclass that refuses to be built wrong**, with rendering a
total function from it. Put the invariants in a template and they become
conventions that hold until somebody adds a column.

`assert_no_overstated_force` re-reads **rendered output** and fails if the word
"statutory" appears in the badge of a rule that is not binding law. Checking the
output rather than the inputs is deliberate: the failure mode is a call site that
bypassed the badge module entirely. Badge *tone* is keyed off the label, not the
basis, because colour is a claim about force too.

Server-rendered HTML with no build step. The repo has no JavaScript toolchain, and
a console that could not be exercised by the test suite would be the one screen
whose claims are unverified — precisely backwards for the screen whose job is to
show that the other claims are true.

The demo is nine beats: the batch lands, diagnosis splits it, the allocator runs,
the compliance firewall live, one voice call, the hardship stop, the scoreboard,
the estimate checked against truth, and one claim replayed. The pauses are part of
the design: the outage suppression and the hardship stop both land only if somebody
has time to say what just happened.

`make demo SEED=3` reads no clock and prints no wall time, so three consecutive
runs are byte-identical. The digest hashes **the figures the demo prints**, not the
result object — the object holds floats that could differ in a last bit without
changing anything a judge sees, and it omits framing text a careless edit could
change while the numbers held. Hashing what is printed is hashing what was claimed.
It is pinned as a constant in `arc/core/reproducibility.py` and asserted against the
real command, because recomputing it would mean running the judged seed to find out
what the judged seed produces, which is not a check.

The adversarial suite runs 20 attacks, each **through the real component**, not a
mock. Every line carries what was attempted, that it was refused, and **which rule
refused it** — a refusal nobody can attribute is indistinguishable from a bug that
happened to help. Three of the twenty are in no specification: each was a real
defect found while building, and a guard nobody demonstrates is a guard nobody has
reason to trust.

---

## 18. What is enforced rather than documented

Convention fails under deadline pressure. A build that fails is evidence; a comment
is a promise.

| Guard | Mechanism | Test |
|---|---|---|
| Import bans (22 pairs) | AST walk resolving relative and dynamic imports | `test_import_bans.py` |
| One clock | AST scan for `datetime.now`, aliased and indirect | `test_no_direct_datetime_now` |
| Gate purity | No I/O, no clock, no model calls on the evaluation path | `test_gate_performs_no_io_and_reads_no_clock` |
| No compliance rule in the Allocator | AST walk of `arc/allocator/` | `test_allocator_contains_no_compliance_rule_of_its_own` |
| No decisions in channels | AST walk, identifiers **and** string literals | `test_channels_contain_no_decision_logic` |
| No decisions in adapters | AST walk of `arc/ingest/adapters/` | `test_adapters_contain_no_decision_logic` |
| Sentinel check order | `ORDERED_CHECKS` iterated, nothing called by name | `test_diagnose_iterates_the_declared_order_and_calls_nothing_by_name` |
| Observability boundary | Attribute, `__dict__`, dataclass, pickle and graph walks | `test_simulator.py` |
| Force never overstated | Re-reads rendered HTML for the word "statutory" | `assert_no_overstated_force` |
| No global RNG | Every generator seeded and injected | `test_no_global_rng_and_replay_is_exact` |
| Append-only ledger | Postgres trigger **and** the hash chain | `test_ledger.py` |
| Absorbing states | Postgres trigger **and** the Python table | `test_core.py`, `test_conductor.py` |
| Every stop subscribed | `assert_cancels_on_every_stop` | `test_inngest_fns.py` |
| No config can disable a voice rule | Reads `VoiceConfig`'s own fields | `assert_no_configuration_can_disable` |

Each AST guard is itself tested against a planted violation, so the scanner is not
trusted on an empty tree.

A ban whose package directory does not exist **skips** rather than passes, so the
gap is visible in `pytest -v` instead of reading as green. Exactly one ban is in
that state today: `arc/money -> arc.llm_service`, because there is no `arc/money/`
package — money lives in `arc/core/money.py`. The ban is pending, not satisfied.

**Current state:** 513 tests across 18 files. `make test` gives 512 passed, 1
skipped in 613 seconds, against Postgres 16. The skip is the pending ban above.

---

## 19. Determinism and replay

Replay is a product feature, not a test convenience. Four things make it hold:

1. **One clock.** Nothing reads wall time except `TimeAuthority`, and every function
   that needs "now" receives it.
2. **Seeded, injected randomness.** No global RNG. Every module that samples takes
   `rng: np.random.Generator` as a parameter. The one exception is the subject
   store's data key, which must not be reproducible.
3. **Derived identifiers.** Claim ids come from `(source, event_id)`; certificate
   ids from the registry version, claim, action, moment and every verdict.
4. **Pinned rule versions.** A certificate names the registry content hash it was
   evaluated under, so a replay cannot silently re-decide under today's rules.

Postgres is configured with `--locale=C`, UTF-8 and UTC, because collation affects
`ORDER BY` on text columns and replay must be byte-identical.

---

## 20. Operational surface

```bash
make up        # Postgres 16 via docker compose, waits for healthy
make migrate   # apply migrations/*.sql in order, once each
make test      # pytest
make lint      # ruff check + ruff format --check
make validate  # simulator distributions against published anchors
make console   # build the four screens into console/
make demo      # deterministic replay, SEED=3
make demo-live # real-time, jittered
make demo-adversarial
```

Python 3.11 under `uv`, CPU-only throughout. Migrations are plain SQL applied in
filename order and recorded with a SHA-256, so editing an applied migration is an
error rather than a silent divergence. No ORM: the outbox depends on index and
locking details an ORM would obscure.

CI runs lint, migrate and test against a real Postgres 16 service on every push.

Five migrations: `001_core` (domain enums, claims, absorbing-state trigger),
`002_ledger` (the three stores), `003_ingest` (archive, dedupe, arms),
`004_conductor` (outbox, budgets, reservations, network attempts), `005_control`
(durable runs and steps, events, kill switch, breakers, held work, erasure
register).

---

## 21. Known limitations

Stated here rather than left to be discovered.

- **GI-7 is unaccounted for.** Eight of the nine invariants are referenced in code;
  the seventh has no reference anywhere in the tree.
- **The paired arm comparison is not a between-subjects randomisation.** It is
  stronger and it is only available because the world is synthetic. The
  subject-level assignment machinery is real and is what would run in production,
  but the reported numbers come from the paired design.
- **Prevention is an expectation, not a realised figure.** It is the ground-truth
  increase in next-presentation success for accounts never contacted, times the
  amount — forward-looking, and reported apart from realised recovery.
- **The cohort detector has a measured blind spot.** The forty-minute outage on a
  thin issuer is a case it legitimately misses. CB-COHORT-BLIND exists so that miss
  surfaces at 40% rather than passing as a clean NORMAL.
- **The DR estimator's error is larger on the judged seed than on the develop
  seed** — 8.59% against 2.00% in the current build. Both are shown; reporting only
  the develop figure would be selecting the seed after seeing the result, which is
  what the three-seed discipline exists to prevent.
- **The spend denominator is marginal channel cost only** — messaging, retries and
  voice minutes. Compute, human-tier time and amortised build are not in it, and
  the console says so on the screen.
- **No LLM provider is wired.** `LLM_ENABLED` is false by default and the
  deterministic path is the supported configuration. A provider drops into
  `LlmClient.invoke`; the boundary, the validator and the fallback all exist and are
  exercised.
- **The console is server-rendered HTML, not React.** The trade was a screen the
  test suite can assert on against a screen with a build step.
- **One import ban is pending rather than satisfied.** `arc/money -> arc.llm_service`
  names a package that does not exist. It skips, visibly, rather than reading as
  green.
- **Exactly-once delivery is not provided and is not claimed.**

---

## 22. Where to start reading

| If you want | Read |
|---|---|
| The domain vocabulary | `arc/core/types.py` |
| Why erasure and an audit chain coexist | `arc/ledger/pii_guard.py`, `arc/ledger/subject_store.py` |
| The compliance model | `arc/gate/evaluator.py`, then `arc/gate/rules/*.yaml` |
| Why the ordering in ingest matters | `arc/ingest/pipeline.py` |
| The single most consequential ordering | `arc/sentinel/diagnose.py` |
| Why the policy is stochastic | `arc/allocator/policy.py` |
| What "exactly once" actually means here | `arc/conductor/commit.py` |
| Why the headline number is defensible | `arc/proving_ground/dr_estimator.py`, `metrics.py` |
| Whether any of it is true | `tests/`, and `make demo-adversarial` |
