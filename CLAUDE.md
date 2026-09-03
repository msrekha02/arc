# CLAUDE.md — ARC working conventions

**ARC (Autonomous Revenue Continuity)** detects revenue at risk, diagnoses it,
chooses a bounded intervention, executes it compliantly, and proves what was
recovered.

Companion documents:

| Document | Read it for |
|---|---|
| `ARC_architecture.md` | The spec — the *what* and the *why*. Every design decision carries a `WHY`. |
| `ARC_BUILD.md` | The plan — the milestone order and the acceptance gates. |
| `CLAUDE.md` (this file) | The conventions that hold in every session. |

---

## 0. Session protocol

**One milestone per session.** Do not start milestone N+1 until milestone N's
acceptance gate passes. If a gate fails, fix it inside that milestone. Debt
carried forward compounds.

At the start of a session you will be told which milestone is active. Build
only that milestone. Do not create module stubs for later milestones — they
cause implementations to be written against interfaces that do not exist yet.

A milestone is done when its **runnable** acceptance gate passes and the actual
terminal output has been shown. "It should pass" is not a gate.

### Milestone order

```
M0  Scaffold                  M9  Conductor
M1  Domain + Time + Money     M10 Channels
M2  Ledger                    M11 Proving Ground   <- cut-line, submittable
M3  Rule Registry + Gate      M12 Inngest durable
M4  World Simulator [FREEZE]  M13 Breakers + Kill switch
M5  Ingest + Normalise        M14 Console
M6  Sentinel                  M15 Adversarial suite
M7  Forecaster                M16 Voice [stretch]
M8  Allocator                 M17 Demo harness
```

M0-M11 plus M14 and M17 is a complete, defensible submission. Do not reorder to
build voice early: it demos well and proves nothing.

---

## 1. Global conventions

### 1.1 Money

```python
# arc/core/money.py - the ONLY monetary type
Paise = NewType("Paise", int)
```

- Money is **integer paise**. No `float`, no `Decimal`, anywhere in the repo.
- `amount: float` for a monetary value is a defect, not a style preference.
- Formatting to rupees happens **only** at the presentation boundary.

**WHY:** float arithmetic on money produces silent, compounding errors, and the
headline number of this system is a sum of money. (Global invariant GI-2.)

### 1.2 Time

```python
class TimeAuthority:
    def now(self) -> datetime: ...
    def local(self, utc: datetime, tz_basis: TimezoneBasis) -> datetime: ...
    def is_bank_holiday(self, d: date, region: str) -> bool: ...
    def next_legal_window(self, after: datetime, tz_basis) -> datetime: ...
```

- All timestamps are **timezone-aware UTC**.
- **Nothing** calls `datetime.now()` except `TimeAuthority`. An AST test
  enforces this.
- Every function that needs "now" **receives it as a parameter**. In particular
  the Gate takes `at: datetime`; it never reads a clock.
- All rolling windows are **half-open**: `[t - 7d, t)`.

**WHY:** it makes the Gate pure, makes replay possible, and makes tests
deterministic. Half-open windows make boundary behaviour defined and
property-testable.

### 1.3 Import bans — enforced in CI

```
arc/allocator/    MUST NOT import  arc/llm_service/
arc/gate/         MUST NOT import  arc/llm_service/
arc/gate/         MUST NOT import  arc/models/          (no ML in the Gate)
arc/money/        MUST NOT import  arc/llm_service/
arc/inngest_fns/  MUST NOT import  arc/channels/
arc/sentinel/     MUST NOT import  arc/channels/
arc/simulator/    MUST NOT import  arc/allocator/ arc/forecaster/ arc/gate/
```

The last one is the **anti-circularity guard**: the simulated world must not
know about the policy.

Enforced by `tests/test_import_bans.py`, which AST-walks every file under the
package prefix and resolves relative and dynamic imports before matching. A ban
whose package directory does not exist yet **skips**, so the gap is visible in
`pytest -v` rather than reading as green. Add new bans to `BANS` as milestones
land; never relax one.

**WHY CI-enforced rather than documented:** convention fails under deadline
pressure. A build that fails on a forbidden import is evidence, not a promise.

### 1.4 Determinism

- Every random source is **seeded and injected**. No global RNG, ever.
- Every module that samples takes `rng: np.random.Generator` as a parameter.

**WHY:** "run it again" must produce the same number in front of judges.

### 1.5 Repo layout

```
arc/                          <- repo root
  CLAUDE.md
  pyproject.toml              uv, Python 3.11
  docker-compose.yml          Postgres 16
  Makefile                    up down migrate test lint fmt
  arc/                        <- the Python package
    core/                     money, time_authority, ids, types
    ledger/                   decision_ledger, subject_store, money_ledger, pii_guard
    gate/                     registry, evaluator, lattice, rules/*.yaml
    simulator/                world, response_model, wire_fake, seeds  [FROZEN at M4]
    ingest/                   adapters/, normaliser
    sentinel/                 cohort, mandate_health, code_map, diagnose
    forecaster/               bounce, uplift, ptp, calibration
    allocator/                candidates, lagrangian, policy
    conductor/                outbox, worker, fsm, reservations
    channels/
    inngest_fns/
    proving_ground/           arms, dr_estimator, metrics
    llm_service/              contracts, redactor, validator, client
    console/                  React
  migrations/                 *.sql, applied in filename order
  scripts/
  tests/
```

Subpackages are created by the milestone that needs them, not in advance.

### 1.6 Global invariants

Nine invariants (GI-1 to GI-9) hold everywhere and are asserted in code, not
documented in prose. Violation **halts** rather than continues. See
`ARC_architecture.md` section 3 for the full list and the rationale for each.

---

## 2. The twenty critical points

These are the failure modes that look correct and are not. Each one is a thing
that will feel like the natural implementation and will break a later milestone
or silently invalidate the headline number.

| # | It will want to | Do this instead | Why |
|---|---|---|---|
| 1 | Return an argmax from the allocator | Softmax with epsilon floor, return the propensity | A deterministic policy cannot be evaluated off-policy — this silently destroys the headline metric |
| 2 | Let the Gate read a clock | Pass `at: datetime` always | Purity is what makes replay and testing possible |
| 3 | Short-circuit the Gate on first BLOCK | Evaluate all rules, return the full verdict list | The audit trail needs every verdict, not just the blocker |
| 4 | Write a second rule set for `project()` | One evaluator, filtered by decidability class | Two rule sets drift apart silently |
| 5 | Assign experiment arms per claim | Per subject, inherited by all their claims | Claim-level randomisation violates SUTVA under shared budgets |
| 6 | Drop Gate-vetoed decisions from the DR sample | Collapse them onto `do_nothing` with summed mass | Dropping is selection bias |
| 7 | Put raw evidence text in the decision ledger | Structured fields + ref + hash only | Makes erasure impossible without breaking the chain |
| 8 | Make the PII guard warn and continue | Raise and fail the write | A guard with a bypass is not a guard |
| 9 | Execute an expired certificate "since it's nearly valid" | Cancel and requeue for re-decision | Stale authorisation is the whole reason certificates have windows |
| 10 | Put the attempt counter in the idempotency key | `claim:action:cycle:certificate` only | Retries must reuse the key; re-decisions must not |
| 11 | Use a message broker instead of the outbox | Postgres outbox + `SKIP LOCKED` | The transaction is the guarantee; a broker breaks atomicity with state |
| 12 | Coerce `INSUFFICIENT_POWER` to `NORMAL` | Keep it distinct, cap confidence | Silent no-op on thin issuers is the exact bug we designed around |
| 13 | Use an S-learner for uplift | X-learner | S-learner under-detects when other features dominate |
| 14 | Skip calibration on the bounce model | Isotonic on held-out | Raw GBDT scores are not probabilities and they feed an EV product |
| 15 | Code unresolved promises as broken | Treat as censored | Systematically biases the PTP model pessimistic |
| 16 | Tune simulator constants after seeing results | Freeze at `simulator-frozen-v1`, never touch | This is the circularity attack; the git tag is your defence |
| 17 | Merge prevention into the recovery number | Separate line | Money that never failed was never recovered |
| 18 | Let an Inngest function escalate after a broken promise | Requeue to the allocator | Escalation authority belongs to L4 |
| 19 | Relax a budget when infeasible | Shrink the treated set, log drops | Constraint relaxation under pressure is how compliance systems fail |
| 20 | Label a policy choice as statutory | `basis` + `status` on every rule | Overstating regulatory force discredits the whole compliance story |

---

## 3. Commands

```bash
make up        # docker compose up -d --wait, then show container status
make down      # docker compose down
make migrate   # apply migrations/*.sql in order, once each
make test      # uv run pytest
make lint      # ruff check + ruff format --check
make fmt       # ruff format + ruff check --fix
```

Everything runs CPU-only.
