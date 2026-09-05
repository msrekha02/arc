"""Five arms, one batch, and the ground truth that validates the estimate.

THE PAIRED DESIGN, AND WHY IT IS HONEST HERE. In production the arms would be
a randomised split: each subject is assigned once, at subject level, and sees
exactly one policy. `arms.py` implements precisely that and M5 tests it,
because it is the only design available when there is one real world.

A simulator has more than one. `World.fork()` was frozen at M4 for this
purpose: it returns a fresh interaction history over the SAME population, so
every arm can be run over every subject without arm B's contacts raising arm
E's annoyance. That is a paired comparison rather than a between-subjects one,
it removes between-arm sampling variation entirely, and it is available only
because the world is synthetic.

    SO THE HONEST STATEMENT IS: the randomisation machinery is real and is
    what would run against a real population; the numbers below come from the
    stronger paired design the simulator permits. Both are reported - the
    stratified assignment is computed and carried on every row - and the
    limitation is stated rather than discovered.

ONLY ARC'S LOGS ARE EVALUATED OFF-POLICY. The other four arms are
deterministic, so they have no propensity to condition on and off-policy
evaluation of them is undefined. They do not need it: each is run directly
against the world and measured on-policy. The doubly-robust estimator exists
to answer what a DIFFERENT policy would have recovered from ARC's logs, and
that question needs a stochastic logging policy, which is what the epsilon
floor at M8 guarantees.

GROUND TRUTH IS READ BEFORE THE WORLD MOVES. `counterfactual()` is a function
of state, and taking an action changes the state. So every row's truth vector
is captured immediately before the action is applied, at exactly the state the
decision was made in. Reading it afterwards would compare the estimate against
a different question.

WHAT PREVENTION MEANS HERE, PRECISELY. Money that never failed was never
recovered, so prevention gets its own line and its own definition: for an
account that was NEVER CONTACTED and did not pay, the increase in the
ground-truth probability that its next presentation succeeds, times the
amount. That is the value of having removed a blocking fault at the rail -
an orphaned mandate re-registered, a stale credential refreshed - with zero
customer contact. It is forward-looking and it is an expectation, and it is
therefore reported apart from realised recovery rather than added to it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from uuid import UUID

import numpy as np

import arc.simulator.response_model as rm
from arc.allocator.budgets import CONTACT_ACTIONS, BudgetKey, Budgets, cost_of
from arc.core.ids import deterministic_uuid, subject_token
from arc.core.money import Paise, paise
from arc.core.time_authority import TimezoneBasis, TzBasisKind
from arc.core.types import (
    ActionType,
    Cause,
    Claim,
    ClaimState,
    ClaimType,
    CohortVerdict,
    DiagnosisPath,
    Rail,
)
from arc.gate.context import (
    ACTION_CHANNEL,
    CONTACT_CHANNELS,
    Channel,
    ConsentState,
    ContactEvent,
    ContactOutcome,
    DeclineCategory,
    GateContext,
    RetryEvent,
    SubjectFlags,
)
from arc.proving_ground.arms import (
    Arm,
    Strata,
    assign_arm,
    claim_count_bucket,
    decile_cutoffs,
    value_decile,
)
from arc.proving_ground.composed import (
    DecisionKey,
    composed_propensity,
    veto_diagnostics,
)
from arc.proving_ground.dr_estimator import LoggedDecision
from arc.proving_ground.metrics import (
    ArmReport,
    Diagnostics,
    Headline,
    Scoreboard,
    guardrails_from_counts,
)
from arc.proving_ground.policies import ArmPolicy, ClaimCase, SubjectCase, build_arms
from arc.sentinel.code_map import code_lookup
from arc.sentinel.cohort import CohortHistory
from arc.simulator.seeds import EPOCH, Stream, rng
from arc.simulator.world import World, sleeping_dogs

# Pepper for the harness's own token derivation. A literal, because the batch
# must be byte-identical across runs and a random pepper would move every
# stratum and therefore every arm assignment.
HARNESS_PEPPER = b"arc-proving-ground-pepper-000001"

# Claim ids are derived from the account, not drawn, so a replay reconstructs
# the same claim and the same ledger keys.
CLAIM_NAMESPACE = UUID("6f3c1a52-9d84-4f27-8b61-2a0e5c7d4419")

# Cycles are two days apart. Close enough that FREQ-7D (three contacts per
# rolling seven days) binds inside the run, which is the point: a frequency
# cap that never fires has not been demonstrated.
CYCLE_GAP = timedelta(days=2)

# 06:00 UTC is 11:30 in Kolkata - inside the statutory contact window, so the
# window is not silently blocking every arm and hiding the comparison.
FIRST_CYCLE_OFFSET = timedelta(hours=6)

DEFAULT_CYCLES = 4

# `tz_basis` is a RECORDED DECISION rather than a guess. The three candidate
# bases - declared preference, billing address, telecom circle - disagree, and
# picking one silently is how a system produces out-of-hours contact with a
# clean audit log. This batch is Indian recurring payments billed to an Indian
# address, so the billing address is the basis and the Gate is told so.
#
# It must be a `TimezoneBasis` and not a bare `ZoneInfo`: the Gate reads an
# unresolved timezone as unresolved and BLOCKS every time-bounded channel,
# which is the correct fail-closed behaviour and not something to work around.
IST_BASIS = TimezoneBasis(kind=TzBasisKind.BILLING_ADDRESS, zone="Asia/Kolkata")

# How long before a presentation the pre-debit notice went out. The e-mandate
# framework requires not less than twenty-four hours, and a compliant merchant
# sends it before the original debit - so a claim arriving in this batch has
# one behind it. Without it NET-PREDEBIT has no computable eligible time and
# blocks every retry, which is right for a claim that genuinely has no notice
# and wrong for this batch.
PREDEBIT_LEAD = timedelta(hours=26)

# A subscription's remaining lifetime value, in months of the plan. Derived
# from the observable plan value alone; nothing latent enters it.
# source: subscription retention curves in Indian recurring-payments books,
# where median remaining tenure at the point of a failed debit sits around a
# year for an established relationship.
LTV_MONTHS = 12

# Outcomes that close a claim: nothing further is attempted this run.
CLOSING_OUTCOMES: frozenset[rm.OutcomeKind] = frozenset(
    {
        rm.OutcomeKind.PAID,
        rm.OutcomeKind.OPT_OUT,
        rm.OutcomeKind.VOLUNTARY_CANCEL,
        rm.OutcomeKind.DISPUTE,
    }
)

_HARD_LABELS = {"hard_decline"}
_STOP_LABELS = {"do_not_retry"}


@dataclass
class _AccountState:
    """One account's evolving position inside one arm's run."""

    account_id: str
    claim_id: UUID
    amount_paise: Paise
    rail: Rail
    contacts: list[ContactEvent] = field(default_factory=list)
    retries: list[RetryEvent] = field(default_factory=list)
    closed: bool = False
    contacted: bool = False
    paid_paise: int = 0
    promise_due: datetime | None = None
    promise_resolved: str | None = None
    baseline_retry_p: float = 0.0


@dataclass
class ArmRun:
    """Everything one arm produced, before it becomes a report."""

    arm: Arm
    logs: list[LoggedDecision] = field(default_factory=list)
    recovered_paise: int = 0
    spend_paise: int = 0
    prevented_paise: int = 0
    contacts: int = 0
    right_party_contacts: int = 0
    complaints: int = 0
    opt_outs: int = 0
    cancellations: int = 0
    disputes: int = 0
    promises_made: int = 0
    promises_kept: int = 0
    promises_unresolved: int = 0
    treated_subjects: int = 0
    subjects: int = 0
    veto_rate: float = 0.0
    veto_mass: float = 0.0
    explore_mass_share: float = 0.0
    shadow_prices: Mapping[str, float] = field(default_factory=dict)


@dataclass
class HarnessResult:
    """All five runs plus the batch they shared."""

    runs: dict[Arm, ArmRun]
    seed: int
    cycles: int
    subjects: int
    strata: Mapping[str, Strata]
    assignments: Mapping[str, Arm]
    # The batch every arm saw, and the moment the first cycle ran at. Carried
    # so a reader of the result - the console, a replay trace - can re-derive
    # a diagnosis or a claim without re-running the world and getting a
    # different one.
    cases: tuple[SubjectCase, ...] = ()
    at0: datetime = EPOCH
    world: World | None = None
    # GROUND TRUTH, COUNTED HERE AND CARRIED OUT AS INTEGERS.
    #
    # `sleeping_dogs` reads the simulator's counterfactuals, and only the
    # simulator and this harness may do that - `test_import_bans` sweeps every
    # other package for it. The console needs the number and is not allowed to
    # ask for it, which is the correct arrangement: a presentation layer that
    # can reach ground truth can render a figure the running system could never
    # have known. So the question is asked once, on the inside, and what leaves
    # is a count.
    sleeping_dogs_planted: int = 0
    sleeping_dogs_contacted: Mapping[Arm, int] = field(default_factory=dict)

    def arc(self) -> ArmRun:
        return self.runs[Arm.ARC]

    def issuer_map(self) -> dict[str, str | None]:
        """account -> issuer, resolved once. Observable path only."""
        if self.world is None:
            return {}
        return {
            account_id: getattr(self.world.observe(account_id, self.world.epoch), "issuer_id", None)
            for account_id in self.world.account_ids
        }

    def cohort_history(self) -> CohortHistory:
        """The batch's own presentation events, as the Sentinel sees them.

        Captures as well as failures: the denominator is not optional, and a
        detector fed only failures cannot tell a busy hour from a broken
        issuer.
        """
        history = CohortHistory()
        if self.world is None:
            return history
        issuers = self.issuer_map()
        for event in self.world.batch_events():
            history.record(
                issuers.get(event.account_id),
                event.rail,
                event.at,
                succeeded=event.succeeded,
            )
        return history

    def detection_moments(self) -> dict[str, tuple[datetime, str | None]]:
        """When each account's failure was DETECTED, and with what code.

        A claim is diagnosed at the moment its failure arrives, not at the
        moment a later cycle happens to run. Diagnosing everything at the
        cycle time would look tidy and would miss every outage in the batch
        window, because by then it has resolved - which is exactly the number
        the batch screen exists to show.
        """
        if self.world is None:
            return {}
        latest: dict[str, tuple[datetime, str | None]] = {}
        for event in self.world.batch_events():
            if event.succeeded:
                continue
            current = latest.get(event.account_id)
            if current is None or event.at > current[0]:
                latest[event.account_id] = (event.at, event.decline_code)
        return latest


# ---------------------------------------------------------------------------
# Building the batch
# ---------------------------------------------------------------------------
def _consent_map(observation: object) -> dict[Channel, ConsentState]:
    """The subject's consent, translated into the Gate's vocabulary.

    A channel the record does not mention stays absent, and the Gate reads
    absence as UNKNOWN and fails it closed (GI-5). The world only records
    consent for four channels, so payment link, human handoff and postal
    notice are unreachable in this batch - correctly, and visibly, rather
    than by a default that would have quietly granted them.
    """
    granted: dict[Channel, ConsentState] = {
        # Rail-level actions reach nobody, so there is no consent question.
        Channel.SILENT: ConsentState.GRANTED,
        Channel.NONE: ConsentState.GRANTED,
    }
    for name, state in observation.channel_consent_state:  # type: ignore[attr-defined]
        try:
            channel = Channel(name)
        except ValueError:
            continue
        granted[channel] = ConsentState(state)
    # An instalment offer is made on a call, so it inherits the voice answer.
    if Channel.VOICE in granted:
        granted[Channel.INSTALMENT] = granted[Channel.VOICE]
    return granted


def _decline_category(rail: Rail, codes: Sequence[str]) -> DeclineCategory:
    """The last decline, through the Sentinel's own code map.

    The production path rather than a lookup invented here, so the harness
    inherits the same treatment of remapped and unmapped codes that the rest
    of the system gets - including the five percent the world deliberately
    mislabels.
    """
    if not codes:
        return DeclineCategory.NONE
    meaning = code_lookup(rail, codes[-1])
    if meaning.label.value in _HARD_LABELS:
        return DeclineCategory.LOST_OR_STOLEN
    if meaning.label.value in _STOP_LABELS:
        return DeclineCategory.STOP_PAYMENT
    return DeclineCategory.SOFT


def _cause_for(rail: Rail, codes: Sequence[str]) -> Cause:
    """A diagnosed cause from the observable code history.

    The Sentinel is M6's component and has its own gate; what the harness
    needs is a cause of the right SHAPE so the Gate's confidence rule behaves
    as it would in the pipeline. An unmapped code arrives here as UNKNOWN at
    zero confidence, exactly as it should.
    """
    meaning = code_lookup(rail, codes[-1] if codes else None)
    return Cause(
        label=meaning.label,
        layer=meaning.layer,
        confidence=meaning.confidence,
        derived_from=DiagnosisPath.CODE_MAP,
        cohort_power=CohortVerdict.NORMAL,
    )


def _claim_type_for(rail: Rail) -> ClaimType:
    if rail is Rail.INVOICE:
        return ClaimType.INVOICE_OVERDUE
    if rail is Rail.CARD:
        return ClaimType.CARD_DECLINE
    return ClaimType.MANDATE_FAILURE


def build_batch(
    world: World, at: datetime
) -> tuple[list[SubjectCase], dict[str, Strata], dict[str, Arm]]:
    """One `SubjectCase` per subject, from the observable path only.

    Accounts sharing a `customer_ref` are one subject and therefore one unit
    of budget contention, one unit of contact accounting and one arm.
    """
    by_subject: dict[str, list[tuple[str, object]]] = {}
    for account_id in world.account_ids:
        observation = world.observe(account_id, at)
        token = subject_token(world.customer_ref(account_id), pepper=HARNESS_PEPPER)
        by_subject.setdefault(token, []).append((account_id, observation))

    totals = {
        token: paise(sum(int(o.plan_value_paise) for _, o in entries))
        for token, entries in by_subject.items()
    }
    cutoffs = decile_cutoffs(list(totals.values()))

    cases: list[SubjectCase] = []
    strata: dict[str, Strata] = {}
    assignments: dict[str, Arm] = {}

    for token in sorted(by_subject):
        entries = sorted(by_subject[token])
        first_observation = entries[0][1]
        stratum = Strata(
            claim_count_bucket=claim_count_bucket(len(entries)),
            value_decile=value_decile(totals[token], cutoffs),
            rail=first_observation.rail,  # type: ignore[attr-defined]
        )
        strata[token] = stratum
        assignments[token] = assign_arm(token, "m11-proving-ground", stratum)

        claims: list[ClaimCase] = []
        for account_id, observation in entries:
            claims.append(_claim_case(token, account_id, observation, at))
        cases.append(SubjectCase(subject_token=token, claims=tuple(claims), stratum=stratum.key))

    return cases, strata, assignments


def _claim_case(token: str, account_id: str, observation: object, at: datetime) -> ClaimCase:
    rail: Rail = observation.rail  # type: ignore[attr-defined]
    amount: Paise = observation.plan_value_paise  # type: ignore[attr-defined]
    identifier = deterministic_uuid(CLAIM_NAMESPACE, account_id)

    claim = Claim(
        claim_id=identifier,
        subject_token=token,
        amount_paise=amount,
        ltv_remaining_paise=paise(int(amount) * LTV_MONTHS),
        claim_type=_claim_type_for(rail),
        rail=rail,
        detected_at=at,
        state=ClaimState.DIAGNOSED,
    )
    return ClaimCase(
        claim=claim,
        account_id=account_id,
        gate_ctx=_context(claim, observation, contacts=(), retries=()),
        contacts_7d=observation.contact_history_7d,  # type: ignore[attr-defined]
        observation=observation,
    )


def _context(
    claim: Claim,
    observation: object,
    *,
    contacts: Sequence[ContactEvent],
    retries: Sequence[RetryEvent],
) -> GateContext:
    """The Gate's view, rebuilt each cycle so cooldowns actually accumulate.

    A context frozen at the start of the run would let every cooldown pass on
    every cycle, and the frequency caps would never fire. The whole point of
    running four cycles is that they do.
    """
    codes: tuple[str, ...] = observation.decline_code_history  # type: ignore[attr-defined]
    return GateContext(
        claim_id=claim.claim_id,
        subject_token=claim.subject_token,
        rail=claim.rail,
        claim_state=claim.state,
        amount_paise=claim.amount_paise,
        tz_basis=IST_BASIS,
        cause=_cause_for(claim.rail, codes),
        predebit_notice_at=claim.detected_at - PREDEBIT_LEAD,
        mandate_cap_paise=observation.mandate_cap_paise,  # type: ignore[attr-defined]
        decline_category=_decline_category(claim.rail, codes),
        consent=_consent_map(observation),
        contacts=tuple(contacts),
        retries=tuple(retries),
        flags=SubjectFlags(identity_verified=True),
    )


# ---------------------------------------------------------------------------
# Running one arm
# ---------------------------------------------------------------------------
def run_arm(
    policy: ArmPolicy,
    base: World,
    cases: Sequence[SubjectCase],
    gate: object,
    strata: Mapping[str, Strata],
    *,
    cycles: int = DEFAULT_CYCLES,
    at0: datetime,
    seed: int,
    record_truth: bool = True,
) -> ArmRun:
    """One policy, over its own fork of the world, for `cycles` cycles."""
    world = base.fork()
    generator = rng(seed, Stream.OUTCOME)
    run = ArmRun(arm=policy.arm, subjects=len(cases))

    states: dict[str, _AccountState] = {}
    observations: dict[str, object] = {}
    for case in cases:
        for claim_case in case.claims:
            observation = world.observe(claim_case.account_id, at0)
            observations[claim_case.account_id] = observation
            states[claim_case.account_id] = _AccountState(
                account_id=claim_case.account_id,
                claim_id=claim_case.claim_id,
                amount_paise=claim_case.claim.amount_paise,
                rail=claim_case.claim.rail,
                # Ground truth, harness only: the probability a plain retry
                # succeeds before anything has been attempted. The prevention
                # line is measured against this.
                baseline_retry_p=world.counterfactual(claim_case.account_id, ActionType.RETRY, at0),
            )

    treated: set[str] = set()
    composed_policies = []

    for cycle in range(cycles):
        at = at0 + cycle * CYCLE_GAP
        live = _live_cases(cases, states, observations)
        if not live:
            break

        draws = policy.decide(live, cycle, at, generator)
        by_token = {case.subject_token: case for case in live}

        for draw in draws:
            case = by_token.get(draw.subject_token)
            if case is None:
                continue

            if policy.gated:
                # BOTH FILTERS, IN ORDER. Admission first, then the Gate. See
                # `composed.resolve_branch`: leaving admission out of the
                # composition is what makes the logged propensity disagree
                # with the realised action distribution.
                composed = composed_propensity(
                    draw.pi_alloc,
                    gate,
                    case.contexts,
                    at,
                    admissible=draw.admissible,
                )
                composed_policies.append(composed)
                realized = composed.realized(draw.intended)
                pi_exec = dict(composed.pi_exec)
                pi_realized = composed.propensity_of(realized)
                blocking = next(
                    (r.blocking_rule_ids for r in composed.resolutions if r.key == draw.intended),
                    (),
                )
            else:
                # THE UNCONSTRAINED ARM. No certificate, no cooldown, no
                # contact window. It acts against the simulator alone and
                # never reaches a channel, which is the only place that is
                # safe - and the guardrail columns are what it costs.
                realized = draw.intended
                pi_exec = dict(draw.pi_alloc)
                pi_realized = float(draw.pi_alloc.get(realized, 1.0))
                blocking = ()

            _apply(
                run=run,
                world=world,
                case=case,
                states=states,
                observations=observations,
                draw=draw,
                realized=realized,
                pi_exec=pi_exec,
                # THE ALLOCATOR'S OWN MASS FOR THE SAMPLED BRANCH, indexed and
                # never defaulted.
                #
                # WHAT WAS HERE BEFORE. `pi_exec.get(intended, pi_realized)`.
                # `pi_exec` is the composed EXECUTION distribution, so a refused
                # branch is absent from it entirely and the default handed back
                # `pi_realized` instead. Every vetoed row therefore logged the
                # realized propensity twice under two different names, and the
                # replay screen printed "sampled whatsapp_utility at 0.771,
                # realized do_nothing at 0.771" - two numbers that cannot both
                # be right, from one row, with nothing failing.
                #
                # The estimate was never affected, because it divides by
                # `pi_behaviour` which reads `pi_realized`. That is luck rather
                # than design: a silent default on a propensity field is one
                # rename away from being read by the arithmetic.
                #
                # So this indexes. `draw.intended` is drawn FROM `pi_alloc`, so
                # a missing key is not a state the sampler can produce, and a
                # KeyError here is the correct outcome for a bug that would
                # otherwise be invisible.
                pi_intended=float(draw.pi_alloc[draw.intended]),
                pi_realized=pi_realized,
                blocking=tuple(blocking),
                cycle=cycle,
                at=at,
                generator=generator,
                stratum=strata.get(case.subject_token),
                treated=treated,
                record_truth=record_truth,
            )

    _close_promises(run, states, at0 + cycles * CYCLE_GAP)
    _measure_prevention(run, world, states, at0 + cycles * CYCLE_GAP)

    run.treated_subjects = len(treated)
    if composed_policies:
        diagnostics = veto_diagnostics(composed_policies)
        run.veto_rate = diagnostics["veto_rate"]
        run.veto_mass = diagnostics["mean_veto_mass"]
    run.shadow_prices = dict(getattr(policy, "last_shadow_prices", {}))
    run.explore_mass_share = float(getattr(policy, "last_explore_mass", 0.0))
    return run


def _live_cases(
    cases: Sequence[SubjectCase],
    states: Mapping[str, _AccountState],
    observations: Mapping[str, object],
) -> list[SubjectCase]:
    """Subjects with at least one open claim, with contexts refreshed.

    Rebuilding the context here is what makes a cooldown bind on the next
    cycle rather than on the next run.
    """
    live: list[SubjectCase] = []
    for case in cases:
        open_claims = []
        for claim_case in case.claims:
            state = states[claim_case.account_id]
            if state.closed:
                continue
            open_claims.append(
                ClaimCase(
                    claim=claim_case.claim,
                    account_id=claim_case.account_id,
                    gate_ctx=_context(
                        claim_case.claim,
                        observations[claim_case.account_id],
                        contacts=state.contacts,
                        retries=state.retries,
                    ),
                    contacts_7d=len(state.contacts),
                    observation=observations[claim_case.account_id],
                )
            )
        if open_claims:
            live.append(
                SubjectCase(
                    subject_token=case.subject_token,
                    claims=tuple(open_claims),
                    stratum=case.stratum,
                )
            )
    return live


def _truth_vector(
    world: World,
    case: SubjectCase,
    keys: Sequence[DecisionKey],
    at: datetime,
) -> dict[DecisionKey, float]:
    """Expected reward in paise under each decision, read before acting.

    EVALUATION HARNESS ONLY. This is the answer key, and it is captured at the
    state the decision was made in because taking the action changes that
    state.
    """
    truth: dict[DecisionKey, float] = {}
    for key in keys:
        claim_id, action = key
        claim_case = case.case_for(claim_id) or case.claims[0]
        probability = world.counterfactual(claim_case.account_id, action, at)
        truth[key] = probability * float(int(claim_case.claim.amount_paise))
    return truth


def _apply(
    *,
    run: ArmRun,
    world: World,
    case: SubjectCase,
    states: dict[str, _AccountState],
    observations: Mapping[str, object],
    draw: object,
    realized: DecisionKey,
    pi_exec: Mapping[DecisionKey, float],
    pi_intended: float,
    pi_realized: float,
    blocking: tuple[str, ...],
    cycle: int,
    at: datetime,
    generator: np.random.Generator,
    stratum: Strata | None,
    treated: set[str],
    record_truth: bool,
) -> None:
    """Act on the world, then record the row. Truth first, action second."""
    claim_id, action = realized
    claim_case = case.case_for(claim_id) or case.claims[0]
    state = states[claim_case.account_id]

    truth = _truth_vector(world, case, list(pi_exec), at) if record_truth else {}

    outcome = world.outcome(claim_case.account_id, action, at, generator)

    channel = ACTION_CHANNEL[action]
    if channel in CONTACT_CHANNELS:
        run.contacts += 1
        state.contacted = True
        treated.add(case.subject_token)
        if not outcome.wrong_party:
            run.right_party_contacts += 1
        state.contacts.append(
            ContactEvent(
                at=at,
                channel=channel,
                outcome=(
                    ContactOutcome.WRONG_NUMBER if outcome.wrong_party else ContactOutcome.DELIVERED
                ),
            )
        )
    if action in rm.DEBIT_ACTIONS:
        state.retries.append(RetryEvent(at=at, rail=state.rail))

    run.spend_paise += int(cost_of(action).rupee_paise)

    reward = int(outcome.paid_paise)
    if outcome.kind is rm.OutcomeKind.PAID:
        run.recovered_paise += reward
        state.paid_paise += reward
    elif outcome.kind is rm.OutcomeKind.COMPLAINT:
        run.complaints += 1
    elif outcome.kind is rm.OutcomeKind.OPT_OUT:
        run.opt_outs += 1
    elif outcome.kind is rm.OutcomeKind.VOLUNTARY_CANCEL:
        run.cancellations += 1
    elif outcome.kind is rm.OutcomeKind.DISPUTE:
        run.disputes += 1

    if outcome.promise is not None and state.promise_due is None:
        run.promises_made += 1
        state.promise_due = outcome.promise.due_at
    elif state.promise_due is not None and outcome.kind is rm.OutcomeKind.PAID:
        # Paid before the promise fell due. Resolved KEPT here rather than at
        # the sweep, because the payment is what resolves it.
        if at <= state.promise_due:
            state.promise_resolved = "kept"
            run.promises_kept += 1

    if outcome.kind in CLOSING_OUTCOMES:
        state.closed = True

    # NO FALLBACK ON EITHER OF THESE. `draw.intended` is what the allocator
    # sampled and it always exists; substituting `realized` for it would make a
    # vetoed row look like a clean one. See `pi_intended` below for why the
    # same applies to the probability.
    intended = draw.intended  # type: ignore[attr-defined]
    run.logs.append(
        LoggedDecision(
            subject_token=case.subject_token,
            cycle=cycle,
            stratum=stratum.key if stratum is not None else "",
            intended_key=intended,
            pi_intended=pi_intended,
            realized_key=realized,
            pi_realized=pi_realized,
            veto_occurred=intended != realized,
            blocking_rule_ids=blocking,
            reward_paise=reward,
            cost_paise=int(cost_of(action).rupee_paise),
            pi_exec=dict(pi_exec),
            truth=truth,
        )
    )


def _close_promises(run: ArmRun, states: Mapping[str, _AccountState], horizon: datetime) -> None:
    """Resolve promises at the horizon. UNRESOLVED IS NOT BROKEN.

    A promise dated after the run ends is censored, not failed. Coding it
    broken is what biases a promise-to-pay model pessimistic, and the count is
    reported separately so the censoring is visible rather than absorbed.
    """
    for state in states.values():
        if state.promise_due is None or state.promise_resolved is not None:
            continue
        if state.promise_due >= horizon:
            state.promise_resolved = "unresolved"
            run.promises_unresolved += 1
        else:
            state.promise_resolved = "broken"


def _measure_prevention(
    run: ArmRun, world: World, states: Mapping[str, _AccountState], at: datetime
) -> None:
    """Leakage prevented: a fault removed without anyone being messaged.

    Restricted to accounts NEVER CONTACTED, so the measurement cannot pick up
    a response to outreach - and to accounts that did not pay, so it cannot
    overlap with recovery. What remains is the increase in the ground-truth
    probability that the next presentation succeeds, valued at the amount.

    A SEPARATE LINE, ALWAYS. Money that never failed was never recovered.
    """
    for state in states.values():
        if state.contacted or state.paid_paise:
            continue
        now = world.counterfactual(state.account_id, ActionType.RETRY, at)
        gain = now - state.baseline_retry_p
        if gain > 0.0:
            run.prevented_paise += int(round(gain * float(int(state.amount_paise))))


# ---------------------------------------------------------------------------
# Running all five
# ---------------------------------------------------------------------------
def default_budgets(subjects: int) -> Budgets:
    """Caps sized to the portfolio, so the shadow prices are non-trivial.

    Deliberately BINDING. A budget that never binds prices at zero and proves
    nothing about the allocator; these are set so contact and voice are both
    scarce, which is what makes lambda a number worth showing.
    """
    return Budgets(
        {
            BudgetKey.CONTACT: max(subjects // 4, 1),
            BudgetKey.VOICE: max(subjects // 12, 1),
            BudgetKey.RUPEE: max(subjects * 60, 1),
            BudgetKey.RETRY: max(subjects // 2, 1),
        }
    )


def run_all(
    *,
    seed: int,
    size: int = 1_600,
    cycles: int = DEFAULT_CYCLES,
    gate: object,
    budgets: Budgets | None = None,
) -> HarnessResult:
    """Every arm, on the same batch, each on its own fork of the world."""
    base = World(seed=seed, size=size)
    at0 = EPOCH + FIRST_CYCLE_OFFSET
    cases, strata, assignments = build_batch(base, at0)
    caps = budgets or default_budgets(len(cases))

    runs: dict[Arm, ArmRun] = {}
    for policy in build_arms(gate, caps):
        runs[policy.arm] = run_arm(
            policy,
            base,
            cases,
            gate,
            strata,
            cycles=cycles,
            at0=at0,
            seed=seed,
            # Only ARC's rows feed the doubly-robust estimator, and the truth
            # vector is the expensive part of a row.
            record_truth=policy.arm is Arm.ARC,
        )

    planted, reached = _sleeping_dog_reach(base, cases, runs, at0)
    return HarnessResult(
        runs=runs,
        seed=seed,
        cycles=cycles,
        subjects=len(cases),
        strata=strata,
        assignments=assignments,
        cases=tuple(cases),
        at0=at0,
        world=base,
        sleeping_dogs_planted=planted,
        sleeping_dogs_contacted=reached,
    )


def _sleeping_dog_reach(
    world: World,
    cases: Sequence[SubjectCase],
    runs: Mapping[Arm, ArmRun],
    at: datetime,
) -> tuple[int, dict[Arm, int]]:
    """How many planted sleeping dogs each arm contacted.

    A PLANTED SLEEPING DOG IS GROUND TRUTH, not the forecaster's opinion of
    one. `sleeping_dogs` returns the accounts whose counterfactual under every
    digital nudge is worse than doing nothing, so "how many did we contact"
    is a question about behaviour rather than about the model agreeing with
    itself.

    CONTACT MEANS ANY CONTACT. The cohort is defined over digital nudges, but
    an arm can reach the same account by voice or by a human handoff, and the
    unconstrained arm does exactly that. Counting only nudges reports it at
    zero while it contacts most of them.

    COUNTED IN SUBJECTS, because contact is a subject-level act and a subject
    contacted twice is not two harms of the same kind.
    """
    planted_accounts = set(sleeping_dogs(world, at))
    planted_subjects = {
        case.subject_token
        for case in cases
        if {claim.account_id for claim in case.claims} & planted_accounts
    }
    reached = {
        arm: len(
            {
                row.subject_token
                for row in run.logs
                if row.subject_token in planted_subjects and row.realized_key[1] in CONTACT_ACTIONS
            }
        )
        for arm, run in runs.items()
    }
    return len(planted_subjects), reached


DENOMINATOR = (
    "recovered rupees over claims the arm attempted in this batch; "
    "claims never attempted are excluded from the numerator and the denominator alike"
)


def build_scoreboard(
    result: HarnessResult,
    *,
    comparator: Arm = Arm.NAIVE_DUNNING,
    dr_relative_error: float | None = None,
    ci: tuple[Paise, Paise] | None = None,
) -> Scoreboard:
    """Turn the runs into reports. Guardrails are attached, not optional."""
    control = result.runs[Arm.NULL]
    comparator_run = result.runs[comparator]

    reports: list[ArmReport] = []
    for arm in (
        Arm.NULL,
        Arm.NAIVE_DUNNING,
        Arm.GATEWAY_DEFAULT,
        Arm.GREEDY_UNCONSTRAINED,
        Arm.ARC,
    ):
        run = result.runs[arm]
        guardrails = guardrails_from_counts(
            contacts=run.contacts,
            complaints=run.complaints,
            opt_outs=run.opt_outs,
            treated_subjects=run.treated_subjects,
            treated_cancellations=run.cancellations,
            control_subjects=control.subjects,
            control_cancellations=control.cancellations,
            promises_made=run.promises_made,
            promises_kept=run.promises_kept,
            promises_unresolved=run.promises_unresolved,
            right_party_contacts=run.right_party_contacts,
            spend_paise=paise(run.spend_paise),
            recovered_paise=paise(run.recovered_paise),
        )
        headline = Headline(
            arm=arm,
            comparator=comparator,
            recovered_paise=paise(run.recovered_paise),
            comparator_recovered_paise=paise(comparator_run.recovered_paise),
            spend_paise=paise(run.spend_paise),
            denominator=DENOMINATOR,
            guardrails=guardrails,
            ci_low_paise=ci[0] if ci and arm is Arm.ARC else None,
            ci_high_paise=ci[1] if ci and arm is Arm.ARC else None,
        )
        reports.append(
            ArmReport(
                headline=headline,
                prevented_paise=paise(run.prevented_paise),
                subjects=run.subjects,
                diagnostics=Diagnostics(
                    post_allocation_veto_rate=run.veto_rate,
                    explore_mass_share=run.explore_mass_share,
                    dr_relative_error=dr_relative_error if arm is Arm.ARC else None,
                ),
            )
        )

    return Scoreboard(
        reports=tuple(reports),
        comparator=comparator,
        seed=result.seed,
        cycles=result.cycles,
    )
