"""M4 acceptance gate: the seven tests, plus the boundary they defend.

    test_agent_cannot_access_latent_state
    test_same_seed_same_batch
    test_sleeping_dogs_exist
    test_payday_effect_measurable
    test_outage_injected_and_detectable
    test_wire_fake_payloads_hmac_valid
    test_duplicate_and_out_of_order_present

The first one is the milestone. Everything this system reports rests on the
claim that the agent cannot see the answer key, so that claim is tested by
attacking it - attribute access, `__dict__`, dataclass introspection,
pickling, and a walk of the whole object graph - rather than by asserting the
happy path once and moving on.
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import pickle
import types
from datetime import UTC, datetime, timedelta

import pytest
from arc.core.types import ActionType, ClaimType, Rail
from arc.simulator import response_model as rm
from arc.simulator.codes import REMAP_RATE, Semantic, code_for
from arc.simulator.seeds import BATCH_START, DEVELOP_SEED, EPOCH, Stream, rng
from arc.simulator.wire_fake import (
    DUPLICATE_RATE,
    OUT_OF_ORDER_RATE,
    WireFake,
    arrival_inversions,
    late_deliveries,
    sign,
    verify,
)
from arc.simulator.world import (
    FESTIVAL_END,
    FESTIVAL_START,
    LATENT_FIELD_NAMES,
    OBSERVABLE_FIELD_NAMES,
    OUTAGES,
    Account,
    EventKind,
    LatentState,
    ObservableState,
    Promise,
    PromiseStatus,
    World,
    days_since_salary,
    issuer_health,
    sleeping_dogs,
)

SECRET = b"arc-test-secret-key-0123456789ab"
SMALL = 800
MEDIUM = 3_000
LARGE = 20_000


@pytest.fixture(scope="module")
def world() -> World:
    return World(seed=DEVELOP_SEED, size=MEDIUM)


@pytest.fixture(scope="module")
def big_world() -> World:
    """Large enough for the outage to have a sample worth testing."""
    return World(seed=DEVELOP_SEED, size=LARGE)


# ---------------------------------------------------------------------------
# Gate test 1 - the observability boundary
# ---------------------------------------------------------------------------
def _reachable_objects(root: object, limit: int = 400_000) -> list[object]:
    """Everything reachable from `root` by following references.

    Classes, modules and functions are recorded but not descended into: a
    class object refers to its own module, which refers to every name defined
    there, so descending would reach `LatentState` from any object at all and
    the check would be meaningless. What matters is whether a latent INSTANCE
    is reachable through data.
    """
    seen: dict[int, object] = {}
    stack = [root]
    while stack and len(seen) < limit:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen[id(obj)] = obj
        if isinstance(obj, (type, types.ModuleType, types.FunctionType, types.MethodType)):
            continue
        stack.extend(gc.get_referents(obj))
    return list(seen.values())


def test_agent_cannot_access_latent_state(world: World) -> None:
    """Every route to the answer key, tried and refused.

    The type boundary is the claim this whole milestone rests on. If an
    observable record can be persuaded to yield `ability_to_pay`, then the
    forecaster can be trained on ground truth by accident and the headline
    number measures nothing.
    """
    account_id = world.account_ids[0]
    at = EPOCH
    obs = world.observe(account_id, at)

    # -- type level ------------------------------------------------------
    assert isinstance(obs, ObservableState)
    assert not isinstance(obs, (LatentState, Account))
    assert World.observe.__annotations__["return"] == "ObservableState"
    assert not (LATENT_FIELD_NAMES & OBSERVABLE_FIELD_NAMES), (
        "a latent field name appeared in the observable record"
    )

    # -- route 1: attribute access --------------------------------------
    for name in sorted(LATENT_FIELD_NAMES):
        with pytest.raises(AttributeError):
            getattr(obs, name)

    # -- route 2: the instance dictionary --------------------------------
    with pytest.raises(AttributeError):
        obs.__dict__  # noqa: B018 - the point is that it raises
    with pytest.raises(TypeError):
        vars(obs)

    # -- route 3: the whole object graph ---------------------------------
    # Run before the finicky routes, because a back-reference to the account
    # is the leak a field-name check cannot see and it deserves a clear
    # failure rather than a downstream one.
    for candidate in _reachable_objects(obs):
        assert not isinstance(candidate, (LatentState, Account, World)), (
            f"{type(candidate).__name__} is reachable from ObservableState"
        )

    # -- route 4: attaching one anyway -----------------------------------
    # The exception type varies - a frozen slots dataclass can refuse through
    # either path - so the assertion is on the outcome: the write is refused
    # and the record is unchanged.
    before = pickle.dumps(obs)
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        obs.ability_to_pay = 1.0  # type: ignore[attr-defined]
    assert pickle.dumps(obs) == before

    # -- route 5: dataclass introspection --------------------------------
    field_names = {field.name for field in dataclasses.fields(obs)}
    assert not (field_names & LATENT_FIELD_NAMES)
    assert field_names == OBSERVABLE_FIELD_NAMES
    assert not any(
        isinstance(getattr(obs, name), (LatentState, Account, World)) for name in field_names
    )

    # -- route 6: pickling ------------------------------------------------
    try:
        blob = pickle.dumps(obs)
    except TypeError as unpicklable:  # pragma: no cover - only on a leak
        pytest.fail(f"ObservableState holds something unpicklable: {unpicklable}")
    for name in sorted(LATENT_FIELD_NAMES):
        assert name.encode() not in blob, f"{name} survived into the pickled bytes"
    assert b"LatentState" not in blob
    assert pickle.loads(blob) == obs

    # -- route 7: a deep copy carries nothing extra ------------------------
    for candidate in _reachable_objects(copy.deepcopy(obs)):
        assert not isinstance(candidate, (LatentState, Account, World))

    # -- route 8: dir() offers no door -------------------------------------
    assert not (set(dir(obs)) & LATENT_FIELD_NAMES)


def test_latent_state_has_no_instance_dictionary(world: World) -> None:
    """`slots=True` on the latent record is load-bearing, not cosmetic.

    Without it there is a mutable bag of attributes on every latent object,
    and nothing stops a later milestone attaching one to something observable.
    """
    latent = world._latent(world.account_ids[0])
    assert isinstance(latent, LatentState)
    with pytest.raises(AttributeError):
        latent.__dict__  # noqa: B018
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        latent.ability_to_pay = 0.0  # type: ignore[misc]


def test_observe_requires_a_time_and_never_reads_a_clock(world: World) -> None:
    """`at` is a parameter, so the view of the world is replayable."""
    with pytest.raises(TypeError):
        world.observe(world.account_ids[0])  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Gate test 2 - determinism
# ---------------------------------------------------------------------------
def test_same_seed_same_batch() -> None:
    """Three consecutive runs, byte-identical.

    A judge will ask to run it again. If the number moves, nothing that came
    before it was a measurement.
    """
    digests = []
    payloads = []
    for _ in range(3):
        built = World(seed=DEVELOP_SEED, size=SMALL)
        digests.append(built.batch_digest())
        fake = WireFake(built, SECRET)
        payloads.append(
            [(d.event_id, d.received_at, d.signature, d.body) for d in fake.emit("replay", 7)]
        )

    assert len(set(digests)) == 1, "the batch digest moved between runs"
    assert payloads[0] == payloads[1] == payloads[2], "the delivered bytes moved between runs"


def test_a_different_seed_produces_a_different_world() -> None:
    """Determinism must not be achieved by ignoring the seed."""
    first = World(seed=DEVELOP_SEED, size=SMALL).batch_digest()
    second = World(seed=DEVELOP_SEED + 1, size=SMALL).batch_digest()
    assert first != second


def test_counterfactual_is_pure(world: World) -> None:
    """Ground truth does not move when it is asked twice, and asking does not
    advance the world - otherwise the answer key would change the world it is
    the key to."""
    account_id = world.account_ids[3]
    before = world.observe(account_id, EPOCH)
    values = [
        world.counterfactual(account_id, ActionType.WHATSAPP_UTILITY, EPOCH) for _ in range(50)
    ]
    assert len(set(values)) == 1
    assert world.observe(account_id, EPOCH) == before


def test_fork_isolates_interaction_history(world: World) -> None:
    """Each arm at M11 needs the same batch and its own contact history.

    Sharing one world would let arm B's messages raise arm E's annoyance, and
    the comparison would be between arms plus contamination.
    """
    account_id = world.account_ids[5]
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)

    left = world.fork()
    right = world.fork()
    for _ in range(3):
        left.outcome(account_id, ActionType.SMS, EPOCH, generator)

    assert left.contacts_7d(account_id, EPOCH + timedelta(hours=1)) == 3
    assert right.contacts_7d(account_id, EPOCH + timedelta(hours=1)) == 0
    assert left.batch_digest() == right.batch_digest()


def test_contact_window_is_half_open(world: World) -> None:
    """`[t - 7d, t)` - an event at exactly `t` belongs to the next window."""
    account_id = world.account_ids[7]
    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)
    twin.outcome(account_id, ActionType.SMS, EPOCH, generator)

    assert twin.contacts_7d(account_id, EPOCH) == 0, "the boundary must be excluded"
    assert twin.contacts_7d(account_id, EPOCH + timedelta(microseconds=1)) == 1
    assert twin.contacts_7d(account_id, EPOCH + timedelta(days=7)) == 1
    assert twin.contacts_7d(account_id, EPOCH + timedelta(days=7, microseconds=1)) == 0


# ---------------------------------------------------------------------------
# Gate test 3 - sleeping dogs
# ---------------------------------------------------------------------------
def test_sleeping_dogs_exist(world: World) -> None:
    """Some accounts are made worse by every contact channel.

    This is what `b5 > 0` buys. Without it, "contact everyone" would be the
    optimal policy, the uplift model would have nothing to find, and the
    guardrails would be decoration rather than economics.
    """
    assert rm.B5_ANNOYANCE > 0, "the annoyance term must be positive or there are no dogs"

    dogs = sleeping_dogs(world, EPOCH)
    share = len(dogs) / len(world.account_ids)
    assert 0.02 <= share <= 0.45, f"sleeping-dog share {share:.3f} is not a population"

    # And they are genuinely negative, not merely equal.
    worst = dogs[0]
    base = world.counterfactual(worst, ActionType.DO_NOTHING, EPOCH)
    for action in sorted(rm.DIGITAL_NUDGE_ACTIONS):
        assert world.counterfactual(worst, action, EPOCH) < base

    # A silent action still helps them, which is what makes the finding
    # actionable rather than a counsel of despair.
    assert world.counterfactual(worst, ActionType.RETRY, EPOCH) > base


def test_contacting_repeatedly_destroys_value(world: World) -> None:
    """Annoyance accumulates, so the fifth message is worth less than the first."""
    account_id = world.account_ids[11]
    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)

    first = twin.counterfactual(account_id, ActionType.WHATSAPP_UTILITY, EPOCH)
    for _ in range(4):
        twin.outcome(account_id, ActionType.SMS, EPOCH - timedelta(hours=1), generator)
    later = twin.counterfactual(account_id, ActionType.WHATSAPP_UTILITY, EPOCH)

    assert later < first


def test_harm_escalates_superlinearly_with_contact_volume() -> None:
    """Opt-out and complaint hazards rise faster than contact volume.

    The guardrail metrics at M11 need this: an unconstrained arm has to be
    able to blow the complaint rate while winning on gross recovery.
    """
    one = rm.harm_hazards(
        action=ActionType.WHATSAPP_UTILITY,
        annoyance_sensitivity=0.5,
        intent_to_churn=0.2,
        contacts_7d=1,
        prior_attempts=1,
    )
    four = rm.harm_hazards(
        action=ActionType.WHATSAPP_UTILITY,
        annoyance_sensitivity=0.5,
        intent_to_churn=0.2,
        contacts_7d=4,
        prior_attempts=1,
    )
    assert four.opt_out > 4 * one.opt_out
    assert four.complaint > 4 * one.complaint


def test_silent_actions_cannot_produce_an_opt_out() -> None:
    """Nobody was contacted, so nobody can opt out. Without this the silent
    recovery path would carry a cost it does not have."""
    hazards = rm.harm_hazards(
        action=ActionType.RETRY,
        annoyance_sensitivity=1.0,
        intent_to_churn=1.0,
        contacts_7d=9,
        prior_attempts=9,
    )
    assert hazards.opt_out == 0.0


def test_all_six_outcomes_are_reachable(world: World) -> None:
    """The guardrail metrics have no source if the world only ever pays or
    goes quiet."""
    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)
    seen = set()
    for index, account_id in enumerate(world.account_ids):
        action = (ActionType.WHATSAPP_UTILITY, ActionType.VOICE_CALL, ActionType.SMS)[index % 3]
        seen.add(twin.outcome(account_id, action, EPOCH, generator).kind)
    assert seen == set(rm.OutcomeKind), f"never sampled: {set(rm.OutcomeKind) - seen}"


# ---------------------------------------------------------------------------
# Gate test 4 - the payday structure
# ---------------------------------------------------------------------------
def test_payday_effect_measurable(world: World) -> None:
    """A retry aligned to the salary credit recovers materially more.

    This is the structure a fixed T+1/T+3/T+7 calendar cannot exploit, because
    the salary day is latent and jittered per month. It IS inferable from
    `prior_payment_timestamps`, which are observable - so the gap between the
    naive arm and ARC comes from the world, not from tuning.
    """
    aligned: list[float] = []
    misaligned: list[float] = []

    for account_id in world.account_ids[:600]:
        latent = world._latent(account_id)
        for offset, bucket in ((1, aligned), (14, misaligned)):
            at = EPOCH
            for _ in range(40):  # walk to a day at the wanted salary distance
                if (
                    int(
                        days_since_salary(account_id, at, latent.salary_day, latent.salary_variance)
                    )
                    == offset
                ):
                    break
                at += timedelta(days=1)
            else:
                continue
            bucket.append(world.counterfactual(account_id, ActionType.RETRY, at))

    assert len(aligned) > 300 and len(misaligned) > 300
    lift = sum(aligned) / len(aligned) - sum(misaligned) / len(misaligned)
    assert lift > 0.08, f"payday lift {lift:.4f} is too small to be exploitable"


def test_salary_day_is_inferable_from_observables_only(world: World) -> None:
    """The learnable signal really is in the observable record.

    If it were not, the payday effect would be structure no policy could ever
    find, which is a rigged world rather than a hard one.
    """
    hits = 0
    considered = 0
    for account_id in world.account_ids[:400]:
        obs = world.observe(account_id, EPOCH)
        if len(obs.prior_payment_timestamps) < 2:
            continue
        considered += 1
        latent = world._latent(account_id)
        days = [
            days_since_salary(account_id, t, latent.salary_day, latent.salary_variance)
            for t in obs.prior_payment_timestamps
        ]
        if sum(days) / len(days) <= 4.0:
            hits += 1

    assert considered > 100
    assert hits / considered > 0.75, "payments do not cluster near the salary credit"


def test_festival_week_suppresses_ability_to_pay(world: World) -> None:
    """Injected structure the agent is never told about, and can still feel."""
    inside = FESTIVAL_START + timedelta(days=2)
    outside = FESTIVAL_END + timedelta(days=2)
    account_id = world.account_ids[2]

    # Compare at the same salary distance so the payday term is held constant.
    latent = world._latent(account_id)
    same = days_since_salary(account_id, inside, latent.salary_day, latent.salary_variance)
    at = outside
    for _ in range(40):
        if days_since_salary(account_id, at, latent.salary_day, latent.salary_variance) == same:
            break
        at += timedelta(days=1)

    assert world.counterfactual(account_id, ActionType.RETRY, inside) < world.counterfactual(
        account_id, ActionType.RETRY, at
    )


# ---------------------------------------------------------------------------
# Gate test 5 - the outages
# ---------------------------------------------------------------------------
def test_outage_injected_and_detectable(big_world: World) -> None:
    """The outage is findable from the event stream alone.

    Nothing labels it. What the Sentinel gets at M6 is a burst of correlated
    declines against a denominator, and the whole demo beat - "the naive
    system just messaged all of these" - depends on that being genuinely
    discoverable rather than announced.
    """
    outage = OUTAGES[0]
    assert outage.end - outage.start == timedelta(hours=2)
    assert OUTAGES[1].end - OUTAGES[1].start == timedelta(minutes=40)
    assert {o.issuer_id for o in OUTAGES} == {"ISS_LP02", "ISS_PS01"}

    presentations = [
        event
        for event in big_world.batch_events()
        if event.kind is EventKind.PRESENTATION and event.rail is not Rail.INVOICE
    ]

    # The issuer comes from the observable record - the same field the
    # Sentinel will key its cohorts on - looked up once per account rather
    # than once per event.
    issuer_of = {
        account_id: big_world.observe(account_id, EPOCH).issuer_id
        for account_id in big_world.account_ids
    }

    def decline_rate(issuer_id: str, inside: bool) -> tuple[float, int]:
        rows = [
            event
            for event in presentations
            if issuer_of[event.account_id] == issuer_id and outage.covers(event.at) is inside
        ]
        if not rows:
            return 0.0, 0
        return sum(1 for row in rows if not row.succeeded) / len(rows), len(rows)

    hit_rate, hit_n = decline_rate(outage.issuer_id, True)
    base_rate, base_n = decline_rate(outage.issuer_id, False)
    peer_rate, peer_n = decline_rate("ISS_LP01", True)

    assert hit_n >= 30, f"only {hit_n} presentations inside the window - too thin to test"
    assert base_n > 100 and peer_n > 5
    assert hit_rate > 0.70, f"outage decline rate {hit_rate:.3f} is not an outage"
    assert hit_rate > base_rate + 0.35, "the outage is not separable from the baseline"
    assert hit_rate > peer_rate + 0.35, "the outage is not separable from its peers"


def test_outage_is_invisible_in_observable_state(big_world: World) -> None:
    """No field announces it. It has to be inferred from correlated declines."""
    account_id = next(
        event.account_id
        for event in big_world.batch_events()
        if OUTAGES[0].covers(event.at)
        and big_world.observe(event.account_id, EPOCH).issuer_id == OUTAGES[0].issuer_id
    )
    obs = big_world.observe(account_id, OUTAGES[0].end)
    rendered = repr(obs).lower()
    for word in ("outage", "degraded", "health", "incident", "issuer_health"):
        assert word not in rendered


def test_issuer_health_collapses_only_inside_the_window() -> None:
    """Half-open: the instant the outage ends, health is back."""
    outage = OUTAGES[0]
    assert issuer_health(outage.issuer_id, outage.start) == outage.residual_health
    assert issuer_health(outage.issuer_id, outage.end - timedelta(seconds=1)) == (
        outage.residual_health
    )
    assert issuer_health(outage.issuer_id, outage.end) > 0.8
    assert issuer_health(outage.issuer_id, outage.start - timedelta(seconds=1)) > 0.8
    # Another issuer is untouched at the same instant.
    assert issuer_health("ISS_LP01", outage.start) > 0.8


def test_thin_issuers_exist_so_cohort_power_can_run_out(big_world: World) -> None:
    """A long tail with almost no volume is what makes INSUFFICIENT_POWER real.

    If every issuer had thousands of transactions an hour, the Sentinel's
    hierarchical back-off would be decoration.
    """
    issuer_of = {
        account_id: big_world.observe(account_id, EPOCH).issuer_id
        for account_id in big_world.account_ids
    }
    counts: dict[str, int] = {}
    for event in big_world.batch_events():
        issuer = issuer_of[event.account_id]
        counts[issuer] = counts.get(issuer, 0) + 1

    ordered = sorted(counts.values())
    assert ordered[0] * 20 < ordered[-1], "issuer volume is too flat to strain cohort power"
    assert min(counts.values()) < 250


# ---------------------------------------------------------------------------
# Gate tests 6 and 7 - the wire
# ---------------------------------------------------------------------------
def test_wire_fake_payloads_hmac_valid(world: World) -> None:
    """Every delivery carries a signature over its exact bytes.

    The adapter at M5 verifies before it parses, because an unverified webhook
    is attacker-controlled input to a money-moving system. This proves the
    fake gives it something real to verify.
    """
    fake = WireFake(world, SECRET)
    deliveries = fake.emit("replay", DEVELOP_SEED)
    assert len(deliveries) > 500

    assert all(verify(SECRET, delivery) for delivery in deliveries)

    # A different key does not verify.
    assert not any(verify(b"wrong-key-wrong-key-wrong-key123", d) for d in deliveries)

    # A tampered body does not verify - the signature covers the bytes, so
    # changing an amount after signing is detectable.
    victim = deliveries[0]
    tampered = dataclasses.replace(victim, body=victim.body.replace(b'"amount"', b'"amounx"'))
    assert not verify(SECRET, tampered)

    # A replayed signature against a different timestamp does not verify.
    moved = dataclasses.replace(
        victim, event_timestamp=victim.event_timestamp + timedelta(seconds=1)
    )
    assert not verify(SECRET, moved)

    assert sign(SECRET, victim.event_timestamp, victim.body).startswith("t=")


def test_wire_fake_speaks_three_dialects(world: World) -> None:
    """The same fact looks different on every rail, which is why L1 exists."""
    deliveries = WireFake(world, SECRET).emit("replay", DEVELOP_SEED)
    by_source: dict[str, dict] = {}
    for delivery in deliveries:
        by_source.setdefault(delivery.source, delivery.payload())

    assert {"pgw", "npci_nach", "upi_autopay", "billing"} <= set(by_source)
    assert "payload" in by_source["pgw"]
    assert "umrn" in by_source["npci_nach"]
    assert "npci_error_code" in by_source["upi_autopay"]
    assert "invoice" in by_source["billing"]

    # Money crosses the wire as an integer or as a decimal string built from
    # integer paise. A float amount anywhere would be a GI-2 breach at the
    # boundary, which is the easiest place to leak one.
    assert isinstance(by_source["pgw"]["payload"]["payment"]["amount"], int)
    assert isinstance(by_source["npci_nach"]["amount"], str)
    assert b"e-05" not in deliveries[0].body


def test_wire_payloads_carry_pii_for_the_redaction_boundary(world: World) -> None:
    """A bank narration with a real name in it, which must never reach the
    ledger. If the fake did not produce one, M5's redaction boundary and M2's
    write-guard would be tested against nothing."""
    deliveries = WireFake(world, SECRET).emit("replay", DEVELOP_SEED)
    nach = next(d.payload() for d in deliveries if d.source == "npci_nach")

    assert nach["customer"]["name"]
    assert nach["customer"]["mobile"].startswith("+91")
    assert nach["customer"]["name"].upper() in nach["narration"]
    assert len(nach["customer"]["ifsc"]) == 11


def test_duplicate_and_out_of_order_present(world: World) -> None:
    """The two delivery pathologies the adapter must survive, at their rates.

    Both counts are exact rather than approximate, because the fake selects a
    fixed number rather than flipping a coin per event - so a broken injector
    fails the test instead of hiding inside a tolerance band.
    """
    events = world.batch_events()
    deliveries = WireFake(world, SECRET).emit("replay", DEVELOP_SEED)

    seen: dict[str, int] = {}
    for delivery in deliveries:
        seen[delivery.event_id] = seen.get(delivery.event_id, 0) + 1
    repeated = [event_id for event_id, count in seen.items() if count > 1]

    assert len(repeated) == round(DUPLICATE_RATE * len(events))
    assert len(deliveries) == len(events) + len(repeated)

    # A redelivery is byte-identical, including the signature. An adapter that
    # dedupes on a hash of the body would otherwise pass by accident.
    for event_id in repeated[:20]:
        copies = [d for d in deliveries if d.event_id == event_id]
        assert len({(d.body, d.signature) for d in copies}) == 1
        assert {d.delivery for d in copies} == {1, 2}

    # Counted by distinct event, not by delivery: a late event that is also
    # redelivered arrives late twice, and that is one late event, not two.
    late = {delivery.event_id for delivery in late_deliveries(deliveries)}
    assert len(late) == round(OUT_OF_ORDER_RATE * len(events))
    assert arrival_inversions(deliveries) > 0, "nothing actually arrived out of order"

    # Arrival order really does disagree with event order, which is why L0
    # must sort by event time rather than by arrival.
    arrival = [d.event_timestamp for d in deliveries]
    assert arrival != sorted(arrival)


def test_live_mode_requires_an_injected_clock(world: World) -> None:
    """Nothing in the simulator reads a clock, so live mode is handed one."""
    fake = WireFake(world, SECRET)
    with pytest.raises(ValueError, match="never reads a clock"):
        fake.emit("live")
    with pytest.raises(ValueError, match="unknown mode"):
        fake.emit("shadow")  # type: ignore[arg-type]

    now = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)
    live = fake.emit("live", DEVELOP_SEED, now=now)
    assert max(d.event_timestamp for d in live) == now
    assert all(verify(SECRET, delivery) for delivery in live)


# ---------------------------------------------------------------------------
# The injected faults, and the silent repairs that answer them
# ---------------------------------------------------------------------------
def test_orphaned_mandate_cohort_exists_and_is_silent(world: World) -> None:
    """About 3% orphan, and the merchant's own record still says active.

    That is what makes it silent, and why the Sentinel has to infer it from a
    reissue date, a registration date and an unbroken run of failures rather
    than reading a status field.
    """
    account_ids = [
        account_id for account_id in world.account_ids if _is_orphaned(world, account_id)
    ]
    share = len(account_ids) / len(world.account_ids)
    assert 0.015 <= share <= 0.05, f"orphan cohort share {share:.4f}"

    for account_id in account_ids[:10]:
        obs = world.observe(account_id, EPOCH)
        assert obs.mandate_status == "active", "the record must keep lying"
        assert obs.instrument_reissued_at is not None
        assert obs.mandate_registered_at is not None
        assert obs.instrument_reissued_at > obs.mandate_registered_at
        assert obs.rail in (Rail.ENACH, Rail.UPI_AUTOPAY)


def _is_orphaned(world: World, account_id: str) -> bool:
    """Ground truth, read the way only a test may: a retry cannot present."""
    return (
        world.counterfactual(account_id, ActionType.RETRY, EPOCH) == 0.0
        and world.counterfactual(account_id, ActionType.MANDATE_RE_REGISTER, EPOCH) > 0.0
    )


def test_merchant_layer_failure_is_repaired_without_contact(world: World) -> None:
    """The SELF_HEALING path: money recovered with zero customer contact.

    A retry is worthless on an orphaned mandate and re-registration fixes it,
    which is the whole claim behind that state in the FSM.
    """
    account_id = next(a for a in world.account_ids if _is_orphaned(world, a))
    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)

    blocked = twin.outcome(account_id, ActionType.RETRY, EPOCH, generator)
    assert blocked.kind is rm.OutcomeKind.NO_RESPONSE
    assert blocked.true_semantic is Semantic.MANDATE_MISSING
    assert blocked.decline_code is not None

    twin.outcome(account_id, ActionType.MANDATE_RE_REGISTER, EPOCH, generator)
    assert twin.counterfactual(account_id, ActionType.RETRY, EPOCH) > 0.0
    assert twin.contacts_7d(account_id, EPOCH + timedelta(hours=1)) == 0, (
        "the repair must not have contacted anybody"
    )


def test_a_terminal_instrument_is_never_repairable(world: World) -> None:
    """Nothing fixes a closed or stolen card, which is why retrying one is
    network-punished and blocked permanently rather than merely discouraged."""
    account_id = next(
        (
            a
            for a in world.account_ids
            if world.counterfactual(a, ActionType.RETRY, EPOCH) == 0.0
            and world.counterfactual(a, ActionType.MANDATE_RE_REGISTER, EPOCH) == 0.0
            and world.counterfactual(a, ActionType.CARD_UPDATER, EPOCH) == 0.0
        ),
        None,
    )
    assert account_id is not None, "no terminal instrument in the population"

    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)
    for action in (
        ActionType.RETRY,
        ActionType.CARD_UPDATER,
        ActionType.MANDATE_RE_REGISTER,
        ActionType.RAIL_FALLBACK,
    ):
        outcome = twin.outcome(account_id, action, EPOCH, generator)
        assert outcome.kind is not rm.OutcomeKind.PAID
        assert twin.counterfactual(account_id, ActionType.RETRY, EPOCH) == 0.0


def test_stale_numbers_produce_wrong_party_contacts(world: World) -> None:
    """3% of numbers reach somebody else. Nothing is recovered and nothing may
    be disclosed, which is what the wrong-party stopping rule is for."""
    stale = [a for a in world.account_ids if world._latent(a).phone_stale]
    share = len(stale) / len(world.account_ids)
    assert 0.02 <= share <= 0.045

    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.OUTCOME)
    outcome = twin.outcome(stale[0], ActionType.VOICE_CALL, EPOCH, generator)
    assert outcome.wrong_party is True
    assert outcome.kind is not rm.OutcomeKind.PAID
    assert outcome.promise is None, "a wrong party cannot promise anything"


# ---------------------------------------------------------------------------
# Promises, censoring, and the code vocabulary
# ---------------------------------------------------------------------------
def test_promises_can_be_unresolved(world: World) -> None:
    """A promise dated the 20th is neither kept nor broken on the 18th.

    Coding it broken is what biases a promise model pessimistic, so the world
    refuses to and M7 gets genuinely censored labels to handle.
    """
    generator = rng(DEVELOP_SEED, Stream.PROMISE)
    account_id = world.account_ids[0]
    promise = Promise(made_at=EPOCH, due_at=EPOCH + timedelta(days=6), amount_paise=120000)

    assert world.resolve_promise(account_id, promise, EPOCH, generator) is (
        PromiseStatus.UNRESOLVED
    )
    assert (
        world.resolve_promise(account_id, promise, promise.due_at - timedelta(seconds=1), generator)
        is PromiseStatus.UNRESOLVED
    )

    settled = {
        world.resolve_promise(account_id, promise, promise.due_at, generator) for _ in range(60)
    }
    assert settled <= {PromiseStatus.KEPT, PromiseStatus.BROKEN}
    assert PromiseStatus.UNRESOLVED not in settled


def test_prior_promise_records_include_unresolved(world: World) -> None:
    """The observable history carries censored promises too, so a model
    trained on it has to deal with them rather than being handed clean labels."""
    outcomes = [
        outcome
        for account_id in world.account_ids
        for outcome in world.observe(account_id, EPOCH).prior_ptp_outcomes
    ]
    assert outcomes
    assert str(PromiseStatus.UNRESOLVED) in outcomes
    assert str(PromiseStatus.KEPT) in outcomes
    assert str(PromiseStatus.BROKEN) in outcomes


def test_a_conversation_is_needed_to_elicit_a_promise(world: World) -> None:
    """An SMS cannot produce a promise-to-pay. Only a call can."""
    twin = world.fork()
    generator = rng(DEVELOP_SEED, Stream.PROMISE)
    for account_id in world.account_ids[:400]:
        outcome = twin.outcome(account_id, ActionType.SMS, EPOCH, generator)
        assert outcome.promise is None

    voice = world.fork()
    promises = [
        voice.outcome(account_id, ActionType.VOICE_CALL, EPOCH, generator).promise
        for account_id in world.account_ids[:400]
    ]
    assert any(promise is not None for promise in promises)


def test_five_percent_of_codes_are_wrong(world: World) -> None:
    """The gateway lies sometimes, and the Sentinel's code map has to survive it."""
    assert REMAP_RATE == 0.05
    declines = [
        event
        for event in world.batch_events()
        if not event.succeeded and event.true_semantic and event.decline_code
    ]
    codeable = [e for e in declines if code_for(e.rail, e.true_semantic) is not None]
    wrong = [e for e in codeable if e.decline_code != code_for(e.rail, e.true_semantic).code]

    assert len(codeable) > 300
    share = len(wrong) / len(codeable)
    assert 0.03 <= share <= 0.07, f"remap share {share:.4f} is not the injected 5%"


def test_every_failure_reason_has_a_code_on_every_rail() -> None:
    """A decline with no code at all is not something a gateway sends.

    Where a rail's published vocabulary has no distinct code, the collapse is
    declared in the table rather than leaving a hole for a caller to trip on.
    """
    for rail in (Rail.CARD, Rail.ENACH, Rail.UPI_AUTOPAY):
        for semantic in Semantic:
            assert code_for(rail, semantic) is not None, f"{rail} has no code for {semantic}"
    # An invoice does not decline, so it carries no codes at all.
    assert all(code_for(Rail.INVOICE, semantic) is None for semantic in Semantic)


def test_all_four_claim_types_are_generated(world: World) -> None:
    """Four leak surfaces, one claim object. M5 needs all four to normalise."""
    produced = {event.claim_type for event in world.batch_events()}
    assert produced == set(ClaimType)


def test_gateway_retries_appear_in_the_batch(world: World) -> None:
    """The gateway re-presents on its own schedule, and those attempts count
    against the network cap whether or not ARC issued them."""
    gateway = [
        event
        for event in world.batch_events()
        if event.attempt == 2 and str(event.initiated_by) == "gateway"
    ]
    assert len(gateway) > 20
    assert all(event.at < world.epoch for event in gateway)


# ---------------------------------------------------------------------------
# The response model itself, and the freeze discipline
# ---------------------------------------------------------------------------
def _inputs(**overrides: float) -> rm.ResponseInputs:
    base = {
        "ability_to_pay": 0.55,
        "responsiveness": 0.45,
        "timing_fit": 0.25,
        "issuer_health": 0.95,
        "annoyance_sensitivity": 0.40,
        "contacts_7d": 1,
        "friction": 0.35,
        "affordability": 0.70,
    }
    base.update(overrides)
    return rm.ResponseInputs(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "low", "high", "expect_increase"),
    [
        ("ability_to_pay", 0.1, 0.9, True),
        ("responsiveness", 0.1, 0.9, True),
        ("timing_fit", 0.0, 1.0, True),
        ("issuer_health", 0.05, 0.99, True),
        ("affordability", 0.1, 0.9, True),
        ("annoyance_sensitivity", 0.1, 0.9, False),
        ("contacts_7d", 1, 5, False),
        ("friction", 0.05, 0.9, False),
    ],
)
def test_each_term_pushes_the_right_way(
    field: str, low: float, high: float, expect_increase: bool
) -> None:
    """Every one of the seven terms carries the sign it is specified with.

    A sign error here would be invisible - the model would still produce
    plausible probabilities - and it would invert the policy's incentives.
    """
    lower = rm.p_pay(_inputs(**{field: low}))
    upper = rm.p_pay(_inputs(**{field: high}))
    assert (upper > lower) is expect_increase, f"{field} moves the wrong way"


def test_annoyance_term_is_strictly_positive() -> None:
    """b5 > 0 is what makes sleeping dogs possible at all."""
    assert rm.B5_ANNOYANCE > 0


def test_probabilities_stay_in_range_at_the_extremes() -> None:
    """A saturated logit must not overflow into a NaN and silently poison an
    expected-value product downstream."""
    for value in (-500.0, 500.0):
        assert 0.0 <= rm.sigmoid(value) <= 1.0
    extreme = rm.p_pay(rm.ResponseInputs(1.0, 1.0, 1.0, 1.0, 0.0, 0, 0.0, 1.0))
    hopeless = rm.p_pay(rm.ResponseInputs(0.0, 0.0, 0.0, 0.0, 1.0, 200, 1.0, 0.0))
    assert 0.0 < hopeless < extreme < 1.0


def test_do_nothing_is_always_scored(world: World) -> None:
    """The null action has a ground-truth probability like any other, so the
    allocator's baseline is a measurement rather than an assumption."""
    for account_id in world.account_ids[:50]:
        value = world.counterfactual(account_id, ActionType.DO_NOTHING, EPOCH)
        assert 0.0 <= value <= 1.0


CALIBRATION_FILES = ("response_model.py", "world.py")

# Constants that describe structure rather than calibrate behaviour. They have
# no published figure behind them because there is nothing to publish.
STRUCTURAL_CONSTANTS = frozenset(
    {
        "IST",
        "DEFAULT_POPULATION",
        "LAST_WORKING_DAY",
        "SECRET",
        "CODES",
        "ISSUER_BY_ID",
        "LATENT_FIELD_NAMES",
        "OBSERVABLE_FIELD_NAMES",
        "ACTION_CHANNEL",
        "CONTACT_CHANNELS",
        "CONTACT_ACTIONS",
        "DEBIT_ACTIONS",
        "PROMISE_CHANNELS",
        "DIGITAL_NUDGE_ACTIONS",
        "ADVERSE_OUTCOMES",
        "OUTAGES",
        "ISSUERS",
        "FESTIVAL_START",
        "FESTIVAL_END",
        "SALARY_DAYS",
        "_CLAIM_TYPE_BY_RAIL",
        "_FIRST_NAMES",
        "_LAST_NAMES",
        "_CONSENT_CHANNELS",
        "_INVOICE_BUCKETS",
        "_REPAIR_ACTIONS",
        "_PRESENT_RAIL_OFFSET",
        "_MANDATE_RAIL_SHARE",
        "_ORPHAN_GIVEN_REISSUE",
        "_AGEING_DAYS",
    }
)


def _sourced_constants(path) -> list[str]:
    """Module-level constants with no `# source:` above them.

    A constant may inherit the comment block of the group it sits in, since
    three constants describing one curve share one citation. The walk stops at
    a blank line, so a group has to be written as a group.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    unsourced: list[str] = []
    for index, line in enumerate(lines):
        name = line.split(":")[0].split("=")[0].strip()
        if not line or line[0].isspace() or "=" not in line:
            continue
        if not name.replace("_", "").isupper() or not name:
            continue
        if name in STRUCTURAL_CONSTANTS:
            continue

        comments: list[str] = []
        cursor = index - 1
        while cursor >= 0:
            above = lines[cursor]
            if above.startswith("#"):
                comments.append(above)
            elif above and not above[0].isspace() and "=" in above:
                pass  # another constant in the same group
            elif above.startswith(")") or above.startswith("]"):
                pass  # the tail of the previous constant's literal
            else:
                break
            cursor -= 1

        if not any("source:" in comment for comment in comments):
            unsourced.append(f"{path.name}:{index + 1} {name}")
    return unsourced


def test_every_calibration_constant_names_its_source() -> None:
    """The world is defensible only if its numbers can be traced somewhere.

    These constants freeze at `simulator-frozen-v1`. A constant with no stated
    origin is one nobody can check, and an unchecked constant in a frozen file
    is exactly what the circularity attack is looking for.
    """
    from pathlib import Path

    package = Path(rm.__file__).parent
    unsourced = [
        entry for name in CALIBRATION_FILES for entry in _sourced_constants(package / name)
    ]
    assert not unsourced, "calibration constants with no `# source:`:\n" + "\n".join(
        f"  {entry}" for entry in unsourced
    )


def test_no_constant_is_marked_provisional() -> None:
    """The constants freeze at the tag. A TODO in a frozen file is a promise
    nobody will keep, and a reviewer is entitled to read the absence of one as
    a claim that the number is final."""
    from pathlib import Path

    package = Path(rm.__file__).parent
    for name in CALIBRATION_FILES:
        text = (package / name).read_text(encoding="utf-8")
        for marker in ("TODO", "FIXME", "XXX", "provisional", "placeholder", "tune later"):
            assert marker not in text, f"{name} contains {marker!r}"


def test_validate_command_passes(capsys: pytest.CaptureFixture[str]) -> None:
    """`python -m arc.simulator.validate` is the simulator's own evidence.

    It prints the generated distribution beside the published figure it was
    calibrated against and exits non-zero on drift, so the world is checkable
    against something outside itself rather than only against its own tests.
    """
    from arc.simulator.validate import main

    exit_code = main(["--seed", str(DEVELOP_SEED), "--size", "6000"])
    printed = capsys.readouterr().out

    assert exit_code == 0, printed
    assert "DISTRIBUTIONS vs PUBLISHED BENCHMARK" in printed
    assert "INJECTED STRUCTURE vs SPECIFICATION" in printed
    assert "SOURCES" in printed
    assert "DRIFT" not in printed
    # Every published row states where its number came from.
    assert printed.count("NPCI") >= 2


def test_validate_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """The report is only worth reading if a wrong world fails it.

    Proven by moving one metric, not by trusting that a green run means the
    check is live.
    """
    from arc.simulator import validate

    real = validate.measure

    def drifted(seed: int, size: int) -> dict[str, float]:
        measured = dict(real(seed, size))
        measured["nach_decline_rate"] = 0.95
        return measured

    monkeypatch.setattr(validate, "measure", drifted)
    text, passed = validate.report(DEVELOP_SEED, 2000)
    assert not passed
    assert "DRIFT" in text


def test_batch_covers_the_declared_window(world: World) -> None:
    """Every event lands inside `[EPOCH - 14d, EPOCH)`, half-open like the rest."""
    events = world.batch_events()
    assert events
    assert all(event.at < world.epoch for event in events)
    assert min(event.at for event in events) >= BATCH_START - timedelta(hours=6)
    assert list(events) == sorted(events, key=lambda e: (e.at, e.event_id))
