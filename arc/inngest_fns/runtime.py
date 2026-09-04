"""Inngest's step semantics, over Postgres, so they can be tested.

    Postgres owns state. Inngest owns time. Neither owns the other's job.

This module is the local EXECUTION of Inngest's contract: memoised steps,
`sleep_until`, `wait_for_event`, and `cancelOn` evaluated at step boundaries.
It is not a reimplementation of Inngest and does not try to be - it is the
semantics a durable function is written against, executed deterministically so
the acceptance gate can drive a three-day sleep in a millisecond and assert
what happens when a hardship signal lands in the middle of it.

WHY THE STEP RESULT IS MEMOISED IN A TABLE. A durable function is REPLAYED
from the top on every resumption, not resumed mid-line. Every step that already
completed must return its recorded result rather than executing again. Without
that, waking from a sleep would re-run `gatedEnqueue`, issue a second
certificate for the same wake, and put a second row in the outbox under a
different key. The primary key on (run_id, step_id) is the whole guarantee.

CANCELLATION IS EVALUATED AT STEP BOUNDARIES, WHICH IS WHERE IT MATTERS. A
sleeping run is, by definition, between steps. So a cancelOn event that lands
during a sleep is seen the instant the sleep is asked to end - and the run
raises rather than proceeding. That is why nothing polls: the check happens
where the run is, not on a timer somewhere else.

THE EVENT LOG AND THE RUN ROW LIVE IN `arc.events`, NOT HERE. This module
owns step semantics and nothing else. That split is what lets the Conductor's
erasure sweep cancel a subject's runs without importing a durable-function
package, which is the cycle that used to hold only by luck.

NOTHING HERE READS A CLOCK. `Clock` is injected, every step takes the moment it
is evaluated at, and the AST test that enforces this repo-wide covers this file
like any other. A durable function that read a wall clock could not be
replayed, which would make the whole memoisation pointless.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from arc.core.time_authority import ensure_utc
from arc.events.bus import ObservedEvent, Subscription, events_since
from arc.events.names import EventName
from arc.events.runs import DurableRun


class RunCancelled(Exception):
    """A cancelOn event matched. The run stops where it is.

    Carries the event that did it, because a run cancelled by hardship and one
    cancelled by a system freeze are different facts and only one of them is
    about the person.
    """

    def __init__(self, event: EventName, detail: str = "") -> None:
        super().__init__(f"run cancelled by {event.value}{': ' + detail if detail else ''}")
        self.event = event
        self.detail = detail


class StepFailed(RuntimeError):
    """A step raised. The run is marked failed and does not silently continue."""


class Clock(Protocol):
    """Time as a dependency. There is no other way to get it in this package."""

    def now(self) -> datetime: ...


@dataclass
class ManualClock:
    """A clock the test moves. `sleep_until` advances it; nothing else does.

    This is what turns a three-day `step.sleepUntil` into an assertion that
    runs in microseconds without pretending the sleep did not happen: the run
    genuinely resumes at the later moment, and every rule the Gate evaluates on
    wake sees that later moment.
    """

    moment: datetime

    def __post_init__(self) -> None:
        ensure_utc(self.moment)

    def now(self) -> datetime:
        return self.moment

    def advance_to(self, when: datetime) -> None:
        ensure_utc(when)
        if when > self.moment:
            self.moment = when


class Step:
    """The step API a durable function is written against.

    Deliberately small. `run`, `sleep_until`, `wait_for_event` - and nothing
    that reaches the world, because the only path to an effect in this package
    is `gated_enqueue`, which is itself written in terms of `run`.
    """

    def __init__(self, conn: Any, run: DurableRun, clock: Clock) -> None:
        self._conn = conn
        self._run = run
        self._clock = clock
        self.slept_until: list[datetime] = []
        self.replayed: list[str] = []

    @property
    def run(self) -> DurableRun:
        return self._run

    def now(self) -> datetime:
        return self._clock.now()

    async def _check_cancellation(self) -> None:
        """Evaluated at every step boundary. This is where cancelOn lives."""
        observed = await events_since(
            self._conn, self._run.subscriptions(), since=self._run.started_at
        )
        if observed is None:
            return
        await self._conn.execute(
            """
            UPDATE durable_runs
               SET status = 'cancelled', cancelled_by = $2, finished_at = $3,
                   outcome = 'CANCELLED'
             WHERE run_id = $1 AND status NOT IN ('completed', 'cancelled')
            """,
            self._run.run_id,
            observed.name.value,
            observed.occurred_at,
        )
        raise RunCancelled(observed.name, f"matched {observed.match_key}")

    async def run_step(self, step_id: str, fn: Callable[[], Awaitable[Any]]) -> Any:
        """Execute once, ever. A completed step is read back, not re-run.

        The cancellation check happens BEFORE the memo lookup as well as before
        execution, so a replay of an already-cancelled run stops rather than
        walking forward through steps it recorded earlier.
        """
        await self._check_cancellation()

        recorded = await self._conn.fetchval(
            "SELECT result FROM durable_steps WHERE run_id = $1 AND step_id = $2",
            self._run.run_id,
            step_id,
        )
        if recorded is not None:
            self.replayed.append(step_id)
            payload = json.loads(recorded) if isinstance(recorded, str) else recorded
            return payload.get("value")

        try:
            value = await fn()
        except RunCancelled:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
            await self._conn.execute(
                "UPDATE durable_runs SET status = 'failed', finished_at = $2, outcome = $3 "
                "WHERE run_id = $1",
                self._run.run_id,
                self._clock.now(),
                type(exc).__name__,
            )
            raise StepFailed(f"{self._run.function_id}/{step_id}: {exc}") from exc

        await self._conn.execute(
            """
            INSERT INTO durable_steps (run_id, step_id, completed_at, result)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (run_id, step_id) DO NOTHING
            """,
            self._run.run_id,
            step_id,
            self._clock.now(),
            json.dumps({"value": value}, default=str),
        )
        return value

    async def sleep_until(self, step_id: str, when: datetime) -> None:
        """Suspend until `when`. Cancellation is checked on both sides.

        Checked BEFORE sleeping so a run that is already cancelled does not lie
        down, and AFTER so a signal that arrived during the sleep stops it at
        the moment it wakes. The second check is the one that makes
        `test_hardship_event_cancels_mid_sleep` pass without anything polling.
        """
        ensure_utc(when)
        await self._check_cancellation()

        already = await self._conn.fetchval(
            "SELECT 1 FROM durable_steps WHERE run_id = $1 AND step_id = $2",
            self._run.run_id,
            step_id,
        )
        if already is None:
            await self._conn.execute(
                "UPDATE durable_runs SET status = 'sleeping' WHERE run_id = $1",
                self._run.run_id,
            )
            if isinstance(self._clock, ManualClock):
                self._clock.advance_to(when)
            self.slept_until.append(when)
            await self._conn.execute(
                """
                INSERT INTO durable_steps (run_id, step_id, completed_at, result)
                VALUES ($1, $2, $3, $4::jsonb)
                ON CONFLICT (run_id, step_id) DO NOTHING
                """,
                self._run.run_id,
                step_id,
                when,
                json.dumps({"slept_until": when.isoformat()}),
            )
        else:
            # A REPLAY MUST RESUME AT THE MOMENT THE SLEEP ENDED, not at the
            # moment the replay began. Inngest re-executes a function from the
            # top, so without this the clock would still read the original
            # start time, every step id derived from `now()` would differ from
            # the one recorded, and the memo would miss - producing a second
            # certificate and a second outbox row for one wake.
            self.replayed.append(step_id)
            if isinstance(self._clock, ManualClock):
                self._clock.advance_to(when)

        await self._conn.execute(
            "UPDATE durable_runs SET status = 'running' WHERE run_id = $1 AND status = 'sleeping'",
            self._run.run_id,
        )
        await self._check_cancellation()

    async def wait_for_event(
        self,
        step_id: str,
        event: EventName,
        *,
        match: str,
        timeout_at: datetime,
    ) -> ObservedEvent | None:
        """Wait for one event, or give up at `timeout_at`.

        Returns None on timeout. A timeout is a real answer - the promise-to-pay
        tracker needs to tell "they paid" from "the date passed and they did
        not", and coercing the second into an exception would lose it.
        """
        ensure_utc(timeout_at)
        await self._check_cancellation()

        observed = await events_since(
            self._conn, [Subscription(event=event, match=match)], since=self._run.started_at
        )
        if observed is not None and observed.occurred_at <= timeout_at:
            await self._conn.execute(
                "UPDATE durable_runs SET status = 'running' WHERE run_id = $1",
                self._run.run_id,
            )
            if isinstance(self._clock, ManualClock):
                self._clock.advance_to(observed.occurred_at)
            return observed

        await self._conn.execute(
            "UPDATE durable_runs SET status = 'waiting' WHERE run_id = $1", self._run.run_id
        )
        if isinstance(self._clock, ManualClock):
            self._clock.advance_to(timeout_at)
        self.slept_until.append(timeout_at)
        await self._check_cancellation()
        return None
