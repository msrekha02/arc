"""The fake provider: idempotent, failure-injecting, and fully observable.

WHY THE FAILURE SELECTION IS HASHED AND NOT SAMPLED. The obvious way to inject
a five percent error rate is to draw from a seeded generator on each call.
Under M9's twenty concurrent workers that is not reproducible: which key gets
the failure depends on the order the workers happen to reach the provider, so
the same seed produces a different run every time and a flaky test looks like a
concurrency bug. Deriving the draw from the idempotency key instead makes the
outcome a property of the MESSAGE rather than of the schedule - the same key
fails the same way regardless of which worker picks it up, or how many do.

That also means the provider needs no lock for its decisions, only for its
records.

EVERY INVOCATION IS RECORDED, and M9's duplicate detection depends on it. If
two workers dispatch one row, the same key appears in `invocations` twice, and
that is visible even though `effects` would still show one - because this
provider honours the key and absorbs the second call. Counting only effects
would hide the Conductor's failure behind the provider's competence.

TRANSIENT FAILURES ARE PER-KEY AND COUNTED. A key selected for transient
failure fails its first N attempts and then succeeds, so a retry path can
actually be exercised. Failing on a pure hash of the key would fail forever and
every such row would reach the dead-letter queue instead.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from arc.channels.contracts import ChannelOutcome
from arc.conductor.worker import PermanentError, RetryableError


@dataclass(frozen=True)
class ProviderResponse:
    """What a provider says came back. Its vocabulary, not ours."""

    status: str
    reference: str
    deduplicated: bool = False
    detail: str = ""


@runtime_checkable
class Provider(Protocol):
    async def deliver(
        self, channel: str, payload: Mapping[str, Any], idempotency_key: str
    ) -> ProviderResponse: ...


def _band(idempotency_key: str, salt: str) -> float:
    """A stable draw in [0, 1) from the key. Order-independent by construction."""
    digest = hashlib.sha256(f"{salt}\x1f{idempotency_key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# The provider's own status vocabulary. Deliberately not ours: mapping between
# the two is the channel's whole job, and sharing one enum would hide the seam
# where a real vendor's states stop lining up with the domain's.
PROVIDER_STATUSES: tuple[str, ...] = (
    "accepted",
    "opened",
    "responded",
    "undeliverable",
    "reached_third_party",
    "recipient_unsubscribed",
    "carrier_rejected",
)

# How a healthy channel's traffic distributes, in cumulative bands. The tail is
# what M11's guardrails read, so it has to exist rather than being an
# afterthought: a provider that only ever returns success gives the opt-out
# rate no source and the metric silently reads zero forever.
#
# source: messaging delivery-report distributions - the large majority
# delivered, a meaningful read share on rich channels, replies and hard
# failures in the low single digits, and unsubscribes below one percent.
DEFAULT_MIX: tuple[tuple[str, float], ...] = (
    ("accepted", 0.62),
    ("opened", 0.86),
    ("responded", 0.91),
    ("undeliverable", 0.955),
    ("reached_third_party", 0.972),
    ("recipient_unsubscribed", 0.982),
    ("carrier_rejected", 1.0),
)


@dataclass
class FakeProvider:
    """A provider that behaves like a real one, reproducibly.

    `transient_failure_rate` exercises M9's retry path; `permanent_failure_rate`
    exercises its dead-letter path. Both default to zero, so a test that wants
    failures has to ask for them and a test that does not is never surprised.
    """

    seed: str = "arc"
    transient_failure_rate: float = 0.0
    transient_attempts: int = 1
    permanent_failure_rate: float = 0.0
    mix: tuple[tuple[str, float], ...] = DEFAULT_MIX

    invocations: list[str] = field(default_factory=list)
    effects: dict[str, ProviderResponse] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def deliver(
        self, channel: str, payload: Mapping[str, Any], idempotency_key: str
    ) -> ProviderResponse:
        async with self._lock:
            self.invocations.append(idempotency_key)
            self.attempts[idempotency_key] = self.attempts.get(idempotency_key, 0) + 1
            attempt = self.attempts[idempotency_key]
            settled = self.effects.get(idempotency_key)

        # THE IDEMPOTENCY KEY, HONOURED. A key that already took effect returns
        # the original result rather than doing it again, which is what makes
        # the Conductor's guarantee "effectively-once effect" rather than a
        # hope.
        if settled is not None:
            return ProviderResponse(
                status=settled.status,
                reference=settled.reference,
                deduplicated=True,
                detail="replayed from the provider's idempotency record",
            )

        if _band(idempotency_key, f"{self.seed}:permanent") < self.permanent_failure_rate:
            raise PermanentError(
                f"provider rejected {idempotency_key[:12]} permanently: "
                "recipient address is not routable"
            )

        if (
            _band(idempotency_key, f"{self.seed}:transient") < self.transient_failure_rate
            and attempt <= self.transient_attempts
        ):
            raise RetryableError(
                f"provider returned 500 for {idempotency_key[:12]} on attempt {attempt}"
            )

        response = ProviderResponse(
            status=self._status_for(idempotency_key),
            reference=f"{channel}-{idempotency_key[:16]}",
        )
        async with self._lock:
            self.effects[idempotency_key] = response
        return response

    def _status_for(self, idempotency_key: str) -> str:
        draw = _band(idempotency_key, f"{self.seed}:mix")
        for status, ceiling in self.mix:
            if draw < ceiling:
                return status
        return self.mix[-1][0]

    # -- observability -----------------------------------------------------
    @property
    def duplicate_invocations(self) -> int:
        """Keys presented more than once. A Conductor defect, not a provider one."""
        return len(self.invocations) - len(set(self.invocations))

    @property
    def distinct_keys(self) -> int:
        return len(set(self.invocations))

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for response in self.effects.values():
            counts[response.status] = counts.get(response.status, 0) + 1
        return counts


def forced_mix(status: str) -> tuple[tuple[str, float], ...]:
    """A mix that always returns one status, for tests that need a given outcome."""
    if status not in PROVIDER_STATUSES:
        raise ValueError(f"{status!r} is not a provider status")
    return ((status, 1.0),)


class AlwaysOutcome:
    """A provider pinned to one outcome, for exercising a single path.

    Takes a `ChannelOutcome` and works backwards to the provider status that
    produces it, so a test can ask for what it means rather than for the
    vendor's spelling of it.
    """

    def __init__(self, outcome: ChannelOutcome) -> None:
        from arc.channels.effectors import STATUS_TO_OUTCOME

        matches = [status for status, mapped in STATUS_TO_OUTCOME.items() if mapped is outcome]
        if not matches:
            raise ValueError(f"no provider status produces {outcome}")
        self.status = matches[0]
        self.invocations: list[str] = []

    async def deliver(
        self, channel: str, payload: Mapping[str, Any], idempotency_key: str
    ) -> ProviderResponse:
        self.invocations.append(idempotency_key)
        return ProviderResponse(status=self.status, reference=f"{channel}-{idempotency_key[:16]}")
