"""Seed discipline and stream separation.

Three seeds, used for three different purposes, announced in advance:

    DEVELOP  develop against this one, freely
    TUNE     tune against this one, freely
    JUDGED   run ONCE, live, in front of the audience

WHY three: a system tuned on the same seed it is evaluated on has been fitted
to the evaluation set. Announcing which seed is which before the run is what
makes the final number a measurement rather than a selection.

Streams exist so that adding a draw in one part of the world does not shift
every other part. Population, outcomes, promises and wire delivery each pull
from an independently derived generator, so a change to outcome sampling
leaves the generated population byte-identical.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import numpy as np

DEVELOP_SEED = 1
TUNE_SEED = 2
JUDGED_SEED = 3

# The batch's decision moment. A literal, because nothing in this package may
# read a clock - TimeAuthority is the only caller of a wall clock in the repo,
# and a simulator that read one could not be replayed.
EPOCH: datetime = datetime(2025, 11, 3, 0, 0, 0, tzinfo=UTC)

# How far back the batch window reaches. Claims are detected inside
# [EPOCH - BATCH_DAYS, EPOCH); half-open, like every other window in ARC.
BATCH_DAYS = 14

# Depth of observable payment and decline history behind the batch window.
HISTORY_DAYS = 90

BATCH_START: datetime = EPOCH - timedelta(days=BATCH_DAYS)
HISTORY_START: datetime = EPOCH - timedelta(days=HISTORY_DAYS)


class Stream(StrEnum):
    """Independent random streams. One purpose each."""

    POPULATION = "population"
    FAILURES = "failures"
    OUTCOME = "outcome"
    PROMISE = "promise"
    WIRE = "wire"
    VALIDATE = "validate"


def _stream_key(stream: Stream) -> int:
    """A stable 32-bit key per stream, independent of enum ordering.

    Hashing the name rather than using its position means inserting a new
    stream member does not renumber the existing ones and silently change
    every generated batch.
    """
    digest = hashlib.sha256(str(stream).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def rng(seed: int, stream: Stream, *extra: int) -> np.random.Generator:
    """A seeded generator for one purpose. No global RNG exists in this repo.

    `extra` sub-divides a stream further - by account index, by cycle - so a
    per-account draw does not depend on how many accounts were drawn before it.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    return np.random.default_rng(np.random.SeedSequence([seed, _stream_key(stream), *extra]))


def stable_hash(*parts: str) -> int:
    """Deterministic, cross-process 64-bit hash of a delimited join.

    Python's built-in `hash()` is salted per process, so anything that must
    reproduce across runs - a per-month salary jitter, an account's issuer
    assignment - derives from this instead.
    """
    joined = "\x1f".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:8], "big")


def unit_hash(*parts: str) -> float:
    """`stable_hash` mapped into [0, 1). Deterministic, no generator needed."""
    return stable_hash(*parts) / 2**64
