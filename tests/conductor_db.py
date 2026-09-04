"""A disposable database for the Conductor tests that must COMMIT.

WHY THIS EXISTS. Most tests in this repo run inside a transaction that is
rolled back, so they leave nothing behind. The Conductor's concurrency tests
cannot: `SKIP LOCKED` across twenty workers is only meaningful between twenty
COMMITTED sessions, so those tests write for real.

That collides with the decision ledger, which is append-only by construction -
a trigger refuses UPDATE and DELETE, and the hash chain would notice anyway.
Rows written by a test therefore cannot be cleaned up, and running them against
the shared development database leaves permanent audit entries behind. It also
breaks M2's gate, which asserts the ledger starts empty, and that assertion is
right: an audit log with test traffic in it is not an audit log.

So the committing tests get their own database, created at session start and
dropped at the end. Nothing they write can reach the real one.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator
from pathlib import Path

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = REPO_ROOT / "migrations"

ADMIN_DSN = os.environ.get("DATABASE_URL", "postgresql://arc:arc@localhost:5432/arc")

# Deliberately unmistakable. Nobody should be able to read this name and think
# it is a database with anything worth keeping in it.
DB_PREFIX = "arc_scratch"


# ONE DATABASE PER CALLER, and the suffix is what makes that true. Two test
# modules sharing a name is not a theoretical collision: session-scoped
# fixtures in different modules are different fixtures, both live at once, and
# the second one to be created drops the first one's database mid-run. It
# happened to work only because the module names sorted the right way, which is
# not a property worth depending on - and it broke immediately when two pytest
# sessions ran at the same time.
def database_name(suffix: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in suffix).strip("_").lower()
    if not cleaned:
        raise ValueError("a scratch database needs a caller-specific suffix")
    return f"{DB_PREFIX}_{cleaned}"


def _swap_database(dsn: str, database: str) -> str:
    base, _, _ = dsn.rpartition("/")
    return f"{base}/{database}"


def dsn_for(suffix: str) -> str:
    return _swap_database(ADMIN_DSN, database_name(suffix))


async def _recreate(name: str) -> None:
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        # Anything still connected from a previous crashed run would block the
        # drop, so it is evicted rather than waited on.
        await admin.execute(
            """
            SELECT pg_terminate_backend(pid) FROM pg_stat_activity
             WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            name,
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()

    target = await asyncpg.connect(_swap_database(ADMIN_DSN, name))
    try:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            await target.execute(path.read_text(encoding="utf-8"))
    finally:
        await target.close()


async def _drop(name: str) -> None:
    admin = await asyncpg.connect(ADMIN_DSN)
    try:
        await admin.execute(
            """
            SELECT pg_terminate_backend(pid) FROM pg_stat_activity
             WHERE datname = $1 AND pid <> pg_backend_pid()
            """,
            name,
        )
        await admin.execute(f'DROP DATABASE IF EXISTS "{name}"')
    finally:
        await admin.close()


def create_scratch_database(suffix: str) -> str:
    """Build one caller's scratch database and return its DSN.

    Synchronous on purpose: called from a session-scoped pytest fixture, where
    an async fixture would have to agree with the event-loop scope of every
    test that used it. Running its own loop here sidesteps that entirely.
    """
    name = database_name(suffix)
    asyncio.run(_recreate(name))
    return _swap_database(ADMIN_DSN, name)


def drop_scratch_database(suffix: str) -> None:
    asyncio.run(_drop(database_name(suffix)))


def scratch_database(suffix: str) -> Iterator[str]:
    """Fixture body: create, yield the DSN, drop. One database per suffix."""
    dsn = create_scratch_database(suffix)
    try:
        yield dsn
    finally:
        drop_scratch_database(suffix)
