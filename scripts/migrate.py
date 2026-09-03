"""Apply migrations/*.sql in filename order, exactly once each.

Deliberately tiny: no ORM, no migration framework. The schema is authored as
SQL because the outbox (M9) depends on index and locking details that an ORM
would obscure.

Each applied file is recorded with a SHA-256 of its contents, so editing a
migration that has already run is an error rather than a silent divergence.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from pathlib import Path

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = REPO_ROOT / "migrations"
DEFAULT_DSN = "postgresql://arc:arc@localhost:5432/arc"

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT        PRIMARY KEY,
    checksum    TEXT        NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def _checksum(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


async def migrate(dsn: str) -> int:
    files = sorted(MIGRATIONS_DIR.glob("*.sql")) if MIGRATIONS_DIR.is_dir() else []
    if not files:
        print("migrate: no migrations found (nothing to apply)")
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(BOOTSTRAP)
        applied = {
            r["filename"]: r["checksum"]
            for r in await conn.fetch("SELECT filename, checksum FROM schema_migrations")
        }

        for path in files:
            sql = path.read_text(encoding="utf-8")
            digest = _checksum(sql)

            if path.name in applied:
                if applied[path.name] != digest:
                    print(f"migrate: FAIL {path.name} changed after being applied")
                    return 1
                print(f"migrate: skip {path.name} (already applied)")
                continue

            # One transaction per migration: a failure leaves no partial schema.
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename, checksum) VALUES ($1, $2)",
                    path.name,
                    digest,
                )
            print(f"migrate: applied {path.name}")
    finally:
        await conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(migrate(os.environ.get("DATABASE_URL", DEFAULT_DSN))))
