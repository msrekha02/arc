# ARC - Autonomous Revenue Continuity

Detect revenue at risk, diagnose it, choose a bounded intervention, execute it
compliantly, and prove what was recovered.

See `ARC_architecture.md` for the design, `ARC_BUILD.md` for the build order,
and `CLAUDE.md` for the working conventions.

## Quick start

```bash
make up        # Postgres 16 via docker compose, waits for healthy
make migrate   # apply migrations/*.sql
make test      # pytest
make lint      # ruff
```

Requires Python 3.11 (managed by uv), Docker, and GNU make.
