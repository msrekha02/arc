-- ---------------------------------------------------------------------------
-- 005_control.sql - M12 durable functions, M13 kill switch, breakers, erasure
--
-- Postgres owns state. Inngest owns time. This file is the state half of that
-- split: durable runs and their memoised steps, the event log that cancels a
-- sleeping run, the kill-switch mode, the breakers, held work, and the erasure
-- register.
-- ---------------------------------------------------------------------------


-- ---------------------------------------------------------------------------
-- Durable runs.
--
-- One row per function invocation. `cancelled_by` names the event that killed
-- it, because "why did this stop" is the first question asked of a run that
-- did not finish, and reconstructing it from step rows is guesswork.
-- ---------------------------------------------------------------------------
CREATE TYPE run_status AS ENUM (
    'running',
    'sleeping',
    'waiting',
    'completed',
    'cancelled',
    'failed'
);

CREATE TABLE durable_runs (
    run_id          UUID          PRIMARY KEY,
    function_id     TEXT          NOT NULL,
    claim_id        UUID,
    subject_token   TEXT,
    tenant_id       TEXT          NOT NULL DEFAULT 'default',

    status          run_status    NOT NULL DEFAULT 'running',
    started_at      TIMESTAMPTZ   NOT NULL,
    finished_at     TIMESTAMPTZ,

    -- Set when a cancelOn event matched. The event name, not a boolean: a run
    -- cancelled by hardship and one cancelled by a system freeze are different
    -- facts and only one of them is about the person.
    cancelled_by    TEXT,
    outcome         TEXT,

    CONSTRAINT subject_token_is_pseudonymous_run
        CHECK (subject_token IS NULL OR subject_token ~ '^sub_[0-9a-f]{32}$')
);

CREATE INDEX ix_runs_claim   ON durable_runs (claim_id);
CREATE INDEX ix_runs_subject ON durable_runs (subject_token);
CREATE INDEX ix_runs_live    ON durable_runs (function_id)
    WHERE status IN ('running', 'sleeping', 'waiting');


-- ---------------------------------------------------------------------------
-- Memoised steps.
--
-- THE PRIMARY KEY IS THE MEMOISATION. A step that has already completed is
-- read back rather than re-executed, which is what makes a replayed function
-- idempotent. Inngest replays a function from the top on every resumption; a
-- step without a stable id would run again each time, and `gatedEnqueue` would
-- issue a second certificate for the same wake.
-- ---------------------------------------------------------------------------
CREATE TABLE durable_steps (
    run_id       UUID          NOT NULL REFERENCES durable_runs (run_id) ON DELETE CASCADE,
    step_id      TEXT          NOT NULL,
    completed_at TIMESTAMPTZ   NOT NULL,
    result       JSONB         NOT NULL DEFAULT '{}'::jsonb,

    PRIMARY KEY (run_id, step_id)
);


-- ---------------------------------------------------------------------------
-- The event log.
--
-- What `cancelOn` watches and what `waitForEvent` waits on. Append-only in
-- practice; nothing updates a row here.
--
-- `match_key` is the value a subscription matches against - a claim id for
-- claim.recovered, a subject token for subject.hardship, a tenant for
-- system.freeze. Keeping it in one column means a subscription is one
-- comparison rather than a per-event-type join.
-- ---------------------------------------------------------------------------
CREATE TABLE durable_events (
    event_seq   BIGSERIAL     PRIMARY KEY,
    name        TEXT          NOT NULL,
    match_key   TEXT          NOT NULL,
    occurred_at TIMESTAMPTZ   NOT NULL,
    payload     JSONB         NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX ix_events_lookup ON durable_events (name, match_key, occurred_at);


-- ---------------------------------------------------------------------------
-- Kill switch and admission ramp.
--
-- One row, id = TRUE, so there is exactly one mode and no way to write a
-- second. A mode is a global fact and a table that permits two of them will
-- eventually hold two.
-- ---------------------------------------------------------------------------
CREATE TYPE system_mode AS ENUM ('normal', 'shadow', 'drain', 'freeze');

CREATE TABLE system_control (
    only_row        BOOLEAN      PRIMARY KEY DEFAULT TRUE CHECK (only_row),
    mode            system_mode  NOT NULL DEFAULT 'normal',
    changed_at      TIMESTAMPTZ  NOT NULL,
    changed_by      TEXT         NOT NULL DEFAULT 'system',
    reason          TEXT         NOT NULL DEFAULT '',

    -- Resume ramp. NULL means not ramping; 0..3 indexes the 5/25/60/100 ladder.
    -- Admission is capped at that share of the trailing median until it clears.
    ramp_step       SMALLINT,
    ramp_started_at TIMESTAMPTZ,

    CONSTRAINT ramp_step_in_range CHECK (ramp_step IS NULL OR ramp_step BETWEEN 0 AND 3)
);

INSERT INTO system_control (only_row, mode, changed_at, changed_by, reason)
VALUES (TRUE, 'normal', now(), 'migration', 'initial state');


-- ---------------------------------------------------------------------------
-- Circuit breakers.
--
-- State, not thresholds: the thresholds live in code beside the rationale for
-- each one. `observed` and `threshold` are recorded at the trip so the reason
-- survives without having to re-derive it from metrics that have since moved.
-- ---------------------------------------------------------------------------
CREATE TABLE breaker_state (
    breaker_id    TEXT          PRIMARY KEY,
    tripped       BOOLEAN       NOT NULL DEFAULT FALSE,
    observed      DOUBLE PRECISION,
    threshold     DOUBLE PRECISION,
    evaluated_at  TIMESTAMPTZ,
    tripped_at    TIMESTAMPTZ,
    cleared_at    TIMESTAMPTZ,
    detail        TEXT          NOT NULL DEFAULT ''
);


-- ---------------------------------------------------------------------------
-- Work held by a FREEZE.
--
-- A separate table rather than a sixth outbox status, because held work is not
-- a dispatch state - it is a decision that has been INVALIDATED and owes the
-- Allocator a fresh one. Recording it separately also means the resume path
-- cannot accidentally select it with an outbox query written for dispatch.
--
-- `released_at` is set when the item is returned to the Allocator. It is never
-- set by anything that executes.
-- ---------------------------------------------------------------------------
CREATE TABLE held_work (
    held_id         BIGSERIAL     PRIMARY KEY,
    claim_id        UUID          NOT NULL,
    subject_token   TEXT          NOT NULL,
    outbox_id       BIGINT,
    idempotency_key TEXT          NOT NULL,
    action_type     action_type   NOT NULL,
    held_at         TIMESTAMPTZ   NOT NULL,
    reason          TEXT          NOT NULL,

    -- Set when the item is requeued for a fresh decision. Nothing dispatches
    -- from this table, so there is no "executed_at" and nowhere to put one.
    released_at     TIMESTAMPTZ,

    CONSTRAINT subject_token_is_pseudonymous_held
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$')
);

CREATE UNIQUE INDEX ux_held_idem ON held_work (idempotency_key);
CREATE INDEX ix_held_pending ON held_work (held_at) WHERE released_at IS NULL;


-- ---------------------------------------------------------------------------
-- Erasure register.
--
-- The request, and what the sweep destroyed. Pseudonymous throughout: this
-- table names a token and counts, and holds nothing that erasure was supposed
-- to remove.
--
-- Recording the erasure is itself an audit obligation, which is why the row
-- survives the data it describes.
-- ---------------------------------------------------------------------------
CREATE TABLE erasure_requests (
    erasure_id        UUID          PRIMARY KEY,
    subject_token     TEXT          NOT NULL,
    requested_at      TIMESTAMPTZ   NOT NULL,
    requested_by      TEXT          NOT NULL,

    completed_at      TIMESTAMPTZ,
    subject_refs_destroyed INTEGER  NOT NULL DEFAULT 0,
    archive_rows_purged    INTEGER  NOT NULL DEFAULT 0,
    outbox_rows_cancelled  INTEGER  NOT NULL DEFAULT 0,
    runs_cancelled         INTEGER  NOT NULL DEFAULT 0,
    tombstone_seq     BIGINT,

    CONSTRAINT subject_token_is_pseudonymous_erasure
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$')
);

CREATE INDEX ix_erasure_subject ON erasure_requests (subject_token);
