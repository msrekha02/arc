-- 004_conductor.sql - the transactional outbox, reservations, and the
-- network attempt counter.
--
-- THE GOVERNING RULE: Postgres owns state, Inngest owns time. Everything here
-- is the state half. There is no message broker, and that is a decision rather
-- than an omission: the decision, the budget reservation, the ledger append
-- and the intent to dispatch have to commit together or not at all, and a
-- broker cannot participate in a Postgres transaction. Handing the intent to
-- Kafka after the commit reintroduces exactly the window this table removes.
--
-- WHAT IS GUARANTEED, PRECISELY:
--
--   exactly-once STATE TRANSITION   yes, by the transaction below
--   at-least-once DISPATCH          yes, by lease and retry
--   effectively-once EFFECT         yes, via the idempotency key the provider
--                                   honours
--   exactly-once DELIVERY           impossible, and not claimed
--
-- Claiming the last one is the tell that somebody has not thought about it.


CREATE TYPE outbox_status AS ENUM (
    'pending',
    'in_flight',
    'sent',
    'failed',
    'dead',
    'cancelled'
);

-- Two tiers, and the difference is the horizon. Anything executing inside the
-- next quarter hour takes a HARD reservation at plan time. Anything further
-- out records SOFT intent instead - visible to the Allocator as pipeline
-- demand, but not locking the budget - and converts to hard at wake.
--
-- WHY: holding a retry reservation for three days starves the portfolio. It is
-- also why a FREEZE can safely release reservations, because long-horizon work
-- never held one.
CREATE TYPE reservation_status AS ENUM (
    'soft',
    'hard',
    'consumed',
    'released',
    'expired'
);

CREATE TYPE retry_initiator AS ENUM (
    'arc',
    'gateway'
);


-- ---------------------------------------------------------------------------
-- The outbox
-- ---------------------------------------------------------------------------
CREATE TABLE outbox (
    id                BIGSERIAL     PRIMARY KEY,
    claim_id          UUID          NOT NULL,
    subject_token     TEXT          NOT NULL,
    cycle_id          UUID          NOT NULL,
    action_type       action_type   NOT NULL,
    channel           TEXT          NOT NULL,
    payload           JSONB         NOT NULL,

    -- sha256(claim_id:action_type:cycle_id:certificate_id).
    --
    -- THE ATTEMPT COUNTER IS NOT IN IT, and that is the whole design. A
    -- dispatch retry of the same decision must reuse the key so the provider
    -- deduplicates it; a genuine RE-DECISION after a wake must produce a new
    -- one so it is allowed through. Putting `attempts` in the key would make
    -- every retry look like a new instruction to charge somebody.
    idempotency_key   TEXT          NOT NULL,

    certificate_id    UUID          NOT NULL,
    cert_valid_from   TIMESTAMPTZ   NOT NULL,
    cert_valid_until  TIMESTAMPTZ   NOT NULL,

    not_before        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    priority          SMALLINT      NOT NULL DEFAULT 0,

    status            outbox_status NOT NULL DEFAULT 'pending',
    attempts          SMALLINT      NOT NULL DEFAULT 0,
    lease_owner       TEXT,
    lease_expires_at  TIMESTAMPTZ,
    last_error        TEXT,
    created_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT cert_window_is_ordered CHECK (cert_valid_until >= cert_valid_from),
    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$'),
    -- A leased row names its owner and its expiry, or it is not leased.
    CONSTRAINT lease_is_whole
        CHECK ((lease_owner IS NULL) = (lease_expires_at IS NULL))
);

-- Idempotent enqueue. `ON CONFLICT (idempotency_key) DO NOTHING` needs this,
-- and an Inngest step replay depends on it to not double-enqueue.
CREATE UNIQUE INDEX ux_outbox_idem ON outbox (idempotency_key);

-- The claim query's index. Partial, so it stays small as `sent` rows
-- accumulate: workers only ever look at pending work.
CREATE INDEX ix_outbox_ready ON outbox (priority DESC, not_before)
    WHERE status = 'pending';

-- The reaper's index, equally partial.
CREATE INDEX ix_outbox_leases ON outbox (lease_expires_at)
    WHERE status = 'in_flight';

CREATE INDEX ix_outbox_claim ON outbox (claim_id);

CREATE FUNCTION outbox_touch_updated_at() RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER outbox_touch
    BEFORE UPDATE ON outbox
    FOR EACH ROW EXECUTE FUNCTION outbox_touch_updated_at();


-- ---------------------------------------------------------------------------
-- Budgets, RESERVED rather than checked (GI-3)
--
-- The cap and the amount already reserved live in one row, so taking a
-- reservation is a single conditional UPDATE. There is no read-then-write
-- window for two workers to race through, which is what a check-then-act
-- implementation leaves open and what produces network-penalty overruns under
-- load.
-- ---------------------------------------------------------------------------
CREATE TABLE budget_caps (
    cycle_id    UUID    NOT NULL,
    budget_key  TEXT    NOT NULL,
    cap         BIGINT  NOT NULL,
    reserved    BIGINT  NOT NULL DEFAULT 0,

    PRIMARY KEY (cycle_id, budget_key),
    CONSTRAINT cap_non_negative      CHECK (cap >= 0),
    -- The invariant the whole table exists for. A reservation that would
    -- break it fails the UPDATE rather than being written and noticed later.
    CONSTRAINT reserved_within_cap   CHECK (reserved >= 0 AND reserved <= cap)
);

CREATE TABLE budget_reservations (
    reservation_id   BIGSERIAL          PRIMARY KEY,
    cycle_id         UUID               NOT NULL,
    claim_id         UUID               NOT NULL,
    subject_token    TEXT               NOT NULL,
    budget_key       TEXT               NOT NULL,
    amount           BIGINT             NOT NULL,
    status           reservation_status NOT NULL,
    idempotency_key  TEXT               NOT NULL,
    reserved_at      TIMESTAMPTZ        NOT NULL,
    -- When an unconsumed reservation goes stale. A leaked reservation is a
    -- slow starvation bug that never surfaces in a short test, so it carries
    -- its own expiry and the reaper sweeps it.
    expires_at       TIMESTAMPTZ        NOT NULL,
    released_at      TIMESTAMPTZ,

    CONSTRAINT amount_positive CHECK (amount > 0),
    CONSTRAINT released_iff_terminal CHECK (
        (status IN ('released', 'consumed', 'expired')) = (released_at IS NOT NULL)
    )
);

-- One reservation per (decision, budget). A retry of the same dispatch must
-- not reserve the budget twice.
CREATE UNIQUE INDEX ux_reservation_idem
    ON budget_reservations (idempotency_key, budget_key);

CREATE INDEX ix_reservations_live ON budget_reservations (expires_at)
    WHERE status IN ('soft', 'hard');

CREATE INDEX ix_reservations_claim ON budget_reservations (claim_id);


-- ---------------------------------------------------------------------------
-- Network attempts
--
-- THE GATEWAY RETRIES ON ITS OWN SCHEDULE, and those attempts count against
-- the network cap whether or not we issued them. A counter that only knows
-- about our own retries is wrong in the unsafe direction: the cap can be
-- exceeded without ARC ever having sent the excess, and the penalty lands on
-- us regardless.
--
-- So both initiators are recorded in one table and the cap is read from the
-- sum. The unique constraint deduplicates redelivered gateway events, which
-- arrive more than once by design.
-- ---------------------------------------------------------------------------
CREATE TABLE network_attempts (
    attempt_id      BIGSERIAL       PRIMARY KEY,
    instrument_ref  TEXT            NOT NULL,
    claim_id        UUID            NOT NULL,
    subject_token   TEXT            NOT NULL,
    rail            rail            NOT NULL,
    attempted_at    TIMESTAMPTZ     NOT NULL,
    initiated_by    retry_initiator NOT NULL,
    -- Stable per observed attempt, so a redelivered gateway webhook counts
    -- once. Ours is the idempotency key; theirs is their event id.
    attempt_ref     TEXT            NOT NULL,

    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$')
);

CREATE UNIQUE INDEX ux_network_attempt_ref
    ON network_attempts (instrument_ref, attempt_ref);

CREATE INDEX ix_network_attempts_window
    ON network_attempts (instrument_ref, attempted_at DESC);
