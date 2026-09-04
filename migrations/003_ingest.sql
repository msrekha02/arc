-- M5: ingest. The raw archive, the dedupe window, and the arm registry.
--
-- Three tables, three different retention rules, and the differences are the
-- point:
--
--   raw_events     holds untrusted bytes including personal data. DELETABLE,
--                  and erasure must sweep it - it is the one place outside the
--                  subject store where a bank narration comes to rest.
--   ingest_dedupe  holds nothing personal. Rolls forward on a 30-day window.
--   subject_arms   holds the randomisation. Written ONCE per subject and never
--                  updated, because a subject who changes arm mid-experiment
--                  is in neither.

CREATE TYPE experiment_arm AS ENUM (
    'null',
    'naive_dunning',
    'gateway_default',
    'greedy_unconstrained',
    'arc'
);


-- ---------------------------------------------------------------------------
-- The raw immutable archive, written BEFORE the payload is parsed.
--
-- Records DELIVERIES, not events: a redelivered webhook is archived twice and
-- deduplicated once. Keeping those two facts apart is what makes "how often
-- does this gateway redeliver" answerable.
--
-- No unique constraint on (source, event_id). Duplicates are the traffic this
-- table exists to record, and event_id is nullable because a delivery that
-- failed to verify or to parse never produced one.
-- ---------------------------------------------------------------------------
CREATE TABLE raw_events (
    archive_id       BIGSERIAL     PRIMARY KEY,
    source           TEXT          NOT NULL,
    payload_hash     BYTEA         NOT NULL,
    signature        TEXT          NOT NULL,
    body             BYTEA         NOT NULL,
    received_at      TIMESTAMPTZ   NOT NULL,
    signature_valid  BOOLEAN       NOT NULL,

    -- Filled in after the parse. Null means it never got that far.
    event_id         TEXT,
    event_timestamp  TIMESTAMPTZ,
    parse_error      TEXT,

    -- Filled in after normalisation, so erasure can find every delivery
    -- belonging to a subject and delete it. Without this column the archive
    -- would be an unerasable copy of everything the subject store shreds.
    subject_token    TEXT,

    CONSTRAINT payload_hash_is_sha256 CHECK (octet_length(payload_hash) = 32)
);

CREATE INDEX ix_raw_events_source   ON raw_events (source, archive_id);
CREATE INDEX ix_raw_events_event    ON raw_events (source, event_id);
CREATE INDEX ix_raw_events_subject  ON raw_events (subject_token)
    WHERE subject_token IS NOT NULL;
CREATE INDEX ix_raw_events_unparsed ON raw_events (source, received_at)
    WHERE parse_error IS NOT NULL;


-- ---------------------------------------------------------------------------
-- Dedupe on (source, event_id) over a rolling 30-day window.
--
-- The primary key IS the check. A check-then-insert races two workers into
-- both believing they were first, and the cost of that race is a second claim,
-- a second diagnosis, and a second message to the same person.
-- ---------------------------------------------------------------------------
CREATE TABLE ingest_dedupe (
    source         TEXT        NOT NULL,
    event_id       TEXT        NOT NULL,
    first_seen_at  TIMESTAMPTZ NOT NULL,

    PRIMARY KEY (source, event_id)
);

CREATE INDEX ix_dedupe_age ON ingest_dedupe (first_seen_at);


-- ---------------------------------------------------------------------------
-- Subject-level arm assignment (GI-8).
--
-- Keyed by subject_token, NOT by claim_id, and there is nowhere in this table
-- to put a claim id. Claim-level randomisation violates SUTVA under a shared
-- contact budget, and the schema is the cheapest place to make that
-- impossible rather than merely discouraged.
-- ---------------------------------------------------------------------------
CREATE TABLE subject_arms (
    subject_token       TEXT           PRIMARY KEY,
    experiment_id       TEXT           NOT NULL,
    arm                 experiment_arm NOT NULL,

    -- The stratum as it stood when the assignment was made. Kept so the
    -- balance can be audited later against the arm that was actually given.
    claim_count_bucket  TEXT           NOT NULL,
    value_decile        SMALLINT       NOT NULL,
    rail                rail           NOT NULL,

    assigned_at         TIMESTAMPTZ    NOT NULL,

    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$'),
    CONSTRAINT value_decile_in_range
        CHECK (value_decile BETWEEN 0 AND 9),
    CONSTRAINT claim_count_bucket_is_known
        CHECK (claim_count_bucket IN ('1', '2_3', '4_plus'))
);

CREATE INDEX ix_subject_arms_arm ON subject_arms (experiment_id, arm);
