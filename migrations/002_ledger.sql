-- 002_ledger.sql - the three stores and the boundary between them.
--
--   decision_ledger  immutable, hash-chained, pseudonymous
--   subject_keys     one data key per subject; NULL means shredded
--   subject_records  ciphertext only, unreadable once the key is gone
--   money_entries    double-entry legs; balances are sums, never stored totals

CREATE TYPE ledger_event_type AS ENUM (
    'claim_detected',
    'claim_diagnosed',
    'state_transition',
    'decision',
    'certificate_issued',
    'gate_veto',
    'abandoned_unexecuted',
    'channel_dispatched',
    'money_transition',
    'promise_outcome',
    'tombstone'
);

CREATE TYPE money_account AS ENUM (
    'external',
    'at_risk',
    'in_treatment',
    'recovered',
    'settled',
    'reversed'
);


-- ---------------------------------------------------------------------------
-- Decision ledger
-- ---------------------------------------------------------------------------
CREATE TABLE decision_ledger (
    seq             BIGINT            PRIMARY KEY,

    claim_id        UUID,
    -- Pseudonymous, and NULL for entries that belong to no single subject.
    subject_token   TEXT,

    event_type      ledger_event_type NOT NULL,
    occurred_at     TIMESTAMPTZ       NOT NULL,
    recorded_at     TIMESTAMPTZ       NOT NULL DEFAULT now(),

    -- The exact bytes that were hashed. Verification re-reads these rather
    -- than re-deriving a serialisation, so it can never disagree with the
    -- writer about key order or number formatting.
    body_canonical  TEXT              NOT NULL,

    prev_hash       BYTEA             NOT NULL,
    entry_hash      BYTEA             NOT NULL UNIQUE,

    CONSTRAINT prev_hash_is_sha256  CHECK (octet_length(prev_hash) = 32),
    CONSTRAINT entry_hash_is_sha256 CHECK (octet_length(entry_hash) = 32),
    CONSTRAINT seq_is_positive      CHECK (seq >= 1),
    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token IS NULL OR subject_token ~ '^sub_[0-9a-f]{32}$')
);

-- Queryable projection with no second copy of the truth: it is derived from
-- the hashed bytes, so it cannot drift from what was signed.
ALTER TABLE decision_ledger
    ADD COLUMN body JSONB GENERATED ALWAYS AS (body_canonical::jsonb) STORED;

CREATE INDEX ix_ledger_claim   ON decision_ledger (claim_id, seq);
CREATE INDEX ix_ledger_subject ON decision_ledger (subject_token, seq);
CREATE INDEX ix_ledger_type    ON decision_ledger (event_type, seq);

-- Append-only, enforced rather than documented. The hash chain detects
-- tampering after the fact; this refuses it in the first place. Both exist
-- because the trigger can be disabled by whoever owns the table and the chain
-- cannot.
CREATE FUNCTION decision_ledger_append_only() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'decision_ledger is append-only (attempted % on seq %)',
        TG_OP, COALESCE(OLD.seq, NEW.seq);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_decision_ledger_append_only
    BEFORE UPDATE OR DELETE ON decision_ledger
    FOR EACH ROW
    EXECUTE FUNCTION decision_ledger_append_only();


-- ---------------------------------------------------------------------------
-- Subject store
-- ---------------------------------------------------------------------------
CREATE TABLE subject_keys (
    subject_token TEXT        PRIMARY KEY,
    -- NULL is the erasure. The row survives so the tombstone has something to
    -- point at, and so a later put() cannot silently re-key an erased subject.
    data_key      BYTEA,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    shredded_at   TIMESTAMPTZ,

    CONSTRAINT data_key_is_aes256 CHECK (data_key IS NULL OR octet_length(data_key) = 32),
    CONSTRAINT shredded_means_keyless
        CHECK ((shredded_at IS NULL) = (data_key IS NOT NULL)),
    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$')
);

CREATE TABLE subject_records (
    ref           TEXT        PRIMARY KEY,
    subject_token TEXT        NOT NULL REFERENCES subject_keys (subject_token),
    record_seq    INTEGER     NOT NULL,

    nonce         BYTEA       NOT NULL,
    ciphertext    BYTEA       NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT nonce_is_96_bit CHECK (octet_length(nonce) = 12),
    CONSTRAINT one_seq_per_subject UNIQUE (subject_token, record_seq)
);

CREATE INDEX ix_subject_records_token ON subject_records (subject_token);

COMMENT ON TABLE subject_records IS
    'Ciphertext only. Plaintext personal data never leaves this table decrypted.';


-- ---------------------------------------------------------------------------
-- Money ledger
-- ---------------------------------------------------------------------------
CREATE TABLE money_entries (
    id           BIGSERIAL     PRIMARY KEY,
    -- The two legs of one movement share a group_id and sum to zero.
    group_id     UUID          NOT NULL,
    claim_id     UUID          NOT NULL,
    account      money_account NOT NULL,
    delta_paise  BIGINT        NOT NULL,
    occurred_at  TIMESTAMPTZ   NOT NULL,
    recorded_at  TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT delta_is_a_movement CHECK (delta_paise <> 0)
);

CREATE INDEX ix_money_claim_account ON money_entries (claim_id, account);
CREATE INDEX ix_money_account       ON money_entries (account);
CREATE INDEX ix_money_group         ON money_entries (group_id);

COMMENT ON TABLE money_entries IS
    'Double-entry legs in integer paise. sum(delta_paise) over the table is 0.';
