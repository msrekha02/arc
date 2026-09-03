-- 001_core.sql - the domain contract, in the database.
--
-- Mirrors arc/core/types.py. The enum labels here are the StrEnum *values*,
-- so a Python enum member round-trips through the database as itself.
--
-- Money is BIGINT paise. There is no NUMERIC and no DOUBLE PRECISION on any
-- monetary column anywhere in this schema, now or later (GI-2).

CREATE TYPE claim_type AS ENUM (
    'mandate_failure',
    'card_decline',
    'checkout_abandon',
    'invoice_overdue'
);

CREATE TYPE rail AS ENUM (
    'upi_autopay',
    'enach',
    'card',
    'invoice'
);

CREATE TYPE cause_layer AS ENUM (
    'issuer',
    'merchant',
    'customer',
    'unknown'
);

CREATE TYPE cohort_verdict AS ENUM (
    'degraded',
    'normal',
    'insufficient_power'
);

CREATE TYPE diagnosis_path AS ENUM (
    'cohort',
    'mandate',
    'code_map',
    'llm'
);

CREATE TYPE cause_label AS ENUM (
    'issuer_outage',
    'issuer_degraded',
    'mandate_orphaned',
    'mandate_cap_exceeded',
    'mandate_expired',
    'predebit_notice_missing',
    'wrong_debit_date',
    'insufficient_funds',
    'card_expired',
    'hard_decline',
    'do_not_retry',
    'mandate_revoked',
    'checkout_abandoned',
    'invoice_awaiting_approval',
    'invoice_disputed',
    'unknown'
);

CREATE TYPE claim_state AS ENUM (
    'detected',
    'diagnosed',
    'suppressed',
    'self_healing',
    'planned',
    'in_treatment',
    'promised',
    'escalated',
    'disputed',
    'recovered',
    'reversed',
    'written_off',
    'forborne'
);

-- CLOSED. Thirteen members, matching ActionType.
CREATE TYPE action_type AS ENUM (
    'do_nothing',
    'retry',
    'card_updater',
    'mandate_re_register',
    'rail_fallback',
    'whatsapp_utility',
    'sms',
    'email',
    'payment_link',
    'voice_call',
    'instalment_offer',
    'human_handoff',
    'statutory_notice'
);

CREATE TYPE tz_basis_kind AS ENUM (
    'declared',
    'billing_address',
    'telecom_circle'
);


CREATE TABLE claims (
    claim_id             UUID          PRIMARY KEY,

    -- Pseudonymous only. The CHECK is the database-side half of the redaction
    -- boundary: a raw phone number or email cannot be stored here at all.
    subject_token        TEXT          NOT NULL,

    amount_paise         BIGINT        NOT NULL,
    ltv_remaining_paise  BIGINT        NOT NULL,

    claim_type           claim_type    NOT NULL,
    rail                 rail          NOT NULL,
    detected_at          TIMESTAMPTZ   NOT NULL,

    -- Closed-vocabulary structured fields only. Raw text lives in the subject
    -- store behind evidence_ref, so erasure destroys it without touching the
    -- hash chain.
    evidence_structured  JSONB         NOT NULL DEFAULT '{}'::jsonb,
    evidence_ref         TEXT,
    evidence_hash        BYTEA         NOT NULL,

    -- Cause is filled by the Sentinel. All five columns move together.
    cause_label          cause_label,
    cause_layer          cause_layer,
    cause_confidence     DOUBLE PRECISION,
    cause_derived_from   diagnosis_path,
    cause_cohort_power   cohort_verdict,

    state                claim_state   NOT NULL DEFAULT 'detected',

    created_at           TIMESTAMPTZ   NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ   NOT NULL DEFAULT now(),

    CONSTRAINT amount_non_negative
        CHECK (amount_paise >= 0),
    CONSTRAINT ltv_non_negative
        CHECK (ltv_remaining_paise >= 0),
    CONSTRAINT evidence_hash_is_sha256
        CHECK (octet_length(evidence_hash) = 32),
    CONSTRAINT subject_token_is_pseudonymous
        CHECK (subject_token ~ '^sub_[0-9a-f]{32}$'),
    CONSTRAINT cause_is_whole_or_absent
        CHECK (
            (cause_label IS NULL AND cause_layer IS NULL AND cause_confidence IS NULL
             AND cause_derived_from IS NULL AND cause_cohort_power IS NULL)
            OR
            (cause_label IS NOT NULL AND cause_layer IS NOT NULL AND cause_confidence IS NOT NULL
             AND cause_derived_from IS NOT NULL AND cause_cohort_power IS NOT NULL)
        ),
    CONSTRAINT cause_confidence_is_a_probability
        CHECK (cause_confidence IS NULL OR (cause_confidence >= 0.0 AND cause_confidence <= 1.0)),
    -- An LLM-derived cause can never alone justify a money-moving action.
    CONSTRAINT llm_cause_confidence_capped
        CHECK (cause_derived_from IS DISTINCT FROM 'llm' OR cause_confidence <= 0.70)
);

CREATE INDEX ix_claims_subject      ON claims (subject_token);
CREATE INDEX ix_claims_state        ON claims (state);
CREATE INDEX ix_claims_detected_at  ON claims (detected_at);

COMMENT ON COLUMN claims.amount_paise IS
    'Integer paise. What failed.';
COMMENT ON COLUMN claims.ltv_remaining_paise IS
    'Integer paise. What is at risk, and what the objective is weighted by.';


-- The absorbing states, enforced by the database rather than trusted to the
-- application. FORBORNE is the hardship path; nothing reopens it, including a
-- transition to WRITTEN_OFF. The full transition table stays in Python so it
-- is single-sourced; only this one property is duplicated here, because it is
-- the one whose violation is silent and irreversible.
CREATE FUNCTION claims_absorbing_state_guard() RETURNS trigger AS $$
BEGIN
    IF OLD.state IN ('forborne', 'written_off')
       AND NEW.state IS DISTINCT FROM OLD.state THEN
        RAISE EXCEPTION
            'claim %: % is absorbing, no transition out is legal (attempted -> %)',
            OLD.claim_id, OLD.state, NEW.state;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_claims_absorbing_state
    BEFORE UPDATE ON claims
    FOR EACH ROW
    EXECUTE FUNCTION claims_absorbing_state_guard();
