"""The LLM boundary. Four sanctioned tasks, one client, one validator.

    contracts.py  the closed enums and schemas, and LLM_ENABLED
    redactor.py   PII scrubbing, and fencing untrusted text as data
    validator.py  schema, then groundedness, then safety, then canned fallback
    client.py     the ONLY module that talks to a model

ENFORCED BY IMPORT BAN, NOT BY CONVENTION. `arc/allocator`, `arc/gate` and
`arc/money` may not import this package at all, so an LLM answer has no path
into an allocation, a rule evaluation or an amount however it is dressed up.
The build fails on the import rather than a reviewer noticing it.
"""

from arc.llm_service.client import MODEL_VERSION, LlmClient
from arc.llm_service.contracts import (
    EXTRACTION_THRESHOLD,
    LLM_CONFIDENCE_CAP,
    STOP_INTENTS,
    CauseAnswer,
    GroundingFacts,
    Intent,
    IntentAnswer,
    LlmTask,
    Utterance,
    llm_enabled,
)
from arc.llm_service.redactor import Redaction, fence, redact
from arc.llm_service.validator import (
    Rejection,
    ValidationLog,
    Verdict,
    canned,
    validate,
    validate_grounding,
    validate_safety,
    validate_schema,
    validated_or_canned,
)

__all__ = [
    "EXTRACTION_THRESHOLD",
    "LLM_CONFIDENCE_CAP",
    "MODEL_VERSION",
    "STOP_INTENTS",
    "CauseAnswer",
    "GroundingFacts",
    "Intent",
    "IntentAnswer",
    "LlmClient",
    "LlmTask",
    "Redaction",
    "Rejection",
    "Utterance",
    "ValidationLog",
    "Verdict",
    "canned",
    "fence",
    "llm_enabled",
    "redact",
    "validate",
    "validate_grounding",
    "validate_safety",
    "validate_schema",
    "validated_or_canned",
]
