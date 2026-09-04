"""The only module that talks to a model, and the only one that may.

With `LLM_ENABLED` false - the default - every call returns the deterministic
path. That is not a stub standing in for the real thing; it IS the supported
configuration, exercised on every run, and the adversarial suite runs the whole
pipeline in it.

THERE IS NO NETWORK CALL IN THIS FILE, deliberately. Wiring a provider would
make the demo depend on a key and a rate limit and would prove nothing this
design has not already committed to. What matters is that the boundary exists,
that every output goes through the validator before it can reach a person, and
that the system is complete without it. A provider drops into `invoke`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from arc.llm_service.contracts import (
    GroundingFacts,
    Intent,
    LlmTask,
    Utterance,
    llm_enabled,
)
from arc.llm_service.redactor import fence, redact
from arc.llm_service.validator import (
    ValidationLog,
    Verdict,
    canned,
    digest,
    validated_or_canned,
)

MODEL_VERSION = "none/disabled"


@dataclass
class LlmClient:
    """The boundary. Returns validated output or the canned fallback, never raw.

    `invoke` is injectable so the adversarial suite can drive a model that
    returns a wrong amount and watch the validator refuse it. That is the only
    honest way to test a groundedness check: with output that is plausible and
    wrong, rather than output that is obviously broken.
    """

    invoke: Callable[[LlmTask, str], object] | None = None
    log: list[ValidationLog] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return llm_enabled() and self.invoke is not None

    def compose_message(
        self, *, template_id: str, facts: GroundingFacts, context: str = ""
    ) -> tuple[Utterance, Verdict]:
        """Task 2. Validated against the source record, or canned."""
        if not self.enabled:
            fallback = canned(template_id, facts)
            self._record(LlmTask.MESSAGE_GENERATION, "", fallback.text, "disabled")
            return fallback, Verdict(True, detail="llm disabled; canned template")

        prompt = fence(context, label="account context") if context else template_id
        raw = self.invoke(LlmTask.MESSAGE_GENERATION, prompt)  # type: ignore[misc]
        utterance, verdict = validated_or_canned(raw, facts, template_id=template_id)
        self._record(
            LlmTask.MESSAGE_GENERATION,
            prompt,
            getattr(raw, "text", str(raw)),
            verdict.refused_by,
        )
        return utterance, verdict

    def classify_intent(self, *, transcript: str) -> Intent:
        """Task 4, narrowed. UNCLEAR is a real answer and it is the default."""
        if not self.enabled:
            self._record(LlmTask.INTENT_EXTRACTION, "", "", "disabled")
            return Intent.UNCLEAR

        prompt = fence(transcript)
        raw = self.invoke(LlmTask.INTENT_EXTRACTION, prompt)  # type: ignore[misc]
        self._record(LlmTask.INTENT_EXTRACTION, prompt, str(raw), "parsed")
        return raw if isinstance(raw, Intent) else Intent.UNCLEAR

    def _record(self, task: LlmTask, prompt: str, output: str, verdict: str) -> None:
        self.log.append(
            ValidationLog(
                prompt_hash=digest(redact(prompt).text),
                output_hash=digest(output),
                model_version=MODEL_VERSION,
                verdict=verdict,
                task=task.value,
            )
        )
