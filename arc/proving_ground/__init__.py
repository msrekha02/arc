"""L8's measurement half: arms, the doubly-robust estimator, and the metrics.

Only `arms.py` exists at M5, because randomisation has to happen before any
treatment decision. Assigning an arm retrospectively - after seeing which
claims the policy chose to act on - does not measure a policy, it describes
one.

This package and `arc.simulator` are the only two allowed to read ground
truth. Everything else is banned from it by name in CI.
"""

from arc.proving_ground.arms import (
    ARMS,
    Arm,
    ArmRegistry,
    Strata,
    assign_arm,
)
from arc.proving_ground.composed import (
    DO_NOTHING,
    ComposedPolicy,
    DecisionKey,
    MassNotConserved,
    Resolution,
    composed_propensity,
    veto_diagnostics,
)
from arc.proving_ground.dr_estimator import (
    Estimate,
    LoggedDecision,
    OutcomeModel,
    dr_estimate,
    fit_outcome_model,
    ground_truth_value,
    ips_estimate,
    on_policy_target,
)
from arc.proving_ground.metrics import (
    ArmReport,
    Diagnostics,
    Guardrails,
    GuardrailsMissing,
    Headline,
    PreventionMerged,
    Scoreboard,
)

__all__ = [
    "ARMS",
    "DO_NOTHING",
    "Arm",
    "ArmRegistry",
    "ArmReport",
    "ComposedPolicy",
    "DecisionKey",
    "Diagnostics",
    "Estimate",
    "Guardrails",
    "GuardrailsMissing",
    "Headline",
    "LoggedDecision",
    "MassNotConserved",
    "OutcomeModel",
    "PreventionMerged",
    "Resolution",
    "Scoreboard",
    "Strata",
    "assign_arm",
    "composed_propensity",
    "dr_estimate",
    "fit_outcome_model",
    "ground_truth_value",
    "ips_estimate",
    "on_policy_target",
    "veto_diagnostics",
]
