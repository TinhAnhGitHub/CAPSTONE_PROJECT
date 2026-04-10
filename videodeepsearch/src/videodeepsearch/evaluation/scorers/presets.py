"""Scorer presets for VideoDeepSearch evaluation.

Provides convenient preset configurations for different evaluation scenarios.
All presets use MLflow ToolCallCorrectness (ground-truth free) instead of DeepEval ToolCorrectness.
"""

from __future__ import annotations

from videodeepsearch.evaluation.scorers.config import (
    ScorerSettings,
    OPENROUTER_JUDGE_MODELS,
    DEFAULT_JUDGE_MODEL,
)

PRESET_GROUND_TRUTH_FREE = "ground_truth_free"
PRESET_WITH_EXPECTATIONS = "with_expectations"
PRESET_FULL_AGENTIC = "full_agentic"


def get_default_settings() -> ScorerSettings:
    """Get default scorer settings.

    Ground-truth free by default - works from MLflow trace alone.
    """
    return ScorerSettings()


def get_ground_truth_free_scorers(
    judge_model: str | None = None,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.5,
    include_reason: bool = True,
) -> list:
    """Get scorers that don't require ground truth expectations.

    These scorers analyze the MLflow trace to evaluate:
    - Tool call efficiency and correctness (MLflow, ground-truth free)
    - Response relevance to query
    - Task completion
    - Execution efficiency
    - Plan adherence
    - Tool argument correctness

    Args:
        judge_model: Full model URI (e.g., "openrouter:/openai/gpt-4o-mini")
        judge_model_id: OpenRouter model shortcut (default: "gpt-4o-mini")
        threshold: Pass threshold for DeepEval scorers
        include_reason: Include reasoning explanation in output

    Returns:
        List of scorer instances ready for mlflow.genai.evaluate()
    """
    settings = ScorerSettings(
        judge_model=judge_model,
        judge_model_id=judge_model_id,
        threshold=threshold,
        include_reason=include_reason,
        # Enable all ground-truth free scorers
        use_tool_call_efficiency=True,
        use_tool_call_correctness=True,  # MLflow ground-truth free
        use_relevance_to_query=True,
        use_task_completion=True,
        use_step_efficiency=True,
        use_plan_adherence=True,
        use_plan_quality=False,
        use_argument_correctness=True,
        # Disable expectation-based scorers
        use_correctness=False,
        use_equivalence=False,
    )
    return settings.get_scorers()


def get_with_expectations_scorers(
    judge_model: str | None = None,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.5,
    include_reason: bool = True,
) -> list:
    """Get scorers that work with ground truth expectations.

    Includes all ground-truth free scorers PLUS:
    - Correctness (requires expected_response or expected_facts)
    - Equivalence (requires expected_response)

    Use this preset when you have reference answers to compare against.

    Args:
        judge_model: Full model URI
        judge_model_id: OpenRouter model shortcut
        threshold: Pass threshold for DeepEval scorers
        include_reason: Include reasoning explanation in output

    Returns:
        List of scorer instances
    """
    settings = ScorerSettings(
        judge_model=judge_model,
        judge_model_id=judge_model_id,
        threshold=threshold,
        include_reason=include_reason,
        # Enable all ground-truth free scorers
        use_tool_call_efficiency=True,
        use_tool_call_correctness=True,
        use_relevance_to_query=True,
        use_task_completion=True,
        use_step_efficiency=True,
        use_plan_adherence=True,
        use_plan_quality=False,
        use_argument_correctness=True,
        # Enable expectation-based scorers
        use_correctness=True,
        use_equivalence=False,  # Usually too strict, correctness is better
    )
    return settings.get_scorers()


def get_full_agentic_scorers(
    judge_model: str | None = None,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.5,
    include_reason: bool = True,
    include_plan_quality: bool = True,
) -> list:
    """Get comprehensive agentic evaluation scorers.

    Includes ALL available scorers:
    - All MLflow built-in judges
    - All DeepEval agentic metrics
    - Plan quality (if agent uses planning)

    Use this for thorough evaluation of complex agent behavior.

    Args:
        judge_model: Full model URI
        judge_model_id: OpenRouter model shortcut
        threshold: Pass threshold for DeepEval scorers
        include_reason: Include reasoning explanation in output
        include_plan_quality: Include PlanQuality scorer (requires planning in trace)

    Returns:
        List of scorer instances
    """
    settings = ScorerSettings(
        judge_model=judge_model,
        judge_model_id=judge_model_id,
        threshold=threshold,
        include_reason=include_reason,
        # Enable all scorers
        use_tool_call_efficiency=True,
        use_tool_call_correctness=True,
        use_relevance_to_query=True,
        use_task_completion=True,
        use_step_efficiency=True,
        use_plan_adherence=True,
        use_plan_quality=include_plan_quality,
        use_argument_correctness=True,
        use_correctness=True,
        use_equivalence=False,
    )
    return settings.get_scorers()


def get_scorers_by_preset(
    preset: str,
    judge_model: str | None = None,
    judge_model_id: str = DEFAULT_JUDGE_MODEL,
    threshold: float = 0.5,
    include_reason: bool = True,
) -> list:
    """Get scorers by preset name.

    Args:
        preset: Preset name (ground_truth_free, with_expectations, full_agentic)
        judge_model: Full model URI
        judge_model_id: OpenRouter model shortcut
        threshold: Pass threshold
        include_reason: Include reasoning explanation

    Returns:
        List of scorer instances

    Raises:
        ValueError: If preset name is unknown
    """
    preset_functions = {
        PRESET_GROUND_TRUTH_FREE: get_ground_truth_free_scorers,
        PRESET_WITH_EXPECTATIONS: get_with_expectations_scorers,
        PRESET_FULL_AGENTIC: get_full_agentic_scorers,
    }

    if preset not in preset_functions:
        available = ", ".join(preset_functions.keys())
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    return preset_functions[preset](
        judge_model=judge_model,
        judge_model_id=judge_model_id,
        threshold=threshold,
        include_reason=include_reason,
    )


__all__ = [
    # Preset names
    "PRESET_GROUND_TRUTH_FREE",
    "PRESET_WITH_EXPECTATIONS",
    "PRESET_FULL_AGENTIC",
    # Preset functions
    "get_ground_truth_free_scorers",
    "get_with_expectations_scorers",
    "get_full_agentic_scorers",
    "get_scorers_by_preset",
    # Default settings
    "get_default_settings",
]