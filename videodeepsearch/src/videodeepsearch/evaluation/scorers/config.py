from __future__ import annotations

from pydantic import BaseModel, Field

OPENROUTER_JUDGE_MODELS = {
    "gpt-4o-mini": "openrouter:/openai/gpt-4o-mini",
    "gpt-4o": "openrouter:/openai/gpt-4o",
    "claude-3.5-sonnet": "openrouter:/anthropic/claude-3.5-sonnet",
    "claude-3-haiku": "openrouter:/anthropic/claude-3-haiku",
    "gemini-2.0-flash": "openrouter:/google/gemini-2.0-flash-001",
    "llama-3.3-70b": "openrouter:/meta-llama/llama-3.3-70b-instruct",
}

DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

class ScorerSettings(BaseModel):
    """Settings for evaluation scorers.

    All scorers are ground-truth free - they work from MLflow trace alone.
    """

    judge_model: str | None = Field(
        default=None,
        description="Full model URI for LLM judges"
    )
    judge_model_id: str = Field(
        default=DEFAULT_JUDGE_MODEL,
        description="OpenRouter model shortcut"
    )

    # MLflow scorers (ground-truth free)
    use_tool_call_efficiency: bool = Field(
        default=True,
        description="Check for redundant tool calls"
    )
    use_tool_call_correctness: bool = Field(
        default=True,
        description="Check if tools called are reasonable (ground-truth free)"
    )
    use_relevance_to_query: bool = Field(
        default=True,
        description="Check response relevance"
    )

    # DeepEval scorers (ground-truth free, require trace)
    use_task_completion: bool = Field(
        default=True,
        description="Check if task was completed"
    )
    use_step_efficiency: bool = Field(
        default=True,
        description="Check execution efficiency"
    )
    use_plan_adherence: bool = Field(
        default=True,
        description="Check if agent followed its plan"
    )
    use_plan_quality: bool = Field(
        default=False,
        description="Check plan quality (requires planning in trace)"
    )
    use_argument_correctness: bool = Field(
        default=True,
        description="Check tool argument correctness"
    )

    # Scorers requiring expectations
    use_correctness: bool = Field(
        default=False,
        description="Check expected facts (requires expected_response)"
    )
    use_equivalence: bool = Field(
        default=False,
        description="Check exact response match (requires expected_response)"
    )

    # DeepEval-specific settings
    threshold: float = Field(
        default=0.5,
        description="Pass threshold for DeepEval scorers"
    )
    include_reason: bool = Field(
        default=True,
        description="Include reasoning explanation in DeepEval scorer output"
    )

    def get_judge_model_uri(self) -> str:
        if self.judge_model:
            return self.judge_model
        if self.judge_model_id in OPENROUTER_JUDGE_MODELS:
            return OPENROUTER_JUDGE_MODELS[self.judge_model_id]
        return self.judge_model_id

    def get_scorers(self) -> list:
        
        from mlflow.genai.scorers import (
            ToolCallEfficiency,
            ToolCallCorrectness,
            RelevanceToQuery,
            Correctness,
            Equivalence,
        )
        from mlflow.genai.scorers.deepeval import (
            TaskCompletion,
            StepEfficiency,
            PlanAdherence,
            PlanQuality,
            ArgumentCorrectness,
        )

        model = self.get_judge_model_uri()
        scorers = []


        if self.use_tool_call_efficiency:
            scorers.append(ToolCallEfficiency(model=model)) #type:ignore

        if self.use_tool_call_correctness:
            scorers.append(ToolCallCorrectness(model=model)) #type:ignore

        if self.use_relevance_to_query:
            scorers.append(RelevanceToQuery(model=model))

        if self.use_task_completion:
            scorers.append(TaskCompletion(
                model=model, #type:ignore
                threshold=self.threshold, #type:ignore
                include_reason=self.include_reason, #type:ignore
            ))

        if self.use_step_efficiency:
            scorers.append(StepEfficiency(
                model=model, #type:ignore
                threshold=self.threshold, #type:ignore
                include_reason=self.include_reason, #type:ignore
            ))

        if self.use_plan_adherence:
            scorers.append(PlanAdherence(
                model=model, #type:ignore
                threshold=self.threshold, #type:ignore
                include_reason=self.include_reason, #type:ignore
            ))

        if self.use_plan_quality:
            scorers.append(PlanQuality(
                model=model, #type:ignore
                threshold=self.threshold, #type:ignore
                include_reason=self.include_reason, #type:ignore
            ))

        if self.use_argument_correctness:
            scorers.append(ArgumentCorrectness(
                model=model, #type:ignore
                threshold=self.threshold, #type:ignore
                include_reason=self.include_reason, #type:ignore
            ))

        if self.use_correctness:
            scorers.append(Correctness(model=model)) #type:ignore

        if self.use_equivalence:
            scorers.append(Equivalence(model=model)) #type:ignore

        return scorers

    def get_scorer_names(self) -> list[str]:
        """Get list of enabled scorer names."""
        return [s.name for s in self.get_scorers()]


def get_default_settings() -> ScorerSettings:
    """Get default scorer settings for ground-truth free evaluation."""
    return ScorerSettings()


def get_settings_for_model(model_id: str) -> ScorerSettings:
    """Get settings configured for a specific OpenRouter model."""
    return ScorerSettings(judge_model_id=model_id)

__all__ = [
    "ScorerSettings",
    "OPENROUTER_JUDGE_MODELS",
    "DEFAULT_JUDGE_MODEL",
    "get_default_settings",
    "get_settings_for_model",
]
