"""Evaluation record schemas for VideoDeepSearch agent evaluation.

Supports general agentic evaluation without bias toward specific tool call expectations.
Works with MLflow GenAI built-in scorers and DeepEval agentic metrics.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

class EvalInput(BaseModel):
    """Input for evaluation run.

    Maps to MLflow scorer inputs.
    """

    user_id: str = Field(description="User identifier")
    ground_truth_video_ids: list[str] = Field(default_factory=list, description="List of ground truth video IDs to search")
    total_video_haystack_ids: list[str] = Field(default_factory=list, description="List of all video IDs in the haystack")
    user_demand: str = Field(description="User's question or request")
    session_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Session ID (auto-generated)"
    )

class ExpectedFact(BaseModel):
    """A specific fact that should be present in the agent's response.

    Used by: Correctness scorer
    """

    fact: str = Field(description="The expected fact")
    source: str | None = Field(
        default=None,
        description="Source of this fact (e.g., 'video_123', 'ground_truth')"
    )


class EvalExpectation(BaseModel):
    """Expected outputs for evaluation.

    All fields are optional - only provide what's needed for specific scorers.

    Ground-truth free scorers (ToolCallEfficiency, TaskCompletion, etc.)
    don't need any expectations - they work from trace alone.
    """

    # For Correctness scorer
    expected_response: str | None = Field(
        default=None,
        description="Expected response text. Used by Correctness/Equivalence scorers"
    )
    expected_facts: list[ExpectedFact] | None = Field(
        default=None,
        description="List of facts that should be in the response. Used by Correctness scorer"
    )

    # Video search specific (for custom evaluation)
    expected_video_ids: list[str] | None = Field(
        default=None,
        description="Expected related video IDs (for domain-specific evaluation)"
    )

    # Context/notes for human reference
    notes: str | None = Field(
        default=None,
        description="Notes for evaluation reference (not used by scorers)"
    )

    
class EvalRecord(BaseModel):
    """Single evaluation record for MLflow GenAI.

    Minimal schema - most evaluation is done via trace analysis
    without needing ground truth expectations.

    Compatible with MLflow EvalDataset format:
    {
        "inputs": {...},
        "expectations": {...},
    }
    """

    inputs: EvalInput = Field(description="Input data for the agent")
    expectations: EvalExpectation | None = Field(
        default=None,
        description="Optional expectations (only for ground-truth based metrics)"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional context (tags, notes, etc.)"
    )

    def to_mlflow_format(self) -> dict[str, Any]:
        """Convert to MLflow dataset record format."""
        result: dict[str, Any] = {
            "inputs": {
                "user_id": self.inputs.user_id,
                "ground_truth_video_ids": self.inputs.ground_truth_video_ids,
                "total_video_haystack_ids": self.inputs.total_video_haystack_ids,
                "user_demand": self.inputs.user_demand,
            }
        }

        if self.expectations:
            expectations: dict[str, Any] = {}

            if self.expectations.expected_response:
                expectations["expected_response"] = self.expectations.expected_response

            if self.expectations.expected_facts:
                expectations["expected_facts"] = [
                    {"fact": f.fact, **({"source": f.source} if f.source else {})}
                    for f in self.expectations.expected_facts
                ]

            if self.expectations.expected_video_ids:
                expectations["expected_video_ids"] = self.expectations.expected_video_ids

            if expectations:
                result["expectations"] = expectations

        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalRecord":
        """Create EvalRecord from dictionary."""
        inputs_data = data.get("inputs", {})
        expectations_data = data.get("expectations")
        guidelines_data = data.get("guidelines")
        metadata = data.get("metadata")

        
        eval_input_args = {
            'user_id': inputs_data.get("user_id", ""),
            'ground_truth_video_ids': inputs_data.get("ground_truth_video_ids", []),
            'total_video_haystack_ids': inputs_data.get("total_video_haystack_ids", []),
            'user_demand': inputs_data.get("user_demand", ""),
        }
        print(eval_input_args)
        if inputs_data.get("session_id"):
            eval_input_args.update(
                {
                    'session_id': inputs_data.get("session_id")
                }
            )
        inputs = EvalInput(
            **eval_input_args
        )

        expectations = None
        if expectations_data:
            expected_facts = None
            if "expected_facts" in expectations_data:
                expected_facts = [
                    ExpectedFact(**f) if isinstance(f, dict) else ExpectedFact(fact=f)
                    for f in expectations_data["expected_facts"]
                ]

            expectations = EvalExpectation(
                expected_response=expectations_data.get("expected_response"),
                expected_facts=expected_facts,
                expected_video_ids=expectations_data.get("expected_video_ids"),
                notes=expectations_data.get("notes"),
            )
            
        return cls(
            inputs=inputs,
            expectations=expectations,
            metadata=metadata,
        )

class ToolCallTrace(BaseModel):
    """Captured tool call from agent execution.

    Field names match DeepEval ToolCall for compatibility:
    - name: Tool name
    - input_parameters: Arguments passed to tool
    - output: Tool result
    """

    name: str = Field(description="Tool name")
    input_parameters: dict[str, Any] | None = Field(
        default=None,
        description="Arguments passed to tool (None if no args)"
    )
    output: Any | None = Field(
        default=None,
        description="Tool result"
    )


class TokenUsage(BaseModel):
    """Token usage for a single model."""

    input_tokens: int = Field(default=0, description="Input tokens")
    output_tokens: int = Field(default=0, description="Output tokens")
    total_tokens: int = Field(default=0, description="Total tokens")


class EvalTrace(BaseModel):
    """Trace from running agent on an evaluation record.

    Populated after running agent. Used by ground-truth free scorers
    that analyze the trace directly.
    """

    session_id: str = Field(description="Session ID for this run")
    record: EvalRecord = Field(description="Original evaluation record")

    input: str = Field(description="User demand (input)")
    actual_output: str = Field(
        description="Final agent response. If agent failed, contains error trace log"
    )
    tool_trace: list[ToolCallTrace] = Field(
        default_factory=list,
        description="Tool calls made during execution"
    )

    token_usage: dict[str, TokenUsage] = Field(
        default_factory=dict,
        description="Token usage per model ID: {model_id: {input, output, total}}"
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Time taken to process the task (seconds)"
    )

    success: bool = Field(description="Whether the agent run succeeded")
    error_msg: str | None = Field(
        default=None,
        description="Error message if failed"
    )



__all__ = [
    # Input
    "EvalInput",
    # Expectations
    "EvalExpectation",
    "ExpectedFact",

    
    # Record
    "EvalRecord",
    # Trace
    "ToolCallTrace",
    "TokenUsage",
    "EvalTrace",
]
