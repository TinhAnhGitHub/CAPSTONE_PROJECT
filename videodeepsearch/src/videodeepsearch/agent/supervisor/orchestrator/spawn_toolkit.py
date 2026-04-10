from loguru import logger
from typing import Annotated, AsyncIterator, Union
from dataclasses import dataclass

from agno.agent import Agent
from agno.models.base import Model
from agno.hooks import hook
from agno.run.agent import RunOutput, RunOutputEvent
from agno.tools import tool, Toolkit
from agno.run import RunContext

from videodeepsearch.agent.member.worker.agent import get_worker_agent
from videodeepsearch.agent.member.worker.tool_selector import ToolSelector


_TOOL_NAME_FORMAT_HINT = (
    "Tool names use the format 'alias.function_name'. "
    "Call get_available_worker_tools() first to see the full list."
)


@dataclass
class WorkerModel:
    """A model available for worker agents with metadata for selection."""
    model: Model
    description: str
    strengths: list[str]  # e.g., ["vision", "fast", "cheap", "reasoning"]


class SpawnWorkerToolkit(Toolkit):
    """Toolkit for spawning worker agents with configurable models.

    Supports multiple models - the orchestrator can choose which model
    to use for each worker via the model_name parameter.
    """

    def __init__(
        self,
        worker_models: dict[str, WorkerModel],
        worker_instructions: list[str],
        tool_selector: ToolSelector,
    ) -> None:
        self.worker_models = worker_models
        self.worker_instructions = worker_instructions
        self.tool_selector = tool_selector

        super().__init__(
            name="spawn_worker_toolkit",
            tools=[
                self.get_available_worker_tools,
                self.get_available_models,
                self.spawn_and_run_worker,
            ],
            async_tools=[
                (self.aspawn_and_run_worker, "spawn_and_run_worker"), #type:ignore
            ],
        )

    @tool(
        description=(
            "List all tools available to worker agents, grouped by toolkit alias. "
            "Call this BEFORE writing a plan so you know exactly which tool names to pass "
            "to spawn_and_run_worker(). Tool names follow the format 'alias.function_name'."
        ),
        cache_results=False,
    )
    def get_available_worker_tools(self) -> str:
        """List all tools available to worker agents, grouped by toolkit alias."""
        all_tools = self.tool_selector.list_all()
        if not all_tools:
            return "No worker tools registered."

        lines = ["## Available Worker Tools\n"]
        for alias, tools_dict in sorted(all_tools.items()):
            lines.append(f"### {alias}")
            for name, instructions in tools_dict.items():
                lines.append(f"  - `{alias}.{name}`: \n{instructions}\n")
            lines.append("")
        lines.append(_TOOL_NAME_FORMAT_HINT)
        return "\n".join(lines)

    @tool(
        description=(
            "List all available models for worker agents with their capabilities. "
            "Use this to decide which model is best suited for each task. "
            "Consider: task complexity, speed requirements, and specialized capabilities."
        ),
        cache_results=False,
    )
    def get_available_models(self) -> str:
        """List all available models for worker agents with their capabilities."""
        if not self.worker_models:
            return "No models registered."

        lines = ["## Available Worker Models\n"]
        lines.append("Choose the appropriate model based on task requirements:\n")

        for name, wm in sorted(self.worker_models.items()):
            model_id = getattr(wm.model, 'id', str(wm.model))
            strengths_str = ", ".join(wm.strengths) if wm.strengths else "general"

            lines.append(f"### `{name}`")
            lines.append(f"**Model:** {model_id}")
            lines.append(f"**Description:** {wm.description}")
            lines.append(f"**Best for:** {strengths_str}")
            lines.append("")

        lines.append("\n---\n**Tip:** Match model strengths to task requirements for optimal results.")
        return "\n".join(lines)

    def spawn_and_run_worker(
        self,
        run_context: RunContext,
        agent_name: Annotated[str, "Unique snake_case name for this worker"],
        description: Annotated[str, "One-sentence description of what this worker does"],
        task: Annotated[str, "The specific, scoped task this worker must complete"],
        detail_plan: Annotated[str, "The full step-by-step execution plan"],
        user_demand: Annotated[str, "The original user message"],
        model_name: Annotated[str, "Name of the model to use"],
        tool_names: Annotated[list[str], "Subset of tool names"] = [],
    ) -> str:
        """Spawn a Worker Agent (sync fallback - use async version)."""
        raise NotImplementedError("Use aspawn_and_run_worker instead")

    @tool(
        description=(
            "Spawn a Worker Agent, run it on a specific task, and return its result. "
            "Each worker is isolated with its own toolkit instance and result store. "
            "Choose the model based on task requirements (call get_available_models() first)."
        ),
        instructions=(
            "Use when: need to delegate a specific subtask to a specialized worker, "
            "want parallel execution of independent tasks, "
            "need to isolate tool access for focused context."
        ),
        cache_results=False,
    )
    async def aspawn_and_run_worker(
        self,
        agent_name: Annotated[
            str,
            "Unique snake_case name for this worker, e.g. 'image_search_worker_01'",
        ],
        description: Annotated[
            str,
            "One-sentence description of what this worker does",
        ],
        task: Annotated[
            str,
            "The specific, scoped task this worker must complete",
        ],
        detail_plan: Annotated[
            str,
            "The full step-by-step execution plan from the Planning Agent",
        ],
        user_demand: Annotated[
            str,
            "The original user message — provides full context to the worker",
        ],
        model_name: Annotated[
            str,
            (
                "Name of the model to use. Call get_available_models() to see options. "
                "Match model strengths to task: vision tasks → vision model, "
                "simple tasks → fast model, complex reasoning → reasoning model."
            ),
        ],
        tool_names: Annotated[
            list[str],
            (
                "Subset of tool names this worker needs. "
                "Format: 'alias.function_name' "
                "(e.g. ['search.get_images_from_qwenvl_query']). "
                "Pass [] for ALL tools."
            ),
        ] = [],
    ) -> AsyncIterator[Union[RunOutputEvent, str]]:
        
        func = self.functions.get('spawn_and_run_worker')
        run_context = func._run_context if func else None
        
        worker_model = self.worker_models.get(model_name)
        if worker_model is None:
            available = list(self.worker_models.keys())
            yield f"Error: Model '{model_name}' not found. Available: {available}"
            return
        
        resolved_tools = self._resolve_tools(tool_names)

        worker: Agent = get_worker_agent(
            agent_name=agent_name,
            description=description,
            task=task,
            detail_plan=detail_plan,
            user_demand=user_demand,
            model=worker_model.model,
            instructions=self.worker_instructions,
            functions=resolved_tools,
        )

        parent_run_id = None
        func = self.async_functions.get("spawn_and_run_worker")
        if func and func._run_context:
            parent_run_id = func._run_context.run_id

        stream = worker.arun(input=task, stream_events=True, stream=True)

        final_result = ""
        worker_run_output: RunOutput | None = None

        async for chunk in stream:
            if isinstance(chunk, RunOutput):
                worker_run_output = chunk
                continue

            if parent_run_id and hasattr(chunk, "parent_run_id"):
                chunk.parent_run_id = chunk.parent_run_id or parent_run_id

            yield chunk

            if hasattr(chunk, "content") and chunk.content:
                final_result += str(chunk.content)
        
        # Store worker metrics in session_state for post-hook aggregation
        if worker_run_output and worker_run_output.metrics:
            worker_metrics = worker_run_output.metrics
            if run_context.session_state:
                if "worker_metrics" not in run_context.session_state:
                    run_context.session_state["worker_metrics"] = []
                
                # Get model id and provider from RunOutput
                model_id = worker_run_output.model or ""
                model_provider = worker_run_output.model_provider or ""
                
                run_context.session_state['worker_metrics'].append(
                    {
                        "agent_name": agent_name,
                        "model_name": model_name,
                        "model_id": model_id,
                        "model_provider": model_provider,
                        "input_tokens": worker_metrics.input_tokens or 0,
                        "output_tokens": worker_metrics.output_tokens or 0,
                        "total_tokens": worker_metrics.total_tokens or 0,
                        "cost": worker_metrics.cost or 0.0,
                    }
                )
                
        result = str(worker_run_output.content) if worker_run_output and worker_run_output.content else final_result or "Worker completed but returned no content."

        yield result

    def _resolve_tools(self, tool_names: list[str]):
        """Resolve tool_names to Function objects, falling back to ALL if empty."""
        if tool_names:
            resolved = self.tool_selector.resolve(tool_names)
            if resolved:
                return resolved
            logger.warning(
                f"[SpawnWorkerToolkit] No tools resolved from {tool_names}, "
                f"falling back to ALL tools."
            )

        all_names = [
            f"{alias}.{name}"
            for alias, tools_dict in self.tool_selector.list_all().items()
            for name in tools_dict.keys()
        ]
        return self.tool_selector.resolve(all_names)
