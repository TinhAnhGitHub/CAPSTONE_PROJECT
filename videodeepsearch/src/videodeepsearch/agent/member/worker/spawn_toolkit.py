from loguru import logger
from typing import Annotated
from dataclasses import dataclass

from agno.agent import Agent
from agno.models.base import Model
from agno.tools import tool, Toolkit

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

        super().__init__(name="spawn_worker_toolkit")

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
        for alias, names in sorted(all_tools.items()):
            lines.append(f"### {alias}")
            for name in names:
                lines.append(f"  - {alias}.{name}")
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
    async def spawn_and_run_worker(
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
    ) -> str:
        """Spawn a Worker Agent, run it on a specific task, and return its result."""
        worker_model = self.worker_models.get(model_name)
        if worker_model is None:
            available = list(self.worker_models.keys())
            return f"Error: Model '{model_name}' not found. Available: {available}"

        logger.info(
            f"[SpawnWorkerToolkit] Spawning {agent_name!r} | "
            f"model={model_name} | tools={tool_names if tool_names else 'ALL'} | task={task[:60]!r}..."
        )

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

        try:
            run_output = await worker.arun(input=task)
            result = run_output.content or "Worker completed but returned no content."
            logger.info(f"[SpawnWorkerToolkit] Worker {agent_name!r} finished.")
            return result
        except Exception as e:
            error_msg = f"Worker '{agent_name}' failed: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

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
            for alias, names in self.tool_selector.list_all().items()
            for name in names
        ]
        return self.tool_selector.resolve(all_names)