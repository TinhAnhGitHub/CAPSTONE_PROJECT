from typing import Annotated, TypeVar, Awaitable, Callable, Any
import functools
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCallResult
from llama_index.core.llms import TextBlock
from .sub_orchestration import SUB_ORCHESTRATOR_STATE_KEY,SubOrchestrationState, Evidence
from videodeepsearch.agent.worker.planner.schema import WorkersPlan


async def get_typed_state(ctx: Context) -> SubOrchestrationState:
    """
    Helper Function
    """
    state_dict = await ctx.store.get(SUB_ORCHESTRATOR_STATE_KEY)
    return SubOrchestrationState.model_validate(state_dict)

async def save_types_state(ctx: Context, state_saved: SubOrchestrationState):
    """
    Helper Function
    """
    async with ctx.store.edit_state() as state:
        state[SUB_ORCHESTRATOR_STATE_KEY] = state_saved.model_dump(mode='json')

T = TypeVar("T")
def with_state(auto_save: bool=True):
    def decorator(
        func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(ctx: Context, *args, **kwargs) -> T:
            state_dict = await ctx.store.get(SUB_ORCHESTRATOR_STATE_KEY)
            state = SubOrchestrationState.model_validate(state_dict)
            result = await func(state, *args, **kwargs)
            if auto_save:
                async with ctx.store.edit_state() as s:
                    s[SUB_ORCHESTRATOR_STATE_KEY] = state.model_dump(mode="json")
            return result
        return wrapper
    return decorator



@with_state()
async def set_worker_plan_sub(state: SubOrchestrationState, worker_plan: WorkersPlan):
    state.worker_plans = worker_plan



async def sub_orchestration_state_update_findings(
    parent_ctx: Annotated[Any, "Parent workflow context managing global shared state (injected via partial params)."],
    worker_name: Annotated[str, "Name of the worker or sub-agent submitting findings."],
    report_summary: Annotated[str, "Short summary of the worker's findings or conclusions."],
    evidences: Annotated[list[Evidence], "List of evidence objects supporting the findings."],
    confidence: Annotated[int, "Confidence score for the findings, from 1 to 5."],
    found_answer: Annotated[bool, "Whether the worker successfully found an answer or resolution."],
) -> TextBlock:
    """
    Announce the worker's findings to the global shared state.
    This function updates the orchestration state with the worker's summary, confidence,
    evidences, and completion status. It also determines if the orchestration should exit early
    based on aggregated state.
    You (worker agent) must use this function at the end of every findings, before ending your task. It should refect your overall progress, and how you solve the problems. 
    Returns:
        str: A formatted status message summarizing the global state update.
    """
    state = await get_typed_state(parent_ctx)
    state.add_worker_finding(
        worker_name=worker_name,
        summary=report_summary,
        confidence=confidence,
        evidences=evidences,
        found_answer=found_answer,
    )
    await save_types_state(parent_ctx, state)

    string_return = (
        f" Global state updated from you: {worker_name}\n"
        f" Confidence: {confidence}/5 \n"
        f" Evidence items: {len(evidences)} \n"
        f" Found answer: {found_answer} \n"
    )
    return TextBlock(text=string_return)

async def sub_orchestration_state_update_tool_results(
    parent_ctx: Annotated[Any, "Parent workflow context managing the shared orchestration state (injected via partial params)."],
    worker_name: Annotated[str, "Name of the worker that executed the tool."],
    tool_call_results: Annotated[list[ToolCallResult], "Results returned by the tool call."],
) -> TextBlock:
    """
    Update the orchestration state with tool call results from a worker.

    This function records the output of tool invocations and persists the updated
    orchestration state back to the shared store.
    You (worker agent) must use this function at the end of every findings, before ending your task. It should refect how you use the functions. This is vital function to use for the orchestrator agent to have enough information
    Returns:
        str: A formatted message summarizing the update event.
    """

    state = await get_typed_state(parent_ctx)
    state.add_tool_results(
        worker_name=worker_name,
        tool_results=tool_call_results
    )
    await save_types_state(parent_ctx, state)
    return TextBlock(text=f"Global state updated with tool results from: {worker_name}")

async def sub_orchestration_state_view_results_from_agent_tools(
    parent_ctx: Any,
    agent_name: Annotated[str, "The name of the worker agent whose results you want to inspect."],
    view_tool_usage: Annotated[
        bool,
        (
            "Whether to display detailed information about the tools used by this agent "
            "during its execution. Set True to show the list of tools invoked, their inputs, "
            "outputs, and results; False to omit this section."
        ),
    ],
):
    """
    Retrieve and display the results, chat history, and tool usage details for a specific
    worker agent within the orchestration context.
    ---
    **Description:**
    This function allows developers or orchestrators to introspect what a given
    worker agent has produced or executed during its latest run. It can optionally
    display the full conversation history (including generated artifacts such as
    images, captions, video segments, or code outputs) and the list of tools used,
    along with their invocation details.
    """

    global_state = await get_typed_state(parent_ctx)

    agent_findings = global_state.findings.get(agent_name)
    if agent_findings is None:
        return f"There is no findings of the agent {agent_name}. Maybe it has not been started."

    tool_finding = "No information about the tool usage"
    if view_tool_usage:
        tool_usage = global_state.tool_logs.get(agent_name)
        tool_finding = tool_usage if tool_usage else tool_finding
    
    result_text = (
        f"**Agent Name:** {agent_name}\n"
        f"──────────────────────────────────────────────\n"
        f"**Agent Findings:**\n"
        f"{agent_findings}\n\n"
        f"🔧 **Tool Usage Details:**\n"
        f"{tool_finding}\n"
        f"──────────────────────────────────────────────\n"
        f"End of inspection for agent `{agent_name}`."
    )
    return result_text

async def sub_orchestration_state_synthesize_final_answers(
    parent_ctx: Any,
    answer_found: Annotated[
        bool,
        (
            "Whether the orchestration process successfully found an answer "
            "to the user's query or task. True if an answer is found, otherwise False."
        ),
    ],
    confidence: Annotated[
        int,
        (
            "Confidence score of the final synthesized answer, rated on a scale "
            "from 1 (very low confidence) to 5 (very high confidence)."
        ),
    ],
    final_answer_report: Annotated[
        str,
        (
            "A detailed, human-readable summary or explanation of the final synthesized "
            "answer. This can include reasoning steps, evaluation notes, or key findings."
        ),
    ],
):
    """
    Persist and summarize the final synthesized answer from the orchestration process
    into the global state, including whether an answer was found, its confidence level,
    and a detailed explanatory report.

    ---
    **Description:**
    This function finalizes the agent orchestration pipeline by writing the conclusive
    results of reasoning or computation into the shared `Context` state. It records:
    - Whether a valid answer was discovered.
    - The confidence rating assigned by the agent(s).
    - A detailed explanatory report summarizing the findings and reasoning chain.

    After updating, it persists the global state for downstream agents or orchestration
    logs to read.
    ---
    **Returns:**
    - `str`:
        Confirmation message indicating that the final answer and metadata have been
        successfully persisted to the global context.
    """

    global_state = await get_typed_state(parent_ctx)

    global_state.answer_found = answer_found
    global_state.confidence = confidence
    global_state.final_answer = final_answer_report  # corrected typo

    await save_types_state(parent_ctx, global_state)

    return (
        "✅ Global state successfully updated.\n"
        f"• Answer found: {answer_found}\n"
        f"• Confidence level: {confidence}/5\n"
        f"• Final report saved.\n\n"
        "You may now return or display the synthesized overall result."
    )
