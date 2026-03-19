from agno.agent import Agent
from agno.models.base import Model
from agno.tools import Toolkit, Function


WORKER_SYSTEM_PROMPT: str = """
You are a specialised Worker Agent with the identifier: {agent_name}
 
## Original User Demand
{user_demand}
 
## Your Assigned Task
{task}
 
## Execution Plan
{detail_plan}
 
## Rules
- Complete ONLY your assigned task. Do not exceed its scope.
- Use only the tools available to you.
- Store every piece of evidence or intermediate result you find.
- Report your findings clearly and concisely when done.
- If a tool fails, retry once then report the failure with full details.
"""


def get_worker_agent(
    agent_name: str,
    description: str,
    task: str,
    detail_plan: str,
    user_demand: str,
    model: Model,
    instructions: list[str],
    toolkits: list[Toolkit] | None = None,
    functions: list[Function] | None = None,
    tool_call_limit: int = 20,
) -> Agent:
    """
    Spawn a Worker Agent for a single task.

    Called by the Orchestrator for each step in the execution plan.
    Workers are stateless — they receive all context they need via
    their system prompt and tools.

    Args:
        agent_name:     Unique snake_case identifier (e.g. "frame_extractor_01").
        description:    One-sentence description of what this worker does.
        task:           The specific task it must complete (from the plan).
        detail_plan:    Full step-by-step plan from the Planning Agent.
        user_demand:    The original user message for full context.
        user_id:        The user this run belongs to.
        list_video_ids: Video IDs this worker is authorised to access.
        model:          The LLM model to use.
        extra_toolkit:  Optional additional toolkit to merge with the base one.
    """
    
    assert toolkits or functions, "If using worker agents, please specify the extra functions + toolkits"
    
    system_prompt = WORKER_SYSTEM_PROMPT.format(
        agent_name=agent_name,
        user_demand=user_demand,
        task=task,
        detail_plan=detail_plan,
    )
    total_tools = (toolkits or []) + (functions or [])

    return Agent(
        name=agent_name,
        role=description,
        model=model,

        system_message=system_prompt,
        instructions=instructions,
        tools=total_tools,
        tool_call_limit=tool_call_limit,        
        add_session_state_to_context=False,
        enable_agentic_state=False,
        add_history_to_context=False,
        update_memory_on_run=False,
        enable_session_summaries=False,

        markdown=True,

        retries=1,
        delay_between_retries=1,

        debug_mode=False,
    )