from typing import Callable, Coroutine, Any

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import StopEvent

from .definition import WorkerCodeVideoAgent
from .prompt import WORKER_SYSTEM_PROMPT, CODE_ACT_PROMPT

from videodeepsearch.agent.worker.planner.schema import WorkerBluePrint
from videodeepsearch.tools.type.factory import ToolFactory
from videodeepsearch.agent.state.sub_orchestration import SubOrchestrationState
from videodeepsearch.agent.state.sub_orc_state_tool import get_typed_state
from videodeepsearch.core.app_state import Appstate
import logging
from llama_index.core.evaluation import RelevancyEvaluator, EvaluationResult
SUB_WORKER_NAME = "SUB_WORKER_AGENT"

async def run_worker_function_as_tools(
    ctx: Context, # agent context (for streaming)
    parent_ctx: Any, # parent workflow context (for shared state)
    agent_name: str, # agent name (append at run time) orchestration agent ignore
    user_message: str, # user_message provide at run time, orchestration agent ignore
    additional_tools: list[FunctionTool], # provide at run time, ignore
    llm: FunctionCallingLLM, # provide at run time, ignore
    user_id: str,
    list_video_ids: list[str],
    verbose: bool = False,# provide at run time, ignore
    timeout: int = 3600,# provide at run time, ignore
)-> str:
    """
    Asynchronous worker function that constructs and runs a specialized worker agent
    with dynamically selected tools based on its blueprint, executes the user's task,
    streams intermediate events, and returns the final result while persisting state.

    ---
    **Workflow Overview:**
    1. Retrieve the current orchestration or global state using the provided `ctx`.
    2. Extract the worker plan blueprint (`worker_blue_print`) associated with `agent_name`
       from the overall plan (`get_state.worker_plans.plan_detail`).
    3. Build the set of tools available to the worker by combining:
         - Tools defined in the worker blueprint.
         - Any `additional_tools` passed at runtime.
    4. Instantiate a `WorkerCodeVideoAgent` configured with:
         - The selected tools.
         - The LLM for reasoning and function calling.
         - System and code-acting prompts.
         - The code execution environment.
    5. Execute the agent asynchronously with the given `user_message`.
    6. Stream all intermediate events (e.g., tool calls, intermediate results)
       back through the context event stream.
    7. Await the final result and persist the outcome in the global context.
    8. Return the result as a string.
    ---
    **Returns:**
    - `str`:
        The final result of the worker agent's execution, serialized as a string.
        This includes the completed reasoning or task output after all tools
        and prompts have been applied.
    """
    get_state = await get_typed_state(parent_ctx)
    the_plan = get_state.worker_plans.plan_detail #type:ignore
    worker_blue_print = next(filter(lambda x: x.name == agent_name, the_plan))
    all_tools = Appstate().tool_factory.get_all_tools_functool(user_id=user_id, list_video_id=list_video_ids)
    filter_tools = list(filter(lambda x: x[0] in worker_blue_print.tools, all_tools.items()))
    total_tools = additional_tools + list(dict(filter_tools).values())
    async def _run(code:str) -> str:
        result = Appstate().code_sandbox.execute(code)
        return result.to_message()
    agent_instance = WorkerCodeVideoAgent(
        name=agent_name,
        execution_history_key=f"{agent_name}_worker_space",
        code_execute_fn=_run,
        description=worker_blue_print.description,
        system_prompt=WORKER_SYSTEM_PROMPT.format(user_demand=user_message, task=worker_blue_print.task),
        tools=total_tools,
        llm=llm,
        code_act_system_prompt=CODE_ACT_PROMPT,
        verbose=verbose,
        timeout=timeout
    )

    handler = agent_instance.run(user_msg=user_message)

    async for event in handler.stream_events():
        if isinstance(event, StopEvent):
            continue
        print(event)
        ctx.write_event_to_stream(event)
    
    try:
        result: AgentOutput = await handler
    except Exception as e:
        raise e
    return str(result)




    
