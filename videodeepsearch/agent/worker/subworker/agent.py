from typing import Any, Annotated
import re

from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.workflow import StopEvent

from .definition import WorkerCodeVideoAgent
from .prompt import WORKER_SYSTEM_PROMPT, CODE_ACT_PROMPT
from videodeepsearch.core.app_state import Appstate
from videodeepsearch.agent.state.sub_orc_state_tool import(
    sub_orchestration_state_update_findings,
    sub_orchestration_state_update_tool_results,
)


SUB_WORKER_NAME = "SUB_WORKER_AGENT"



# def create_additional_tools(
#     agent_name: str,
#     parent_ctx: Any,
# ) -> list[FunctionTool]:
    
#     findings_schema = create_schema_from_function(
#         name=f"sub_orc_update_findings_{agent_name}",
#         func=sub_orchestration_state_update_findings,
#         ignore_fields=['parent_ctx']
#     )

#     tool_results_schema = create_schema_from_function(
#         name=f"sub_orc_update_tool_results_{agent_name}",
#         func=sub_orchestration_state_update_tool_results,
#         ignore_fields=["parent_ctx"],
#     )
#     additional_tools = [
#         FunctionTool.from_defaults(
#             async_fn=sub_orchestration_state_update_findings,
#             fn_schema=findings_schema,
#             name=f"update_findings_{re.sub(r'[^a-zA-Z0-9_]+', '_', agent_name)}",
#             partial_params={
#                 'parent_ctx': parent_ctx,
#                 'worker_name': agent_name,
#             },
#         ),
#         FunctionTool.from_defaults(
#             async_fn=sub_orchestration_state_update_tool_results,
#             fn_schema=tool_results_schema,
#             name=f"update_tool_results_{re.sub(r'[^a-zA-Z0-9_]+', '_', agent_name)}",
#             partial_params={
#                 'parent_ctx': parent_ctx,
#                 'worker_name': agent_name,
#             }
#         )
#     ]
#     return additional_tools


async def run_worker_function_as_tools(
    agent_name: Annotated[str, "The specific name of the agent (camel-case syntax)"], 
    description: Annotated[str, "The description detail of the agent. This description will refect its jobs, role, and how it should reasoning, perspective, to get the answer, based on the provided tools"],
    task: Annotated[str, "The specific task that it must complete"],
    tools: Annotated[list[str], "The available tools that the system offer"],

    ctx: Context, # agent context (for streaming)
    parent_ctx: Any, # parent workflow context (for shared state)
    user_message: str, # user_message provide at run time, orchestration agent ignore
    llm: FunctionCallingLLM, # provide at run time, ignore
    user_id: str,
    list_video_ids: list[str],
    verbose: bool = False,# provide at run time, ignore
    timeout: int = 3600,# provide at run time, ignore
)-> str:
    
    # additional_tools = create_additional_tools(
    #     agent_name=agent_name,
    #     parent_ctx=parent_ctx
    # )
    all_tools = Appstate().tool_factory.get_all_tools_functool(user_id=user_id, list_video_id=list_video_ids)
    filter_tools = list(filter(lambda x: x[0] in tools, all_tools.items()))

    total_tools = [tool[1] for tool in filter_tools]  # + additional_tools


    async def _run(code:str) -> str:
        result = Appstate().code_sandbox.execute(code)
        return result.to_message()
    

    agent_instance = WorkerCodeVideoAgent(
        name=agent_name,
        execution_history_key=f"{agent_name}_worker_space",
        code_execute_fn=_run,
        description=description,
        system_prompt=WORKER_SYSTEM_PROMPT.format(user_demand=user_message, task=task),
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





    
