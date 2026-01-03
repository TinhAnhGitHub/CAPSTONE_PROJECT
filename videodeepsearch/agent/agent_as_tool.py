from typing import Annotated, cast
import logging

from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.workflow.handler import WorkflowHandler #type:ignore
from llama_index.core.workflow import Context, StopEvent
from llama_index.core.agent.workflow import AgentOutput, ToolCallResult
from llama_index.core.tools import FunctionTool

from videodeepsearch.agent.base import get_global_agent_registry
from videodeepsearch.tools.base.registry import tool_registry, get_registry_tools
from videodeepsearch.agent.context.worker_context import SmallWorkerContext
from videodeepsearch.agent.context.orc_context import WorkerResult, OrchestratorContext, prepare_init_ctx_each_run
from videodeepsearch.core.app_state import get_context_file_sys, get_llm_instance

from .definition import WORKER_AGENT, ORCHESTRATOR_AGENT, PLANNER_AGENT
from .prompt import WORKER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


async def run_planning_agent_as_tool(
    ctx: Context,
    orchestrator_demand: Annotated[str, "This is the demand of the orchestrator agent to the planning agent. Please specify it clearly. Start with Please make a plan about the event of ..."],
) -> str:
    """
    This is the Planning agent as tools. Please use this tool to invoke the planning agent 
    to make the detailed plan to solve/satisfy the user's query.
    """
    try:
        planning_tools = get_registry_tools()

      
        agent_registry = get_global_agent_registry()
        agent_instance = agent_registry.spawn(
            name=PLANNER_AGENT,
            llm=get_llm_instance(PLANNER_AGENT), tools=planning_tools
        )
        user_msg = f"Here is the orchestrator demand: {orchestrator_demand}"
        
        handler: WorkflowHandler = agent_instance.run(user_msg=user_msg)

        
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput) or isinstance(event, StopEvent):
                continue
            ctx.write_event_to_stream(ev=event)
        
        agent_output: AgentOutput = await handler

        response = cast(str, agent_output.response.content)
        
        return response
        
    except Exception as e:
        raise f"Planning agent encountered an error: {str(e)}" #type:ignore


async def running_worker_agent_as_tools(
    ctx: Context,
    agent_name: Annotated[str, "The specific name of the agent (snake-case syntax)"],
    description: Annotated[str, "The description detail of the agent"],
    task: Annotated[str, "The specific task that it must complete"],
    detail_plan: Annotated[str, "The detailed plan for the worker agent"],
    # partial params auto injected
    user_message: str,
    user_id: str,
    list_video_ids: str,
    llm: FunctionCallingLLM,
    session_id: str,
) -> str:
    """
    This is the worker agent as tools. Provide the task, description and a name 
    for the agent to conduct the task.
    """
    try:
        global_registry = get_global_agent_registry()
        agent_name_key = f"{agent_name}"


        worker_partial_params = {
            'user_id': user_id,
            'list_video_id': list_video_ids,
            'agent_name_key': agent_name_key
        }

        worker_tools = tool_registry.get_concrete_agent_tools(
            agent_name=WORKER_AGENT, 
            **worker_partial_params
        )
        system_prompt = WORKER_SYSTEM_PROMPT.format(
            user_demand=user_message,
            task=task,
            detail_plan=detail_plan
        )
        
        agent_instance = global_registry.spawn(
            name=WORKER_AGENT,
            agent_name=agent_name_key,
            description=description,
            system_prompt=system_prompt,
            tools=worker_tools,
            llm=llm
        )
        
        worker_context = Context(agent_instance)
        init_local_context = SmallWorkerContext(
            worker_agent_name=agent_name_key,
            task_objective=task,
        )
        
        async with worker_context.store.edit_state() as worker_ctx_state:
            worker_ctx_state[agent_name_key] = init_local_context.model_dump(mode='json')
        


        handler = agent_instance.run(ctx=worker_context, user_msg=user_message)
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput) or isinstance(event, StopEvent):
                continue
            ctx.write_event_to_stream(ev=event)
        agent_output: AgentOutput = await handler
        
        agent_response = cast(str, agent_output.response.content)
        
        worker_context_data = await worker_context.store.get(agent_name_key)
        if worker_context_data is None:
            raise ValueError(f"Worker context for {agent_name_key} not found")
            
        worker_context_obj = SmallWorkerContext.model_validate(worker_context_data)
        
        
        agent_worker_result = WorkerResult(
            worker_name=agent_name_key,
            task_objective=task,
            worker_chat_history=worker_context_obj.chat_history,
            raw_result_store=worker_context_obj.raw_result_store,
            evidences=worker_context_obj.evidences,
            result_summary=agent_response,
        )
        
        
        async with ctx.store.edit_state() as ctx_state:
            raw_shared_context = ctx_state.get(session_id)
            if raw_shared_context is None:
                raise ValueError(f"Session {session_id} does not contain shared context")
            
            shared_context = OrchestratorContext.model_validate(raw_shared_context)
            shared_context.add_to_latest_worker_results(agent_worker_result)
            ctx_state[session_id] = shared_context.model_dump(mode='json')
    
        
        return agent_response
        
    except Exception as e:
        logger.error(f"Error in running_worker_agent_as_tools for {agent_name}: {e}", exc_info=True)
        raise f"Worker agent '{agent_name}' encountered an error: {str(e)}" #type:ignore
 

async def running_orchestrator_agent_as_tools(
    ctx: Context,
    demand: Annotated[str, "The main demand that planning agent want to demand the orchestrator to do"],
    # auto-injected
    session_id: str,
    user_id: str,
    list_video_id: str,
    user_original_user_message: str,
) -> str:
    """
    Tool to run the orchestrator agent with proper error handling.
    """
    orchestrator_ctx = None
    agent_key_name = f"{ORCHESTRATOR_AGENT}_on_session_{session_id}"
    
    try:
        
        planning_agent_as_function_tool = FunctionTool.from_defaults(
            async_fn=run_planning_agent_as_tool,
        )
        
        worker_partial_params = {
            'user_id': user_id,
            'list_video_ids': list_video_id,
            'user_message': user_original_user_message,
            'llm': get_llm_instance(WORKER_AGENT),
            'session_id': agent_key_name
        }
        
        worker_agent_as_function_tool = FunctionTool.from_defaults(
            async_fn=running_worker_agent_as_tools,
            partial_params=worker_partial_params
        )

        orc_partial_params = {'session_id': agent_key_name}
        orc_tools = tool_registry.get_concrete_agent_tools(
            agent_name=ORCHESTRATOR_AGENT, 
            **orc_partial_params
        )
            
        agent_registry = get_global_agent_registry()
        orc_total_tools = [
            planning_agent_as_function_tool,
            worker_agent_as_function_tool
        ] + orc_tools
    
        orchestrator_agent = agent_registry.spawn(
            name=ORCHESTRATOR_AGENT,
            agent_name=agent_key_name,
            llm=get_llm_instance(ORCHESTRATOR_AGENT),
            tools=orc_total_tools
        )
        
        context_file_manager = get_context_file_sys()
        context_dict = context_file_manager.load_context(session_id)
        
        if context_dict is None:
            
            logger.info(f"Creating new orchestrator context for session {session_id}")
            orchestrator_ctx = Context(orchestrator_agent)
            
            async with orchestrator_ctx.store.edit_state() as ctx_state:
                ctx_state[agent_key_name] = prepare_init_ctx_each_run(
                    current_orc_ctx=None,
                    session_id=session_id,
                    user_demand=user_original_user_message
                )
        else:
            
            logger.info(f"Loading existing orchestrator context for session {session_id}")
            orchestrator_ctx = Context.from_dict(
                workflow=orchestrator_agent,
                data=context_dict
            )
            
            shared_context_data = await orchestrator_ctx.store.get(agent_key_name)
            if shared_context_data is None:
                raise ValueError(f"Agent key {agent_key_name} not found in context")
                
            shared_context = OrchestratorContext.model_validate(shared_context_data)
            
            updated_shared_context = prepare_init_ctx_each_run(
                current_orc_ctx=shared_context,
                session_id=session_id,
                user_demand=user_original_user_message
            )
            
            async with orchestrator_ctx.store.edit_state() as ctx_state:
                ctx_state[agent_key_name] = updated_shared_context
        
        
        prompt = f"""
        Here is the original user demand: {user_original_user_message}

        And here is the task from the greeting agent that you have to get done:
        {demand}
        """
                
        handler = orchestrator_agent.run(user_msg=prompt, ctx=orchestrator_ctx)
        
        async for event in handler.stream_events():
            if isinstance(event, AgentOutput) or isinstance(event, StopEvent):
                continue

            ctx.write_event_to_stream(ev=event)
        
        agent_final_output: AgentOutput = await handler
        

        context_file_manager.save_context(session_id, orchestrator_ctx)
        
        return cast(str, agent_final_output.response.content)
        
    except Exception as e:
        logger.error(f"Error in running_orchestrator_agent_as_tools: {e}", exc_info=True)
        
        raise f"Orchestrator agent encountered an error: {str(e)}" #type:ignore
        