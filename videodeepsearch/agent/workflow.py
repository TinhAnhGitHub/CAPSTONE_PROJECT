"""
The main agent that we will call uponn
"""
import re
#from llama_index.llms.google_genai import GoogleGenAI
from typing import Annotated, cast
import json
from pydantic import BaseModel

from llama_index.core.workflow import (
    Workflow, 
    StopEvent,
    step, 
    Context
)
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function
from llama_index.core.llms import LLM, ChatMessage, MessageRole, ChatResponse
from llama_index.core.agent.workflow import (
    AgentInput, 
    AgentOutput,
    AgentStream, 
    ToolCall,
    ToolCallResult,
    AgentStreamStructuredOutput
)
from llama_index.core.workflow.handler import WorkflowHandler # type:ignore
from llama_index.core.workflow.resource import Resource #type:ignore
from llama_index.core.base.llms.types import ThinkingBlock, ContentBlock, TextBlock


from videodeepsearch.core.app_state import get_app_state, Appstate


from videodeepsearch.agent.worker import GREETER_NAME, PLANNER_NAME, SUB_WORKER_NAME, ORCHESTRATION_NAME, FINAL_RESPONSE_AGENT

from videodeepsearch.agent.worker.greeter.schema import NextAgentDirective
from videodeepsearch.agent.worker.planner.schema import WorkersPlan
from videodeepsearch.agent.worker.subworker.agent import run_worker_function_as_tools

from videodeepsearch.agent.worker.suborchestrate.agent import ORCHESTRATION_NAME

from videodeepsearch.tools.type.factory import ToolFactory
from videodeepsearch.core.dependencies import get_tool_factory



from videodeepsearch.agent.state.full_orc_state_tool import (
    set_add_message_to_chat_history,
    set_reset_plan_state,
    set_state_from_user,
    set_worker_plan,
    get_chat_history,
    get_worker_plan,
    get_state_from_user
)

from videodeepsearch.agent.state.sub_orc_state_tool import (
    set_worker_plan_sub
)

from videodeepsearch.agent.state.sub_orc_state_tool import(
    sub_orchestration_state_update_findings,
    sub_orchestration_state_update_tool_results,
    sub_orchestration_state_view_results_from_agent_tools,
    sub_orchestration_state_synthesize_final_answers,  
)

from videodeepsearch.agent.state import create_sub_orchestrator_initial_state, SUB_ORCHESTRATOR_STATE_KEY


from .orc_prompt import OUTPUT_SYSTEM_PROMPT
from .orc_prompt import GREETING_AGENT_DECISION_OUTPUT, PLANNER_AGENT_OUTPUT
from .orc_prompt import PLANNER_USER_INPUT_TEMPLATE

from .base import AgentRegistry, get_global_agent_registry

from .orc_events import (
    UserInputEvent,
    FinalResponseEvent,
    AgentProgressEvent,
    PlannerInputEvent,
    PlanProposedEvent,
    FinalEvent,
    PlanningAgentEvent
)


from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from llama_index.core.evaluation import RelevancyEvaluator, EvaluationResult

#from llama_index.llms.gemini import Gemini

tracer_provider = register(
    project_name="VideoQA_v2",
    auto_instrument=True,
    endpoint="http://localhost:6006/v1/traces")
LlamaIndexInstrumentor(tracer_provider=tracer_provider).instrument()



def eval():
    pass
async def get_streaming_response( # with thinking
    ctx: Context,
    current_llm_input: list[ChatMessage],
    llm: LLM,
    agent_name: str
) -> ChatMessage:
    response = await llm.astream_chat(
        current_llm_input
    )
    streaming_token_len = 0
    last_chat_response = ChatResponse(message=ChatMessage())
    async for last_chat_response in response:
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        list_of_content_blocks = last_chat_response.message.blocks
        text_block: TextBlock | None = next(
            filter(lambda x: isinstance(x, TextBlock), list_of_content_blocks), None
        ) #type:ignore
        
        thinkink_blocks: list[ThinkingBlock] = list(
            filter(lambda x: isinstance(x, ThinkingBlock), list_of_content_blocks)
        )#type:ignore
        
        if text_block:
                text = text_block.text  
                text_delta = text[streaming_token_len:]

                ctx.write_event_to_stream(
                    AgentStream(
                        delta=text_delta,
                        response=text, # full repsonse
                        tool_calls=[],
                        raw=raw,
                        current_agent_name=agent_name,
                        thinking_delta=None
                    )
                )
                streaming_token_len = len(text)
                continue
        
        if len(thinkink_blocks) > 0:
            try:
                thinking_delta = thinkink_blocks[-2] # [thinking delta, thinking full]
                ctx.write_event_to_stream(
                    AgentStream(
                        delta="", 
                        response="",
                        tool_calls= [],
                        raw=raw,
                        current_agent_name=agent_name,
                        thinking_delta=thinking_delta.content
                    )
                )

            except Exception as e:
                continue
    return last_chat_response.message

class VideoAgentWorkFlow(Workflow):
    """
    Multi-agent orchestration with HIL plan approval
    
    Flow:
        User -> greeting agent -> Planner -> [Plan Review Loop] -> Orchestrator -> Workers -> Response

    The workflow STOPS at plan proposal, waits for user feedback, then continues.
    """ 

    @step
    async def handle_greeting(
        self,
        ctx: Context,
        ev: UserInputEvent,
        app_state: Annotated[Appstate, Resource(get_app_state)],
        agent_registry: Annotated[AgentRegistry, Resource(get_global_agent_registry)]
    ) -> FinalResponseEvent | PlannerInputEvent | PlanProposedEvent:

        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=GREETER_NAME,
                answer="Greeting Agent is handling your request..."
            )   
        )

        user_message = ChatMessage(role='user', content=ev.user_demand)
        await set_state_from_user(ctx=ctx, user_id=ev.user_id, list_video_ids=ev.list_video_ids)
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[user_message])
        greeting_agent = agent_registry.spawn(
            name=GREETER_NAME,
            llm=app_state.llm_instance[GREETER_NAME]
        )  

        chat_history = await get_chat_history(ctx=ctx)
        agent_response: AgentOutput = await greeting_agent.run( user_msg=user_message, chat_history=chat_history,)
        agent_decision = NextAgentDirective.model_validate(agent_response.structured_response)    

        next_agent = agent_decision.choose_next_agent
        reason = agent_decision.reason
        passing_message = agent_decision.passing_message


        message = f"Decided to route to the next agent: {next_agent}" if next_agent else f"{passing_message}. I will wrap up the final response..."
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=GREETER_NAME,
                answer=message
            )
        )

        format_response = GREETING_AGENT_DECISION_OUTPUT.format(agent=next_agent, reason=reason, passing_message=passing_message)
        chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, 
                                        content=format_response,
                                        additional_kwargs={"ent": "greeter"})
                            )
        
        await set_add_message_to_chat_history(
            ctx=ctx,
            chat_messages=[ChatMessage(role=MessageRole.ASSISTANT, content=format_response, additional_kwargs={"ent": "greeter"})]
        )

        passing_messages = [
            ChatMessage(role=MessageRole.ASSISTANT, content=passing_message)
        ]


        if next_agent == "planner": 
            await set_reset_plan_state(ctx=ctx) # sort of brittle, change later
            return PlannerInputEvent(
                user_msg=ev.user_demand,
                planner_demand=passing_message
            )
        
        elif next_agent == "orchestrator":
            worker_plan = await get_worker_plan(ctx=ctx)

            return PlanProposedEvent(
                user_msg=ev.user_demand,
                agent_response=passing_message,
                worker_plan=cast(WorkersPlan, worker_plan),
            )
        return FinalResponseEvent(passing_messages=passing_messages)
        
    @step
    async def final_response(
        self,
        ev:FinalResponseEvent,
        ctx: Context,
        app_state: Annotated[Appstate, Resource(get_app_state)]
    ) -> FinalEvent:
        
        passing_messages = ev.passing_messages
        chat_message = [ChatMessage(role=MessageRole.ASSISTANT, content=OUTPUT_SYSTEM_PROMPT)] + passing_messages
        

        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=FINAL_RESPONSE_AGENT,
                answer="Wrapping your final result..."
            )
        )

        llm = app_state.llm_instance[FINAL_RESPONSE_AGENT]
        final_response_message = await get_streaming_response(ctx=ctx, current_llm_input=chat_message, llm=llm, agent_name="FINAL_RESPONSE_AGENT")

        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[final_response_message] )

        chat_history = await get_chat_history(ctx=ctx)
        return FinalEvent(
            workflow_response=str(final_response_message.content),
            chat_history=chat_history
        )

    
    @step
    async def planning(
        self,
        ctx: Context,
        ev: PlannerInputEvent,
        app_state: Annotated[Appstate, Resource(get_app_state)],
        agent_registry: Annotated[AgentRegistry, Resource(get_global_agent_registry)]
    ) ->PlanProposedEvent | FinalResponseEvent:
    
        user = ev.user_msg
        message_demand = ev.planner_demand

        message = PLANNER_USER_INPUT_TEMPLATE.format(
            user_message=user,
            planner_message=message_demand
        )

        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=PLANNER_NAME,
                answer=f"Planning agent is processing the request from the user demand..."
            )
        )
    
        chat_history = await get_chat_history(ctx=ctx)

        planning_agent = agent_registry.spawn(
            name=PLANNER_NAME,
            llm=app_state.llm_instance[PLANNER_NAME]
        )

        handler: WorkflowHandler = planning_agent.run(user_msg=message, chat_history = chat_history)

        async for event in handler.stream_events():
            if isinstance(event, StopEvent): continue
            ctx.write_event_to_stream(event)
           
        
        agent_output: AgentOutput = await handler
        
        full_response = WorkersPlan.model_validate(agent_output.structured_response)

        plan_summary = full_response.plan_summary
        plan_detail = full_response.plan_detail
        reason = full_response.reason

        plan_detail_str = json.dumps([p.model_dump(mode="json") for p in plan_detail], indent=2)

        output_planner = PLANNER_AGENT_OUTPUT.format(
            user_request=user,
            plan_summary=plan_summary
        )

        ctx.write_event_to_stream(
            PlanningAgentEvent(
                reason=f"My reasoning though would be: {reason}",
                plan_summary=plan_summary,
                plan_detail=[p.model_dump(mode="json") for p in plan_detail]
            )
        )

        messages = [
            ChatMessage(role=MessageRole.ASSISTANT, content=output_planner, additional_kwargs={"ent": "planner"}),
            ChatMessage(role=MessageRole.ASSISTANT, content=plan_detail_str),
            ChatMessage(role=MessageRole.ASSISTANT, content=reason)
        ]

        await set_add_message_to_chat_history(ctx=ctx, chat_messages=messages)
        await set_worker_plan(ctx=ctx, worker_plan=full_response)

        return PlanProposedEvent(
            user_msg=user,
            agent_response=output_planner,
            worker_plan=full_response
        )

    @step
    async def execute_approved_plan(
        self,
        ctx: Context,
        ev: PlanProposedEvent,
        app_state: Annotated[Appstate, Resource(get_app_state)],
        agent_registry: Annotated[AgentRegistry, Resource(get_global_agent_registry)]
    ) -> FinalResponseEvent: 
        
        user_id, list_video_ids = await get_state_from_user(ctx=ctx)
        plan: WorkersPlan = ev.worker_plan

        sub_orchestrate_state = create_sub_orchestrator_initial_state(user_query=f"Passing message from agent: {ev.agent_response}\n\n The main user demand: {ev.user_msg}")
        async with ctx.store.edit_state() as state:
            state[SUB_ORCHESTRATOR_STATE_KEY] = sub_orchestrate_state

        await set_worker_plan_sub(ctx=ctx, worker_plan=plan)
        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer="Crafting the plan and generating worker agents for your request..."
            )    
        )

        list_of_worker_functool: list[FunctionTool] = []

        for plan_blueprint in plan.plan_detail:
            worker_llm = app_state.llm_instance[SUB_WORKER_NAME]

            findings_schema = create_schema_from_function(
                name=f"sub_orc_update_findings_{plan_blueprint.name}",
                func=sub_orchestration_state_update_findings,
                ignore_fields=["parent_ctx"],
            )
            tool_results_schema = create_schema_from_function(
                name=f"sub_orc_update_tool_results_{plan_blueprint.name}",
                func=sub_orchestration_state_update_tool_results,
                ignore_fields=["parent_ctx"],
            )

            additional_tools = [
                FunctionTool.from_defaults(
                    async_fn=sub_orchestration_state_update_findings,
                    fn_schema=findings_schema,
                    name=f"update_findings_{re.sub(r'[^a-zA-Z0-9_]+', '_', plan_blueprint.name)}",
                    partial_params={
                        'parent_ctx': ctx,
                        'worker_name': plan_blueprint.name,
                    },
                ),
                FunctionTool.from_defaults(
                    async_fn=sub_orchestration_state_update_tool_results,
                    fn_schema=tool_results_schema,
                    name=f"update_tool_results_{re.sub(r'[^a-zA-Z0-9_]+', '_', plan_blueprint.name)}",
                    partial_params={
                        'parent_ctx': ctx,
                        'worker_name': plan_blueprint.name,
                    }
                )
            ]

            worker_rename_func = re.sub(r"[^a-zA-Z0-9_]+", "_", plan_blueprint.name)
            worker_schema_fn = create_schema_from_function(
                name=f"run_worker_{worker_rename_func}",
                func=run_worker_function_as_tools,
                ignore_fields=[
                    "ctx",
                    "parent_ctx",
                    "agent_name",
                    "user_message",
                    "additional_tools",
                    "llm",
                    "user_id",
                    "list_video_ids",
                    "verbose",
                    "timeout",
                ],
            )
            agent_as_tool = FunctionTool.from_defaults(
                async_fn=run_worker_function_as_tools,
                fn_schema=worker_schema_fn,
                name=f"run_{worker_rename_func}",
                partial_params={
                    "ctx": ctx,
                    "parent_ctx": ctx,
                    "agent_name": plan_blueprint.name,
                    "user_message": plan_blueprint.task,
                    "additional_tools": [],
                    "llm": worker_llm,
                    "user_id": user_id,
                    "list_video_ids":list_video_ids
                }
            )
            list_of_worker_functool.append(agent_as_tool)



        synth_schema = create_schema_from_function(
            name="sub_orc_synthesize_final_answers",
            func=sub_orchestration_state_synthesize_final_answers,
            ignore_fields=["parent_ctx"],
        )
        view_results_schema = create_schema_from_function(
            name="sub_orc_view_agent_results",
            func=sub_orchestration_state_view_results_from_agent_tools,
            ignore_fields=["parent_ctx"],
        )

        full_orchestration_tool: list[FunctionTool] = [
                FunctionTool.from_defaults(
                    async_fn=sub_orchestration_state_synthesize_final_answers,
                    fn_schema=synth_schema,
                    partial_params={'parent_ctx': ctx},
                ),
                FunctionTool.from_defaults(
                    async_fn=sub_orchestration_state_view_results_from_agent_tools,
                    fn_schema=view_results_schema,
                    partial_params={'parent_ctx': ctx},
                )
            ]  + list_of_worker_functool
        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer=f"Starting orchestration with {len(list_of_worker_functool)} workers"
            )
        )
        
        orchestration_agent = agent_registry.spawn(
            name=ORCHESTRATION_NAME,
            llm=app_state.llm_instance[ORCHESTRATION_NAME],
            tools=full_orchestration_tool
        )
        msg = f"""
            Execute this query: {ev.user_msg}. Use tools based STRICTLY on this plan:
            {ev.worker_plan._get_string()}
        """        
        handler: WorkflowHandler = orchestration_agent.run(
            user_msg=msg,
        )

        async for event in handler.stream_events():
            if isinstance(event, StopEvent):
                break
            ctx.write_event_to_stream(event)

        

        agent_output: AgentOutput = await handler
        
        final_response = ChatMessage( content=agent_output.response.content, 
                                     role =agent_output.response.role,
                                     additional_kwargs={
                                         "tools": [t.name for t in agent_output.tool_calls] or [],
                                         "ent": agent_output.current_agent_name
                                         }
                                    )
        
        final_response = cast(ChatMessage, agent_output.response)
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[final_response])
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer="Orchestration completed, preparing final response"
            )
        )
        
        return FinalResponseEvent(
            passing_messages=[final_response]
        )
    

    
