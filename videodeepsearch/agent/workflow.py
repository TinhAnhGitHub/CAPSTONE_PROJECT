"""
The main agent that we will call uponn
"""
from typing import Annotated, cast
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
from llama_index.core.base.llms.types import ThinkingBlock, TextBlock


from videodeepsearch.core.app_state import get_app_state, Appstate


from videodeepsearch.agent.worker import GREETER_NAME, PLANNER_NAME, SUB_WORKER_NAME, ORCHESTRATION_NAME, FINAL_RESPONSE_AGENT

from videodeepsearch.agent.worker.greeter.schema import NextAgentDirective
from videodeepsearch.agent.worker.subworker.agent import run_worker_function_as_tools

from videodeepsearch.agent.worker.suborchestrate.agent import ORCHESTRATION_NAME

from videodeepsearch.tools.type.registry import get_registry_tools


from videodeepsearch.agent.state.full_orc_state_tool import (
    set_add_message_to_chat_history,
    set_state_from_user,
    get_chat_history,
    get_state_from_user
)



from videodeepsearch.agent.state.sub_orc_state_tool import(
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
)

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
        print(f"{ev.chat_history=}")
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=ev.chat_history)
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[user_message])

        
        greeting_agent = agent_registry.spawn(
            name=GREETER_NAME,
            llm=app_state.llm_instance[GREETER_NAME]
        )  

        chat_history = await get_chat_history(ctx=ctx)
        print(f"{chat_history=}")
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
        chat_history.append(ChatMessage(role=MessageRole.ASSISTANT, content=format_response))
        
        await set_add_message_to_chat_history(
            ctx=ctx,
            chat_messages=[ChatMessage(role=MessageRole.ASSISTANT, content=format_response)]
        )

        passing_messages = [
            ChatMessage(role=MessageRole.ASSISTANT, content=passing_message)
        ]

        if next_agent == "planner": 
            return PlannerInputEvent(
                user_msg=ev.user_demand,
                planner_demand=passing_message
            )
        
        elif next_agent == "orchestrator":
            return PlanProposedEvent(
                user_msg=ev.user_demand,
                agent_response=passing_message,
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

        final_agent_output = AgentOutput(
            response=final_response_message,
            structured_response=None,
            current_agent_name=FINAL_RESPONSE_AGENT,
            tool_calls=[],
            retry_messages=[]
        )

        ctx.write_event_to_stream(
            final_agent_output
        )
        
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[final_response_message])

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
    ) -> FinalResponseEvent:
    
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
        
        full_chat_resp = agent_output.response
        full_response_text = full_chat_resp.content
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=PLANNER_NAME,
                answer=full_response_text
            )
        )
        await set_add_message_to_chat_history(ctx=ctx, chat_messages=[full_chat_resp])

        return FinalResponseEvent(
            passing_messages=[full_chat_resp]
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
        chat_history = await get_chat_history(ctx=ctx)

        sub_orchestrate_state = create_sub_orchestrator_initial_state(user_query=f"Passing message from agent: {ev.agent_response}\n\n The main user demand: {ev.user_msg}")

        async with ctx.store.edit_state() as state:
            state[SUB_ORCHESTRATOR_STATE_KEY] = sub_orchestrate_state

        worker_llm = Appstate().llm_instance[SUB_WORKER_NAME]
        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer="Crafting the plan and generating worker agents for your request..."
            )    
        )

        worker_schema_fn = create_schema_from_function(
            name=f"run_worker_as_tool",
            func=run_worker_function_as_tools,
            ignore_fields=[
                "ctx",
                "parent_ctx",
                "user_message",
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
            partial_params={
                "ctx": ctx,
                "parent_ctx": ctx,
                "user_message": ev.user_msg,
                "llm": worker_llm,
                "user_id": user_id,
                "list_video_ids":list_video_ids
            }
        )

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
            ]  + [agent_as_tool] + get_registry_tools()
        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer=f"Starting orchestration"
            )
        )
        
        orchestration_agent = agent_registry.spawn(
            name=ORCHESTRATION_NAME,
            llm=app_state.llm_instance[ORCHESTRATION_NAME],
            tools=full_orchestration_tool
        )

        print(f"{chat_history=}")
                
        handler: WorkflowHandler = orchestration_agent.run(
            user_msg=ev.user_msg,
            chat_history=chat_history
        )

        async for event in handler.stream_events():
            if isinstance(event, StopEvent):
                continue
            ctx.write_event_to_stream(event)

        

        agent_output: AgentOutput = await handler
        final_response = cast(ChatMessage, agent_output.response)

        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=ORCHESTRATION_NAME,
                answer="Orchestration completed, preparing final response"
            )
        )
        
        return FinalResponseEvent(
            passing_messages=[final_response]
        )
    

    
