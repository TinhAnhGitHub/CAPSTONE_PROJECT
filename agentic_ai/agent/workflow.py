"""
The main agent that we will call uponn
"""

from typing import List, Any, Annotated, cast
import asyncio

from llama_index.core.workflow import Workflow, step, Context
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.tools import BaseTool
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.agent.workflow import ToolCall
from llama_index.core.workflow.handler import WorkflowHandler # type:ignore



from .state import AgentState

from .events import (
    UserInputEvent,
    FinalResponseEvent,
    AgentProgressEvent,
    AgentResponse,
    PlannerInputEvent,
    PlanProposedEvent,
    ExecutePlanEvent,
    AllWorkersCompleteEvent
)

from .agents import (
    create_greeting_agent,
    create_planner_agent,
    create_orchestrator_agent,
    create_worker_agent
)

from .agents import (
    WorkerBluePrint,
    WorkersPlan
)

async def _stream_event(handler: WorkflowHandler, ctx: Context[AgentState], agent_name: str) -> str:
    message_stream_list = []
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            message_stream = event.delta
            message_stream_list.append(message_stream)

            ctx.write_event_to_stream(
                AgentResponse(
                    agent_name=agent_name,
                    message=''.join(message_stream_list)
                )
            )
        
    return ''.join(message_stream_list)

class VideoAgentWorkFlow(Workflow):
    """
    Multi-agent orchestration with HIL plan approval
    
    Flow:
        User -> greeting agent -> Planner -> [Plan Review Loop] -> Orchestrator -> Workers -> Response

    The workflow STOPS at plan proposal, waits for user feedback, then continues.
    """

    def __init__(
        self,
        llm: LLM, 
        context_tools: Annotated[list[BaseTool], "A list of functions that expose the tools of the system, so the agent can reason upon"],
        all_tools: Annotated[dict[str, BaseTool], "All the tools related to video search stuff"],
        timeout: float = 600.0,
        verbose: bool = True
    ):
        super().__init__(timeout=timeout, verbose=verbose)
        
        self.llm = llm
        self.context_tools = context_tools
        self.all_tools = all_tools
        
        self.greeting_agent = create_greeting_agent(llm=llm)
        self.orchestrator_agent = create_orchestrator_agent(llm=llm, all_tools=all_tools)
    


    @step
    async def handle_greeting(
        self,
        ctx: Context[AgentState],
        ev: UserInputEvent
    ) -> FinalResponseEvent | PlannerInputEvent:
        
        await ctx.store.set('state', ev.state)
        chat_history = ev.chat_history.copy()
        chat_history.append(
            ChatMessage(role='user', content=ev.user_msg)
        )
        user_message = ev.user_msg

        await ctx.store.set('chat_history', chat_history)

        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=self.greeting_agent.name,
                message="I am reading your request..."
            )
        )

        handler = self.greeting_agent.run(
            user_msg=ev.user_msg,
            chat_history=chat_history
        )

        full_response = await _stream_event(handler=handler, ctx=ctx, agent_name=self.greeting_agent.name)

        next_agent = await ctx.store.get('greeting_state.choose_next_agent')
        reason = await ctx.store.get('greeting_state.reason')
        passing_message = await ctx.store.get('greeting_state.passing_message')


        if next_agent == "planner": 
            chat_history.append(
                ChatMessage(role="assistant", content=str(full_response))
            )

            return PlannerInputEvent(
                user_msg=user_message,
                planner_demand="\n\n".join([reason + passing_message])
            )

        else:
            return FinalResponseEvent(
                response=str(full_response)
            )
    
    async def planning(
        self,
        ctx: Context[AgentState],
        ev: PlannerInputEvent
    ) -> PlanProposedEvent:
        
        

        planning_agent = create_planner_agent(llm=self.llm, registry_tools=self.context_tools)

        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name=planning_agent.name,
                message="Analyzing user request and creating plan for video deep researching..."
            )
        )

        message = f"The original user message: {ev.user_msg}. And here is the instruction of the greeting agent. {ev.planner_demand}"


        handler = planning_agent.run(user_msg=message)

        full_response = await _stream_event(handler=handler, ctx=ctx, agent_name=planning_agent.name)

        
        plan_description = await ctx.store.get('planner_state.plan_description')
        plan_detail = await ctx.store.get('planner_state.plan')

        # chat_history = await ctx.store.get('chat_history') append thougts to the chat history


        return PlanProposedEvent(
            agent_response=full_response,
            plan_detail=plan_detail,
            plan_summary=plan_description
        )
    


    @step
    async def execute_approved_plan(
        self,
        ctx: Context[AgentState],
        ev: ExecutePlanEvent
    ):
        ctx.write_event_to_stream(
            AgentResponse(
                agent_name="Orchestrator Agent",
                message="Spawning worker agents"
            )
        )

        plan = ev.plan
        
        async def run_worker(idx: int, blueprint: WorkerBluePrint, context: Context | None = None):
            worker_name = blueprint.name
            ctx.write_event_to_stream(
                AgentProgressEvent(
                    agent_name=worker_name,
                    message=f"⚙️ Starting task: {blueprint.task}"
                )
            )

            worker_tool_names = blueprint.tools
            worker_tools = [
                self.all_tools[tool_name] for tool_name in worker_tool_names if tool_name in self.all_tools
            ]

            """
            Spawning worker agent env
            """
            code_execute_fn = create_code_executor_for_worker(worker_tools)

            worker = create_worker_agent(
                llm=self.llm,
                name=worker_name,
                description=blueprint.get("description", ""),
                tools=worker_tools,
                code_execute_fn=code_execute_fn
            )

            try:
                result = await worker.run(
                    user_msg=spec.get("task", ""),
                    ctx=ctx # prompt the orhestration agent to prepare the context for each small agent
                )
                ctx.write_event_to_stream(
                    AgentProgressEvent(
                        agent_name=worker_name,
                        message=f"✅ Completed: {spec.get('task', '')}"
                    )
                )
                return {
                    'worker_name': worker_name,
                    'task': blueprint.task,
                    'result': str(result),
                    'success': True
                }

            except Exception as e:
                ...
            
        
        worker_tasks = [
            asyncio.create_task(run_worker(idx, cast(WorkerBluePrint,blueprint))) for idx, blueprint in enumerate(plan)
        ]
        result = await asyncio.gather(*worker_tasks)
        ### prepare the content for the final orchestration

        return AllWorkersCompleteEvent(
            result=result
        )
    

    @step
    async def consolidate_results(
        self, 
        ctx: Context,
        ev: AllWorkersCompleteEvent
    ) -> FinalResponseEvent:
        
        ctx.write_event_to_stream(
            AgentProgressEvent(
                agent_name="",
                message="Consolidating results..."
            )
        )

        # prepare prompt

        # consolidate the final 


    
    
    