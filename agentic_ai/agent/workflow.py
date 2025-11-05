"""
The main agent that we will call uponn
"""

from typing import List, Any, Annotated, cast
import asyncio

from llama_index.core.workflow import Workflow, step, Context
from llama_index.core.llms import LLM, ChatMessage
from llama_index.core.tools import BaseTool
from llama_index.core.agent.workflow import AgentStream
from llama_index.core.agent.workflow import ToolCall, AgentOutput
from llama_index.core.workflow.handler import WorkflowHandler # type:ignore



if __name__=="__main__":
    from state import AgentState

    from events import (
        UserInputEvent,
        FinalResponseEvent,
        AgentProgressEvent,
        AgentResponse,
        PlannerInputEvent,
        PlanProposedEvent,
        ExecutePlanEvent,
        AllWorkersCompleteEvent
    )

    from agents import (
        create_greeting_agent,
        create_planner_agent,
        create_orchestrator_agent,
        create_worker_agent,
        create_consolidation_agent
    )

    from schema import (
        WorkerBluePrint,
        WorkersPlan
    )
else:
    from .state import *

    from .events import (
        UserInputEvent,
        FinalResponseEvent,
        AgentProgressEvent,
        AgentResponse,
        PlannerInputEvent,
        PlanProposedEvent,
        ExecutePlanEvent,
        AllWorkersCompleteEvent,
        StopEvent
    )

    from .agents import (
        create_greeting_agent,
        create_planner_agent,
        create_orchestrator_agent,
        create_worker_agent,
        create_consolidation_agent
    )

    from .schema import (
        WorkerBluePrint,
        WorkersPlan
    )


async def _stream_event(handler: WorkflowHandler, ctx: Context[AgentState], agent_name: str) -> str:
    message_stream_list = []
    print("Tinh Anh Gay")
    async for event in handler.stream_events(True):
        print(">>> EVENT:\n", vars(event))
        if isinstance(event, StopEvent):
            message_stream = event.response if hasattr(event, 'response') else str(event)
            message_stream_list.append(message_stream)

            ctx.write_event_to_stream(
                AgentResponse(
                    agent_name=agent_name,
                    message=''.join(message_stream_list)
                )
            )
            break
    return ''.join(message_stream_list)


from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent
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
        logger,
        timeout: float = 600.0,
        verbose: bool = True
    ):
        super().__init__(timeout=timeout, verbose=verbose)
        
        self.llm = llm
        self.context_tools = context_tools
        self.all_tools = all_tools
        
        self.greeting_agent = create_greeting_agent(llm=llm, output_cls=GreetingState)
        self.orchestrator_agent = create_orchestrator_agent(llm=llm, all_tools=all_tools, output_cls= OrchestratorState)
        self.consolidator = create_consolidation_agent(llm)
        self.planning_agent = create_planner_agent(llm=self.llm, registry_tools=self.context_tools, output_cls=PlannerState)

        self.logger = logger

        print("Greeting agent", type(self.greeting_agent))
        print("Orc agent", type(self.orchestrator_agent))
        print("Const agent", type(self.consolidator))
        print("Planning agent", type(self.planning_agent))


    @step
    async def handle_greeting(
        self,
        ctx: Context[AgentState],
        ev: UserInputEvent
    ) -> FinalResponseEvent | PlannerInputEvent:

        
        chat_history = ev.chat_history.copy()
        chat_history.append(
            ChatMessage(role='user', content=ev.input)
        )
        user_message = ev.input
        # print(" ================= Start greeting agent ================= \n")
        agent_output =await self.greeting_agent.run( # FunctionAgent
                        user_msg=user_message,
                        chat_history=chat_history,
                    )

        # print(f" ================= Check greeting agent: {type(full_response)} ================= \n",vars(agent_output))
        full_response = self.greeting_agent.extract_response(agent_output)

        next_agent = full_response.choose_next_agent
        reason = full_response.reason
        passing_message = full_response.passing_message

        print(f"Next agent: {next_agent}, reason: {reason}, passing message: {passing_message}")
        async with ctx.store.edit_state() as state:
           state.greeting_state.choose_next_agent = next_agent
           state.greeting_state.reason = reason
           state.greeting_state.passing_message = passing_message
           state.chat_history.append(
                ChatMessage(role="assistant", content=str(f"{reason=}\n{passing_message=}"))
            )
        
        if next_agent == "planner": 

            return PlannerInputEvent(
                user_msg=user_message,
                planner_demand=passing_message
            )

        else:
            return FinalResponseEvent(
                response=str(full_response)
            )
    
            
    
    @step
    async def planning(
        self,
        ctx: Context[AgentState],
        ev: PlannerInputEvent
    ) -> PlanProposedEvent:
        

        user = ev.user_msg
        message = ev.planner_demand

        message = f"The original user message: {user}. Instruction of the greeting agent. {message}"
        print("User message for planner:", user)
        state = await ctx.store.get_state()
        chat_history = state.chat_history
        
        print(" ================= Start planner ================= \n")
        agent_output = await self.planning_agent.run(user_msg=message, 
                                                     chat_history = chat_history)
        
        full_response = self.greeting_agent.extract_response(agent_output)
        # full_response = await _stream_event(handler=agent_output, ctx=ctx, agent_name=self.planning_agent.name)
        print(" ================= Check planner ================= \n",type(full_response))

        plan_description = full_response.plan_description
        plan_detail = full_response.plan
        print(f"Plan detail: {plan_detail}")

        async with ctx.store.edit_state() as state:
           state.planner_state.plan_description = plan_description
           state.planner_state.plan = plan_detail
           state.chat_history.append(
                ChatMessage(role="assistant", content=str(f"{plan_description=}"))
            )
        return PlanProposedEvent(
            user_msg=user,
            agent_response=full_response,
            plan_detail=plan_detail,
            plan_summary=plan_description
        )
    

        
    @step
    async def execute_approved_plan(
        self,
        ctx: Context[AgentState],
        ev: PlanProposedEvent
    ) -> AllWorkersCompleteEvent: # fix bug
        ctx.write_event_to_stream(
            AgentResponse(
                agent_name="Orchestrator Agent",
                message="Spawning worker agents"
            )
        )

        plan = ev.plan


        def create_code_executor_for_worker(worker_tools: list[BaseTool]):
            """
            Creates a restricted async code executor for worker agents.
            Exposes only allowed tools in a controlled environment.
            """

            async def executor(code: str) -> Any:
                allowed_tools = {tool.name: tool.fn for tool in worker_tools}

                safe_globals = {"__builtins__": {"print": print, "range": range, "len": len}}
                safe_locals = allowed_tools.copy()

                async def _run_async_code():
                    try:
                        exec(
                            f"async def __worker_fn__():\n"
                            + "\n".join(f"    {line}" for line in code.splitlines()),
                            safe_globals,
                            safe_locals,
                        )
                        return await safe_locals["__worker_fn__"]()
                    except Exception as e:
                        return f"Execution error: {type(e).__name__}: {e}"

                try:
                    return await _run_async_code()
                except Exception as e:
                    return f"Worker runtime error: {type(e).__name__}: {e}"

            return executor

        
        async def run_worker(idx: int, blueprint: WorkerBluePrint, ctx: Context | None = None):
            worker_name = blueprint.name


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
                    user_msg=blueprint.task,
                    ctx=ctx 
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
            user_msg = ev.user_msg,
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
                agent_name=self.consolidator.name,
                message="Consolidating results..."
            )
        )
        con = self.consolidator.name
        handler = con.run(user_msg=f"{ev}")

        full_response = await _stream_event(handler=handler, ctx=ctx, agent_name=con.name)
        
        return FinalResponseEvent(response=full_response)



    