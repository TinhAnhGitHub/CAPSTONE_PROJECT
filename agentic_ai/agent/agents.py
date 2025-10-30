from typing import Callable, Any, Annotated
from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, CodeActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context

from .state import AgentState

from .prompts import (
    GREETING_PROMPT,
    PLANNER_PROMPT,
    ORCHESTRATOR_PROMPT,
    WORKER_AGENT_PROMPT_TEMPLATE
)


class WorkerBluePrint(BaseModel):
   name: str = Field(..., description="The specific name of the agents")
   description: str = Field(..., description="The description detail of the agent. This description will refect its jobs, role, and how it should reasoning, perspective, to get the answer, based on the provided tools")
   task: str = Field(..., description="The specific task that it must complete")
   tools: list[str] = Field(..., description="The available tools that the system offer")
   max_iterations: int = Field(3, description="How many times that the agent have to try, before it give up :(  ")


class WorkersPlan(BaseModel):
    plan: list[WorkerBluePrint] = Field(default_factory=list,description="The plan for these agents. Should be around 1-3 agents only. 2 is a sweet spot.")


def create_greeting_agent(
    llm: LLM,
    name="Greeting Agent",
    description="",
):
    async def hand_off_to_agent(
        ctx: Context[AgentState],
        choose_next_agent_name: str,
        reason: str,
        passing_message: str,
    ) -> str:
        """
        Handoff control to the planner agent when the user's request 
        requires video search, retrieval, or complex operations.
        
        Args:
            reason: Why you're handing off to planner
        """
        async with ctx.store.edit_state() as state:
           state.greeting_state.choose_next_agent = choose_next_agent_name
           state.greeting_state.reason = reason
           state.greeting_state.passing_message = passing_message
        
        return f"Handing off to agent {choose_next_agent_name}: {reason}"


    return FunctionAgent(
        name=name,
        description=description,
        system_prompt=GREETING_PROMPT,
        llm=llm,
        tools=[hand_off_to_agent],
        
    )


def create_planner_agent(
    llm: LLM,
    registry_tools: list[BaseTool],
    name="",
    description="",
) -> ReActAgent:
    """
    """
    # async def craft_plan(
    #     ctx: Context,
    #     plan_description: str,
    #     required_tools: list[str],
    #     execution_steps: list[dict[str, str]]
    # ) -> str:
    #     """
    #     Finalize the plan and prepare for the orchestration
    #     Args:
    #         plan_description: High-level description of the plan
    #         required_tools: List of tool names needed
    #         execution_steps: List of steps, each with 'description' and 'tools_needed'
    #     """
    #     async with ctx.store.edit_state() as state:
    #         state['state']['plan'] = {
    #             "description": plan_description,
    #             "required_tools": required_tools,
    #             "steps": execution_steps
    #         }
    #         state["state"]["current_agent"] = "orchestrator"
    #     return f"Plan created with {len(execution_steps)} steps"

    async def sketch_plan(
        ctx: Context[AgentState],
        plan_description: str,
        plan_detail: WorkersPlan | None = None
    ):
        async with ctx.store.edit_state() as state:
            state.planner_state.plan = plan_detail
            state.planner_state.plan_description = plan_description
        return f"Plan sketch steps"
    
    return ReActAgent(
        name=name,
        description=description,
        system_prompt=PLANNER_PROMPT,
        llm=llm,
        tools=registry_tools + [finalize_plan, sketch_plan],
    )

def create_orchestrator_agent(
    llm: LLM,
    all_tools: dict[str, BaseTool]
):
    async def spawn_worker_agents(
        ctx: Context[AgentState],
        reason_why: str,
        worker_specs: WorkersPlan | None = None
    ) -> str:
        async with ctx.store.edit_state() as state:
            state.planner_state.plan = worker_specs
            
            state["state"]["workers_spawned"] = 
        
        return f"Spawned {len(worker_specs.plan)} worker agents"
    
    async def consolidate_results(
        ctx: Context,
        final_answer: str
    ) -> str:
        async with ctx.store.edit_state() as state:
            state["state"]["final_answer"] = final_answer
            state["state"]["workflow_complete"] = True
        
        return final_answer
    
    return FunctionAgent(
        name="OrchestratorAgent",
        description="Orchestrates worker agents to execute the plan",
        system_prompt=ORCHESTRATOR_PROMPT,
        llm=llm,
        tools=[spawn_worker_agents, consolidate_results]
    )

def create_worker_agent(
    llm: LLM,
    name: str,
    description: str,
    tools: list[BaseTool],
    code_execute_fn: Callable
):
    """
    Creates a CodeActAgent worker that can write Python code to use tools
    
    Args:
        llm: Language model
        name: Worker agent name
        description: What this worker does
        tools: Tools available to this worker
        code_execute_fn: Function to execute generated code
    
    Returns:
        Configured CodeActAgent
    """

    system_prompt = WORKER_AGENT_PROMPT_TEMPLATE

    return CodeActAgent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        llm=llm,
        tools=tools,
        code_execute_fn=code_execute_fn
    )


        
        
    


        

