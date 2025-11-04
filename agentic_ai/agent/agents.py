from typing import Callable, Any, Annotated, Literal
from pydantic import BaseModel, Field
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, CodeActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import AgentStream, AgentOutput
from .state import AgentState, PlannerState, GreetingState
from .events import (
    UserInputEvent,
    FinalResponseEvent,
    AgentProgressEvent,
    AgentResponse,
    PlannerInputEvent,
    PlanProposedEvent,
    ExecutePlanEvent,
    AllWorkersCompleteEvent,
    StopEvent,
    StartEvent
)
from .prompts import (
    GREETING_PROMPT,
    PLANNER_PROMPT,
    ORCHESTRATOR_PROMPT,
    WORKER_AGENT_PROMPT_TEMPLATE,
    GREETING_PROMPT_FUNC
)

from .schema import (
    WorkerBluePrint,
    WorkersPlan
)
from llama_index.core.workflow import step
from llama_index.core.agent.workflow.workflow_events import AgentWorkflowStartEvent

import json, re
class Planner(ReActAgent):
    def __hash__(self):
        return hash(self.name)
    def __init__(self, llm : LLM,name = "", description = "", system_prompt = "", tools = []):
        super().__init__(
                llm = llm, 
                name = name, 
                description= description, 
                system_prompt=system_prompt, 
                tools = tools)



class Greeter( FunctionAgent):
    def __hash__(self):
        return hash(self.name)
    def extract_response(self, agent_output: AgentOutput) -> str:
        response = agent_output.response
        
        raw_text = response.blocks[0].text
        
        match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', raw_text)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)
            print("Extracted JSON data:", data)
            return GreetingState.model_validate(data)
            
    def __init__(self, llm : LLM,name = "", description = "", system_prompt = "", tools = [],structured_output_fn = None):
        super().__init__(
                llm = llm, 
                name = name, 
                description= description, 
                system_prompt=system_prompt,
                tools = tools,
                output_cls=structured_output_fn)   
        
class Orchestrator(FunctionAgent):
    def __hash__(self):
        return hash(self.name)
    
class WorkerAgent(CodeActAgent):
    def __hash__(self):
        return hash(self.name)
    
class Planner(ReActAgent):
    def __hash__(self):
        return hash(self.name)

def create_greeting_agent(
    llm: LLM,
    name="Greeting Agent",
    description="An agent to receive user querry",
    output_cls : type[BaseModel] = None
):


    llm = llm.as_structured_llm(output_cls) if output_cls else llm

    return Greeter(
        name=name,
        description=description,
        system_prompt=GREETING_PROMPT,
        llm=llm,
        structured_output_fn=output_cls,
        tools=[]
    )


def create_planner_agent(
    llm: LLM,
    registry_tools: list[BaseTool],
    name="",
    description="",
) -> ReActAgent:
    """
    """

    async def sketch_plan(
        ctx: Context[AgentState],
        plan_description: str,
        plan_detail: WorkersPlan | None = None
    ):
        async with ctx.store.edit_state() as state:
            state.planner_state = PlannerState(
                    plan_description=plan_description,
                    plan=plan_detail
                )


        return "Plan sketch steps"
    
    return Planner(
        name=name,
        description=description,
        system_prompt=PLANNER_PROMPT,
        llm=llm,
        tools=registry_tools + [sketch_plan],
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
            
            state["state"]["workers_spawned"] = [w.name for w in worker_specs.plan]
        
        return f"Spawned {len(worker_specs.plan)} worker agents"
    
    async def consolidate_results(
        ctx: Context,
        final_answer: str
    ) -> str:
        async with ctx.store.edit_state() as state:
            state["state"]["final_answer"] = final_answer
            state["state"]["workflow_complete"] = True
        
        return final_answer
    
    return Orchestrator(
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

    return WorkerAgent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        llm=llm,
        tools=tools,
        code_execute_fn=code_execute_fn
    )

def create_consolidation_agent(
    llm: LLM,
    name="Consolidator Agent",
    description="Combines and summarizes all worker outputs into a final answer",
):
    return Planner(
        name=name,
        description=description,
        system_prompt="You merge, summarize, and synthesize worker agent results.",
        llm=llm,
        structured_output_fn=FinalResponseEvent
    )

        
        
    


        

