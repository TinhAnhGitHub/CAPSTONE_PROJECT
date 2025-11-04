from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field

from typing import Optional, Any

from .schema import WorkersPlan 



class PlannerState(BaseModel):
    plan: Optional[WorkersPlan] = None
    plan_description: str | None = Field('Not yet planned', description="the plan summary")


class GreetingState(BaseModel):
    """
    This is the State of the greeting agent, where it will leave its thought here
    """
    choose_next_agent: str | None = Field(None, description="Choosing the next agent to run, or None, meaning passing the result to the user")
    reason: str = Field('', description="Why did you make this decision")
    passing_message: str | None = Field('', description="The message that the agent want another agent to know")

    def __str__(self):
        return f"GreetingState(choose_next_agent={self.choose_next_agent}, reason={self.reason}, passing_message={self.passing_message})"

class OrchestratorState(BaseModel):
    """
    This is the State of the Orchestator agent
    """
    agent_list: str | None = Field(None, description="Choosing the next agent to run, or None, meaning passing the result to the user")
    reason: str = Field('', description="Why did you make this decision")
    passing_message: str | None = Field('', description="The message that the agent want another agent to know")
    

class AgentState(BaseModel):
    memory: Optional[Any] = None  
    chat_history: list[ChatMessage] = Field(default_factory=list, description="The persistent chat history")
    greeting_state: GreetingState = Field(default_factory=lambda: GreetingState()) #type:ignore
    orchestrator_state: OrchestratorState = Field(default_factory=lambda: OrchestratorState())
    planner_state: PlannerState = Field(default_factory=lambda: PlannerState())#type:ignore

    

