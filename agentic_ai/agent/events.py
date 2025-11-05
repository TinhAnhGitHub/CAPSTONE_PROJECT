from typing import Any
from pydantic import Field
from llama_index.core.workflow import StartEvent, StopEvent, Event
from llama_index.core.llms import ChatMessage

from .schema import WorkersPlan


class UserInputEvent(StartEvent):
    input: str
    chat_history : list[ChatMessage] = []



class FinalResponseEvent(StopEvent):
    response: str 

class PlannerInputEvent(Event):
    user_msg: str
    planner_demand: str
    


class PlanProposedEvent(Event):
    user_msg : str
    agent_response: str
    plan_summary: str
    plan_detail: WorkersPlan



# Progress and streaming events
class AgentProgressEvent(StopEvent):
    agent_name: str
    answer: Any


class AgentResponse(Event):
    agent_name: str
    answer: str




class ExecutePlanEvent(Event):
    plan: WorkersPlan
    plan_description: str
    user_msg: str
    agent_response: str


class AllWorkersCompleteEvent(Event):
    user_msg: str
    result: list


    