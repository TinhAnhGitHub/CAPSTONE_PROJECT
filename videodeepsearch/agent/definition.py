from llama_index.core.llms.function_calling import FunctionCallingLLM
from videodeepsearch.agent.base import register_agent, AgentConfig
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent

from .custom import WorkerAgent, StreamingFunctionAgent
from .prompt import GREETING_SYSTEM_CONTEXT, PLANNER_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT
from llama_index.core.tools import FunctionTool
from videodeepsearch.tools.base.registry import get_registry_tools


GREETER_AGENT = 'GREETER_AGENT'
PLANNER_AGENT = 'PLANNER_AGENT'
ORCHESTRATOR_AGENT = 'ORCHESTRATOR_AGENT'
WORKER_AGENT = 'WORKER_AGENT'


@register_agent(GREETER_AGENT)
def create_greeting_agent(
    llm: FunctionCallingLLM,
    tools: list[FunctionTool]
) -> AgentConfig:
    config = AgentConfig(
        name=GREETER_AGENT,
        description="Greet agent",
        system_prompt=GREETING_SYSTEM_CONTEXT,
        llm=llm,
        tools=tools, #type: ignore
        type_of_agent=StreamingFunctionAgent,
        streaming=True
    )
    return config

@register_agent(PLANNER_AGENT)
def create_planner_agent(
    llm: FunctionCallingLLM,
    tools: list[FunctionTool]
) -> AgentConfig:
    config = AgentConfig(
        name=PLANNER_AGENT,
        system_prompt=PLANNER_PROMPT,
        description="planner agent",
        llm=llm,
        tools=tools, #type:ignore
        type_of_agent=StreamingFunctionAgent,
        streaming=True,
        extra_kwargs={
            'verbose': False,
            'timeout': 3 * 60 # 5 minutes to plan 
        }
    )
    return config

@register_agent(ORCHESTRATOR_AGENT)
def create_orchestrator_agent(
    agent_name: str,
    llm: FunctionCallingLLM,
    tools: list[FunctionTool]
):
    config = AgentConfig(
        name=agent_name,
        llm=llm,
        description="orchestrator agent",
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        streaming=True,
        tools=tools, #type:ignore
        type_of_agent=StreamingFunctionAgent,
         extra_kwargs={
            'verbose': True,
            'timeout': 10 * 60 # 60 minutes orchestrate plan
        }
    )
    return config
    
@register_agent(WORKER_AGENT)
def create_worker_agent(
    agent_name: str ,
    description: str,
    system_prompt: str,
    tools: list[FunctionTool],
    llm: FunctionCallingLLM,
):
    config = AgentConfig(
        name=agent_name,
        description=description,
        system_prompt=system_prompt,
        tools=tools, #type:ignore
        llm=llm,
        extra_kwargs={
            'verbose': False 
        },
        type_of_agent=WorkerAgent
    )
    return config


