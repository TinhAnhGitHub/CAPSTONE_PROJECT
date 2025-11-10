from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool

from videodeepsearch.agent.base import register_agent, AgentConfig
from videodeepsearch.agent.worker.custom import CustomFunctionAgent

from .prompt import ORCHESTRATOR_SYSTEM_PROMPT

ORCHESTRATION_NAME = 'SUBORCHESTRATION_AGENT'

@register_agent(ORCHESTRATION_NAME)
def create_orchestration_config(
    llm:FunctionCallingLLM,
    tools: list[BaseTool]
) -> AgentConfig:
    config = AgentConfig(
        name=ORCHESTRATION_NAME,
        description="",
        llm=llm,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        streaming=True,
        tools=tools,
        type_of_agent=CustomFunctionAgent
    )
    return config