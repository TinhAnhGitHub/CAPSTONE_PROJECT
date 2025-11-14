from llama_index.core.llms import LLM
from llama_index.core.agent import FunctionAgent
from .prompt import GREETING_SYSTEM_CONTEXT
from .schema import NextAgentDirective
from videodeepsearch.agent.base import register_agent, AgentConfig
from videodeepsearch.agent.worker.custom import CustomFunctionAgent
from videodeepsearch.tools.type.registry import get_registry_tools


GREETER_NAME = "GREETER_AGENT"

# pass by AppState()
@register_agent(GREETER_NAME)
def create_planner_config(
    llm: LLM
) -> AgentConfig:
    config = AgentConfig(
        name=GREETER_NAME,
        description="You are the Greeting Agent in a Video Understanding System",
        system_prompt=GREETING_SYSTEM_CONTEXT,
        llm=llm,
        tools=get_registry_tools(), #type: ignore
        output_cls=NextAgentDirective,
        type_of_agent=CustomFunctionAgent,
        streaming=False
    )
    return config