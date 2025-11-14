from llama_index.core.llms import LLM
from videodeepsearch.agent.base import register_agent, AgentConfig
from videodeepsearch.tools.type.registry import get_registry_tools
from .schema import WorkersPlan
from .prompt import PLANNER_PROMPT, PLANNER_DESCRIPTION
from videodeepsearch.agent.worker.custom import CustomFunctionAgent


PLANNER_NAME = "PLANNER_AGENT"

@register_agent(PLANNER_NAME)
def create_planner_config(
    llm: LLM,
) -> AgentConfig:
    config = AgentConfig(
        name=PLANNER_NAME,
        description=PLANNER_DESCRIPTION,
        system_prompt=PLANNER_PROMPT,
        llm=llm,
        tools=get_registry_tools(), #type:ignore
        output_cls=WorkersPlan,
        type_of_agent=CustomFunctionAgent,
        streaming=True
    )
    return config

