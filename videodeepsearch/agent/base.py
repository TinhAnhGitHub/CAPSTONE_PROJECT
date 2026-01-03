"""
Containing the template, the configuration, and the agent factory for register and get the agent based on name
"""

from typing import Any, Callable, Dict, TypeVar
from functools import wraps

from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.agent import BaseWorkflowAgent, FunctionAgent
from pydantic import BaseModel, Field, ConfigDict

class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(..., description="The name of the agent")
    description: str 
    system_prompt: str
    llm: LLM
    tools: list[BaseTool] | None = None
    state_prompt: str  | None = None
    streaming: bool = True
    output_cls: type[BaseModel] | None = None
    extra_kwargs: dict[str, Any] ={}
    type_of_agent: type[BaseWorkflowAgent] = FunctionAgent


class AgentFactory:
    def create_agent(
        self,
        config: AgentConfig,
    ) -> BaseWorkflowAgent:
    

        llm = config.llm
        agent_kwargs = {
            "name": config.name,
            "description": config.description,
            "system_prompt": config.system_prompt,
            "llm": llm,
            "streaming": config.streaming,
        }
        if config.tools is not None:
            agent_kwargs['tools'] = config.tools #type:ignore
            
        if config.state_prompt is not None:
            agent_kwargs["state_prompt"] = config.state_prompt
        
        if config.output_cls is not None:
            agent_kwargs["output_cls"] = config.output_cls #type:ignore
        
        agent_kwargs.update(config.extra_kwargs)

        return config.type_of_agent(**agent_kwargs) #type:ignore


class AgentRegistry:
    def __init__(self):
        self.name2ag_conf: Dict[str, Callable[..., AgentConfig]] = {}
        self.factory = AgentFactory()

    def register(
        self,
        name: str,
        func: Callable[..., AgentConfig]
    ):
        if name in self.name2ag_conf:
            return
        self.name2ag_conf[name] = func
        
    def spawn(
        self,
        name:str,
        **kwargs
    ):
        func = self.name2ag_conf[name]
        agent_config = func(**kwargs)
        return self.factory.create_agent(config=agent_config)


_global_registry = AgentRegistry()


def register_agent(name: str):
    def decorator(func: Callable[..., AgentConfig]) -> Callable[..., AgentConfig]:
        _global_registry.register(name=name, func=func)
        return func        
    return decorator

def get_global_agent_registry() -> AgentRegistry:
    return _global_registry
