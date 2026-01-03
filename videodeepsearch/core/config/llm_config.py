from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from google.genai import types
from typing import Literal
from videodeepsearch.agent.definition import GREETER_AGENT, PLANNER_AGENT, ORCHESTRATOR_AGENT, WORKER_AGENT

VALID_MODELS = {'gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro', 'gemini-3-flash-preview', 'gemini-3-pro-preview'}

class BaseGeminiLLMConfig(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file='/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/videodeepsearch/.env',
        extra='ignore'
    )

    model_name: Literal['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro', 'gemini-3-flash-preview', 'gemini-3-pro-preview']
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    thinking: bool
    thinking_budget: int | None = None
    generation_config: types.GenerateContentConfig | None = None


    @model_validator(mode='after')
    def validate_llm_config(self):
        if self.model_name not in VALID_MODELS:
            raise ValueError(
                f"Invalid model name '{self.model_name}'. Must be one of {VALID_MODELS}."
            )

      
        if self.thinking and self.thinking_budget is None:
            raise ValueError(
                "If 'greeting_llm_thinking' is True, 'greeting_llm_thinking_budget' must be set."
            )

        if self.temperature is not None and not (0 <= self.temperature <= 2):
            raise ValueError("Temperature must be between 0 and 2.")

        if self.top_p is not None and not (0 <= self.top_p <= 1):
            raise ValueError("top_p must be between 0 and 1.")

        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative.")


        thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=self.thinking_budget
        ) if self.thinking else None

        self.generation_config = types.GenerateContentConfig(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            thinking_config=thinking_config
        )
        return self

class GreetingLLMConfig(BaseGeminiLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix="GREETING_LLM_",
        case_sensitive=False
    )
    agent_name: str = GREETER_AGENT

class PlannerLLMConfig(BaseGeminiLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix="PLANNER_LLM_",
        case_sensitive=False
    )
    agent_name: str = PLANNER_AGENT


class SubOrchestratorLLMConfig(BaseGeminiLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix="SUB_ORCHESTRATOR_LLM_",
        case_sensitive=False
    )
    agent_name: str = ORCHESTRATOR_AGENT

class SubWorkerLLMConfig(BaseGeminiLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix="SUB_WORKER_LLM_",
        case_sensitive=False
    )
    agent_name: str = WORKER_AGENT

class OutputRespLLM(BaseGeminiLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix="OUTPUT_RESP_",
        case_sensitive=False
    )
    agent_name: str = 'FINAL_RESPONSE_AGENT'


greeting_llm_config = GreetingLLMConfig() #type:ignore
planner_llm_config = PlannerLLMConfig() #type:ignore
sub_orchestrator_config = SubOrchestratorLLMConfig() #type:ignore
sub_worker_config = SubWorkerLLMConfig() #type:ignore
output_resp_config= OutputRespLLM()#type:ignore


llm_configs = (
    greeting_llm_config,
    planner_llm_config,
    sub_orchestrator_config,
    sub_worker_config,
    output_resp_config
)

# if __name__ == "__main__":
#     import os

#     os.environ["GREETING_LLM_MODEL_NAME"] = "gemini-2.5-flash"
#     os.environ["GREETING_LLM_THINKING"] = "true"
#     os.environ["GREETING_LLM_THINKING_BUDGET"] = "256"

#     cfg = GreetingLLMConfig() #type:ignore
#     print(cfg.model_dump())
