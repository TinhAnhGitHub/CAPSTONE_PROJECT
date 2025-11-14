from pydantic import BaseModel, Field

from llama_index.core.llms import ChatMessage



ORCHESTRATOR_STATE_KEY = 'ORCHESTRATOR_STATE'
class OrchestratorState(BaseModel):
    user_id: str | None = Field(None)
    list_video_ids: list[str] | None = Field(None)
    user_chat_history: list[ChatMessage] = Field(default_factory=list, description="The persistent chat history") 


def create_orchestrator_initial_state() -> dict:
    state = OrchestratorState() # type:ignore
    return state.model_dump(mode='json')
    
