from llama_index.core.llms import LLM

from app.service.chat import ChatService
from typing import cast, AsyncGenerator

class Agent:
    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm    

    def chat(self, text: str) -> str:

        response = self.llm.complete(text)
        return response.text
    
    async def stream_chat(self, text: str) -> AsyncGenerator[str, None]:
        response_gen = await self.llm.astream_complete(text)
        async for event in response_gen:
            if event.delta is not None:
                yield event.delta


