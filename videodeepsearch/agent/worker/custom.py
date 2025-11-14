from typing import List, Sequence
from pydantic import BaseModel
from llama_index.core.agent.workflow import AgentStream
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.base.llms.types import ThinkingBlock, ContentBlock, TextBlock


class CustomFunctionAgent(FunctionAgent):
    async def _get_streaming_response(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
    ) -> ChatResponse:
        chat_kwargs = {
            "chat_history": current_llm_input,
            "tools": tools,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
        }

        if (
            self.initial_tool_choice is not None
            and current_llm_input[-1].role == "user"
        ):
            chat_kwargs["tool_choice"] = self.initial_tool_choice
        
        response = await self.llm.astream_chat_with_tools( #type:ignore
            **chat_kwargs
        )


        last_chat_response = ChatResponse(message=ChatMessage())

        #streaming len
        streaming_token_len = 0

        async for last_chat_response in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )

            # if text block arrive, stop receive the thinking block
            list_of_content_blocks = last_chat_response.message.blocks
            text_block: TextBlock | None = next(
                filter(lambda x: isinstance(x, TextBlock), list_of_content_blocks), None
            )

            thinkink_blocks: list[ThinkingBlock] = list(
                filter(lambda x: isinstance(x, ThinkingBlock), list_of_content_blocks)
            )

            if text_block:
                text = text_block.text  
                text_delta = text[streaming_token_len:]

                ctx.write_event_to_stream(
                    AgentStream(
                        delta=text_delta,
                        response=text, # full repsonse
                        tool_calls=tool_calls or [],
                        raw=raw,
                        current_agent_name=self.name,
                        thinking_delta=None
                    )
                )
                streaming_token_len = len(text)
                continue

            if len(thinkink_blocks) > 0:
                try:

                    thinking_delta = thinkink_blocks[-2] # [thinking delta, thinking full]
                    ctx.write_event_to_stream(
                        AgentStream(
                            delta="", 
                            response="",
                            tool_calls=tool_calls or [],
                            raw=raw,
                            current_agent_name=self.name,
                            thinking_delta=thinking_delta.content
                        )
                    )

                except Exception as e:
                    continue
        return last_chat_response
    

class CustomReActAgent(ReActAgent):
    async def _get_streaming_response(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
    ) -> ChatResponse:
        
        response = await self.llm.astream_chat(
            current_llm_input,
        )
        last_chat_response = ChatResponse(message=ChatMessage())
        streaming_token_len = 0

        async for last_chat_response in response:
            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )

            # if text block arrive, stop receive the thinking block
            list_of_content_blocks = last_chat_response.message.blocks
            text_block: TextBlock | None = next(
                filter(lambda x: isinstance(x, TextBlock), list_of_content_blocks), None
            )#type:ignore

            thinkink_blocks: list[ThinkingBlock] = list(
                filter(lambda x: isinstance(x, ThinkingBlock), list_of_content_blocks)
            )#type:ignore

            if text_block:
                text = text_block.text  
                text_delta = text[streaming_token_len:]

                ctx.write_event_to_stream(
                    AgentStream(
                        delta=text_delta,
                        response=text, # full repsonse
                        tool_calls=tool_calls or [],
                        raw=raw,
                        current_agent_name=self.name,
                        thinking_delta=None
                    )
                )
                streaming_token_len = len(text)
                continue

            if len(thinkink_blocks) > 0:
                try:

                    thinking_delta = thinkink_blocks[-2] # [thinking delta, thinking full]
                    ctx.write_event_to_stream(
                        AgentStream(
                            delta="", 
                            response="",
                            tool_calls=tool_calls or [],
                            raw=raw,
                            current_agent_name=self.name,
                            thinking_delta=thinking_delta.content
                        )
                    )

                except Exception as e:
                    continue
        return last_chat_response
