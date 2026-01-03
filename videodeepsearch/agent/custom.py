from typing import List, Sequence
from pydantic import BaseModel
from llama_index.core.agent.workflow import AgentStream
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import AsyncBaseTool
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.base.llms.types import ThinkingBlock, TextBlock
from llama_index.core.agent.workflow import ToolCallResult, ToolCall
from llama_index.core.memory import BaseMemory
from llama_index.core.llms.function_calling import FunctionCallingLLM
from videodeepsearch.tools.base.middleware.data_handle import DataHandle
from videodeepsearch.agent.context.worker_context import SmallWorkerContext

class WorkerAgent(FunctionAgent):
    async def _get_streaming_response(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
    ) -> ChatResponse:
        chat_kwargs :dict= {
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

    async def handle_tool_call_results(
        self, ctx: Context, results: List[ToolCallResult], memory: BaseMemory
    ) -> None:
        scratchpad: List[ChatMessage] = await ctx.store.get(
            self.scratchpad_key, default=[]
        )
        for tool_call_result in results:
            async with ctx.store.edit_state() as ctx_state:
                agent_context = ctx_state[self.name]
                small_worker_context = SmallWorkerContext.model_validate(agent_context)
                tool_raw_output = tool_call_result.tool_output.raw_output
                persist_result = small_worker_context.raw_result_store



                if isinstance(tool_raw_output, DataHandle):
                    tool_call = ToolCall(tool_name=tool_call_result.tool_name, tool_kwargs=tool_call_result.tool_kwargs, tool_id=tool_call_result.tool_id)
                    tool_raw_output.tool_used=tool_call
                    persist_result.persist_handle(
                        data_handle=tool_raw_output
                    )
                
                small_worker_context.raw_result_store = persist_result
                ctx_state[self.name] = small_worker_context # persist result
            scratchpad.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )
            if (
                tool_call_result.return_direct
                and tool_call_result.tool_name != "handoff"
            ):
                scratchpad.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
                break
        
        async with ctx.store.edit_state() as ctx_state:
            ctx_state[self.scratchpad_key] = scratchpad
                
class StreamingFunctionAgent(FunctionAgent):
    async def _get_streaming_response(
        self,
        ctx: Context,
        current_llm_input: List[ChatMessage],
        tools: Sequence[AsyncBaseTool],
    ) -> ChatResponse:
        chat_kwargs: dict = {
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