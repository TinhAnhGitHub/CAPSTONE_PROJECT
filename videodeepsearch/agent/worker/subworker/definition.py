from typing import Union, Callable, Awaitable, Sequence, cast, get_type_hints, get_origin, get_args, Annotated, Literal
import uuid
import re
import inspect
from llama_index.core.memory import BaseMemory
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)

from llama_index.core.base.llms.types import ChatResponse
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever
from llama_index.core.workflow import Event
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.base.llms.types import ThinkingBlock, TextBlock
from videodeepsearch.agent.orc_events import AgentDecision
from .prompt import MAKE_DECISION_PROMPT

from llama_index.llms.google_genai import GoogleGenAI
"""
This agent will use the LLM instance from llama_index.llms.google_genai import GoogleGenAI

streaming function achat

    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        generation_config = {
            **(self._generation_config or {}),
            **kwargs.pop("generation_config", {}),
        }
        params = {**kwargs, "generation_config": generation_config}
        next_msg, chat_kwargs = await prepare_chat_params(
            self.model, messages, self.use_file_api, self._client, **params
        )
        chat = self._client.aio.chats.create(**chat_kwargs)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            thoughts = ""
            async for r in await chat.send_message_stream(
                next_msg.parts if isinstance(next_msg, types.Content) else next_msg
            ):
                if candidates := r.candidates:
                    if not candidates:
                        continue

                    top_candidate = candidates[0]
                    if response_content := top_candidate.content:
                        if parts := response_content.parts:
                            content_delta = parts[0].text
                            if content_delta:
                                if parts[0].thought:
                                    thoughts += content_delta
                                else:
                                    content += content_delta
                            llama_resp = chat_from_gemini_response(r)
                            llama_resp.delta = llama_resp.delta or content_delta or ""

                            if content:
                                llama_resp.message.blocks = [TextBlock(text=content)]
                            if thoughts:
                                if llama_resp.message.blocks:
                                    llama_resp.message.blocks.append(
                                        ThinkingBlock(content=thoughts)
                                    )
                                else:
                                    llama_resp.message.blocks = [
                                        ThinkingBlock(content=thoughts)
                                    ]
                            yield llama_resp

            if self.use_file_api:
                await delete_uploaded_files(
                    [*chat_kwargs["history"], next_msg], self._client
                )

        return gen()

def chat_from_gemini_response(
    response: types.GenerateContentResponse,
) -> ChatResponse:
    if not response.candidates:
        raise ValueError("Response has no candidates")

    top_candidate = response.candidates[0]
    _error_if_finished_early(top_candidate)

    response_feedback = (
        response.prompt_feedback.model_dump() if response.prompt_feedback else {}
    )
    raw = {
        **(top_candidate.model_dump()),
        **response_feedback,
    }
    thought_tokens: Optional[int] = None
    if response.usage_metadata:
        raw["usage_metadata"] = response.usage_metadata.model_dump()
        if response.usage_metadata.thoughts_token_count:
            thought_tokens = response.usage_metadata.thoughts_token_count

    if hasattr(response, "cached_content") and response.cached_content:
        raw["cached_content"] = response.cached_content

    additional_kwargs: Dict[str, Any] = {"thought_signatures": []}
    content_blocks = []
    if (
        len(response.candidates) > 0
        and response.candidates[0].content
        and response.candidates[0].content.parts
    ):
        parts = response.candidates[0].content.parts
        for part in parts:
            if part.text:
                if part.thought:
                    content_blocks.append(
                        ThinkingBlock(
                            content=part.text,
                            additional_information=part.model_dump(exclude={"text"}),
                        )
                    )
                else:
                    content_blocks.append(TextBlock(text=part.text))
                additional_kwargs["thought_signatures"].append(part.thought_signature)
            if part.inline_data:
                content_blocks.append(
                    ImageBlock(
                        image=part.inline_data.data,
                        image_mimetype=part.inline_data.mime_type,
                    )
                )
                additional_kwargs["thought_signatures"].append(part.thought_signature)
            if part.function_call:
                if (
                    part.thought_signature
                    not in additional_kwargs["thought_signatures"]
                ):
                    additional_kwargs["thought_signatures"].append(
                        part.thought_signature
                    )
                content_blocks.append(
                    ToolCallBlock(
                        tool_call_id=part.function_call.name or "",
                        tool_name=part.function_call.name or "",
                        tool_kwargs=part.function_call.args or {},
                    )
                )
            if part.function_response:
                # follow the same pattern as for transforming a chatmessage into a gemini message: if it's a function response, package it alone and return it
                additional_kwargs["tool_call_id"] = part.function_response.id
                role = ROLES_FROM_GEMINI[top_candidate.content.role]
                return ChatResponse(
                    message=ChatMessage(
                        role=role, content=json.dumps(part.function_response.response)
                    ),
                    raw=raw,
                    additional_kwargs=additional_kwargs,
                )

    if thought_tokens:
        thinking_blocks = [
            i
            for i, block in enumerate(content_blocks)
            if isinstance(block, ThinkingBlock)
        ]
        if len(thinking_blocks) == 1:
            content_blocks[thinking_blocks[0]].num_tokens = thought_tokens
        elif len(thinking_blocks) > 1:
            content_blocks[thinking_blocks[-1]].additional_information.update(
                {"total_thinking_tokens": thought_tokens}
            )

    role = ROLES_FROM_GEMINI[top_candidate.content.role]
    return ChatResponse(
        message=ChatMessage(
            role=role, blocks=content_blocks, additional_kwargs=additional_kwargs
        ),
        raw=raw,
        additional_kwargs=additional_kwargs,
    )

Therefore, we fix the streaming based on this configuration   

"""






class ToolsOrCodeDecision(BaseModel):
    reason: str = Field(..., description="The reason why you obtain this decision")
    decision: Literal['tools', 'code']


EXECUTE_TOOL_NAME = 'execute'



class WorkerCodeVideoAgent(BaseWorkflowAgent):
    """
    A workflow agent that is specialized in generating code and execute a bunch of tools
    """

    execution_history_key: str = Field(
        ...,
        description="Key for storing execution history in context"
    )

    code_execute_fn: Union[Callable, Awaitable] = Field(
        ...,
        description="The function to execute the code"
    )

    code_act_system_prompt: RichPromptTemplate = Field(
        ...,
        description="System prompt for code generation mode"
    )
    
    allow_parallel_tool_calls: bool = Field(
        default=True,
        description="If True, the agent will call multiple tools in parallel. If False, the agent will call tools sequentially.",
    )


    def __init__(
        self,
        execution_history_key:str,
        code_execute_fn: Union[Callable, Awaitable],
        code_act_system_prompt: RichPromptTemplate,
        name: str,
        description: str,
        system_prompt: str,
        tools: list[Union[BaseTool, Callable]] | None = None,
        tool_retriever: ObjectRetriever | None = None,
        llm: FunctionCallingLLM | None = None,
        **kwargs
    ):
        """
        The initialization of the video agent
        - The code_act_system_prompt and the system prompt will be append together during the function calling
        """
    
        tools = tools or []
        tools.append(  
            FunctionTool.from_defaults(code_execute_fn, name='execute')  # type: ignore
        )
        system_prompt = system_prompt
        super().__init__(
            execution_history_key=execution_history_key,
            code_execute_fn=code_execute_fn,
            code_act_system_prompt=code_act_system_prompt,
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            tool_retriever=tool_retriever,
            can_handoff_to=None,
            llm=llm,
            **kwargs
        )
    

    def _get_tool_fns(self, tools: Sequence[BaseTool]) -> list[Callable]:
        """Get the tool functions while validating that they are valid tools for the CodeActAgent."""
        callables = []
        for tool in tools:
            if (
                tool.metadata.name == "handoff"
                or tool.metadata.name == EXECUTE_TOOL_NAME
            ):
                continue

            if isinstance(tool, FunctionTool):
                if tool.requires_context:
                    raise ValueError(
                        f"Tool {tool.metadata.name} requires context. "
                        "CodeActAgent only supports tools that do not require context."
                    )

                callables.append(tool.real_fn)
            else:
                raise ValueError(
                    f"Tool {tool.metadata.name} is not a FunctionTool. "
                    "CodeActAgent only supports Functions and FunctionTools."
                )

        return callables

    def _get_tool_descriptions(self, tools: Sequence[BaseTool]) -> str:
        """
        Generate tool descriptions for the system prompt using tool metadata.

        Args:
            tools: List of available tools

        Returns:
            Tool descriptions as a string

        """
        tool_descriptions = []

        tool_fns = self._get_tool_fns(tools)
        for fn in tool_fns:
            signature = inspect.signature(fn)
            fn_name: str = fn.__name__
            docstring: str | None = inspect.getdoc(fn)

            tool_description = f"def {fn_name}{signature!s}:"
            if docstring:
                tool_description += f'\n  """\n{docstring}\n  """\n'

            try:
                type_hints = get_type_hints(fn)
            except Exception:
                type_hints = {}
            
            ret_type = type_hints.get('return')

            if ret_type:
                origin = get_origin(ret_type)
                args = get_args(ret_type)
                model_cls = None
            
                if origin is Annotated:
                    model_cls = args[0]
                elif origin in (list, Sequence):
                    model_cls = args[0]
                else:
                    model_cls = ret_type

                if inspect.isclass(model_cls) and issubclass(model_cls, BaseModel):
                    tool_description += "\n  # Return fields:\n"
                    for field_name, field in model_cls.model_fields.items():
                        f_type = field.annotation
                        f_type_name = (
                            f_type.__name__ if hasattr(f_type, "__name__") else str(f_type) #type:ignore
                        )
                        desc = field.description or ""
                        default = (
                            f" (default={field.default!r})"
                            if field.default not in (None, inspect._empty)
                            else ""
                        )
                        tool_description += f"  # - {field_name}: {f_type_name}{default}"
                        if desc:
                            tool_description += f" — {desc}"
                        tool_description += "\n"

            tool_description += "\n  ...\n"
            tool_descriptions.append(tool_description)

        return "\n\n".join(tool_descriptions)

    
    async def _handle_tool_call(
        self,
        ctx: Context,
        current_llm_input: list[ChatMessage],
        tools: Sequence[BaseTool]
    ):
        chat_kwargs = {
            "chat_history": current_llm_input,
            "tools": tools,
            "allow_parallel_tool_calls": self.allow_parallel_tool_calls,
        }

        print(f"TOOL IN WOKERAGENT: {tools}")
        for tool in tools:
            print(tool.metadata)
        response = await self.llm.astream_chat_with_tools(  # type: ignore
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

        tool_calls: list[ToolSelection] = cast(FunctionCallingLLM,self.llm).get_tool_calls_from_response(  # type: ignore
            last_chat_response, error_on_no_tool_call=False
        )
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )

        return last_chat_response.message, tool_calls, raw
    
    def _extract_code_from_response(self, response_text: str) -> str | None:
        """
        Extract code from the LLM response using XML-style <execute> tags.

        Args:
            response_text: The LLM response text

        Returns:
            Extracted code or None if no code found

        """
        execute_pattern = r"<execute>(.*?)</execute>"
        execute_matches = re.findall(execute_pattern, response_text, re.DOTALL)

        if execute_matches:
            return "\n\n".join([x.strip() for x in execute_matches])

        return None
    

    
    async def _handle_code_and_execute(
        self,
        ctx: Context,
        current_llm_input: list[ChatMessage],
        tool_descriptions:str
        
    ):
        code_system_msgs = self.code_act_system_prompt.format_messages(
            tool_descriptions=tool_descriptions
        )
        current_llm_input = [*code_system_msgs, *current_llm_input]
    
        print(current_llm_input)
        response = await self.llm.astream_chat(current_llm_input)
        last_chat_response = ChatResponse(message=ChatMessage())
        streaming_token_len = 0
        
        full_response_text = ""

        async for last_chat_response in response:
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            list_of_content_blocks = last_chat_response.message.blocks
            text_block: TextBlock | None = next(
                filter(lambda x: isinstance(x, TextBlock), list_of_content_blocks), None
            )#type:ignore

            thinkink_blocks: list[ThinkingBlock] = list(
                filter(lambda x: isinstance(x, ThinkingBlock), list_of_content_blocks)
            ) #type:ignore
            if text_block:
                text = text_block.text  
                text_delta = text[streaming_token_len:]

                ctx.write_event_to_stream(
                    AgentStream(
                        delta=text_delta,
                        response=text, # full repsonse
                        tool_calls=[],
                        raw=raw,
                        current_agent_name=self.name,
                        thinking_delta=None
                    )
                )
                streaming_token_len = len(text)
                full_response_text = text
                continue
            if len(thinkink_blocks) > 0:
                thinking_delta = thinkink_blocks[-2] # [thinking delta, thinking full]
                ctx.write_event_to_stream(
                    AgentStream(
                        delta="", 
                        response="",
                        tool_calls=[],
                        raw=raw,
                        current_agent_name=self.name,
                        thinking_delta=thinking_delta.content
                    )
                )

        code = self._extract_code_from_response(full_response_text)
        tool_calls = []
        if code:
            tool_id = str(uuid.uuid4())
            tool_calls = [
                ToolSelection(
                    tool_id=tool_id,
                    tool_name=EXECUTE_TOOL_NAME,
                    tool_kwargs={"code": code},
                )
            ]
        
        if isinstance(self.llm, FunctionCallingLLM):
            extra_tool_calls = self.llm.get_tool_calls_from_response(
                last_chat_response, error_on_no_tool_call=False
            )
            tool_calls.extend(extra_tool_calls)
        
        raw = (
            last_chat_response.raw.model_dump()
            if isinstance(last_chat_response.raw, BaseModel)
            else last_chat_response.raw
        )
        message = ChatMessage(role="assistant", content=full_response_text)
        return message, tool_calls, raw

    async def take_step(
        self, 
        ctx: Context,
        llm_input: list[ChatMessage],
        tools:  Sequence[BaseTool],
        memory: BaseMemory,
    )->AgentOutput:
        
        if not self.code_execute_fn:
            raise ValueError("code_execute_fn must be provided for the code execution")
        exe_his = cast(
            list[ChatMessage],
            await ctx.store.get(self.execution_history_key, default=[])
        )
        
        for msg in llm_input:
            for block in msg.blocks:
                if isinstance(block, TextBlock):
                    try:
                        block.text = block.text.encode('latin1').decode('utf-8')
                    except Exception:
                        pass
                        
        current_llm_input = [*llm_input, *exe_his]

        tool_descriptions = self._get_tool_descriptions(tools)


        decision_prompt_text = MAKE_DECISION_PROMPT.format(
            tool_descriptions=tool_descriptions
        )
        decision_input = current_llm_input + [
            ChatMessage(role='user', content=decision_prompt_text)
        ]
        
        ctx.write_event_to_stream(
            AgentInput(
                input=decision_input, current_agent_name=self.name
            )
        )
        raw_response = None
        sllm = self.llm.as_structured_llm(ToolsOrCodeDecision)
        chat_response = await sllm.achat(decision_input)
        raw_response = ToolsOrCodeDecision.model_validate(chat_response.raw)
        
        ctx.write_event_to_stream(
            AgentDecision(
                name=self.name,
                decision=raw_response.decision,
                reason=raw_response.reason
            )
        )

        if raw_response.decision == 'code':
            message, tool_calls, raw = await self._handle_code_and_execute(
                ctx=ctx,
                current_llm_input=current_llm_input,
                tool_descriptions=tool_descriptions
            )
            
        elif raw_response.decision == 'tools':
            message, tool_calls, raw = await self._handle_tool_call(
                tools=tools,
                ctx=ctx,
                current_llm_input=current_llm_input
            )
        else:
            raise ValueError(f"Invalid decision: {raw_response.decision}. Must be 'codes' or 'tools' ")


        exe_his.append(message)
        async with ctx.store.edit_state() as state:
            state[self.execution_history_key] = exe_his
        
        return AgentOutput(
            response=message,
            tool_calls=tool_calls,
            raw=raw,
            current_agent_name=self.name
        )        
    

    async def handle_tool_call_results(
        self, ctx: Context, results: list[ToolCallResult], memory: BaseMemory
    ) -> None:
        
        history_context: list[ChatMessage] = await ctx.store.get(
            self.execution_history_key, default=[]
        )
        
        for tool_call_result in results:
            
            if tool_call_result.tool_name == EXECUTE_TOOL_NAME:
                """
                This is where the agent use the code agent mode
                """
                code_result = f"Result of executing the code given:\n\n{tool_call_result.tool_output.content}"
                history_context.append(
                    ChatMessage(
                        role="user",
                        content=code_result,
                    )
                )
                continue

            history_context.append(
                ChatMessage(
                    role="tool",
                    blocks=tool_call_result.tool_output.blocks,
                    additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                )
            )

            if tool_call_result.return_direct:
                history_context.append(
                    ChatMessage(
                        role="assistant",
                        content=str(tool_call_result.tool_output.content),
                        additional_kwargs={"tool_call_id": tool_call_result.tool_id},
                    )
                )
        async with ctx.store.edit_state() as state:
            state[self.execution_history_key] = history_context

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        history_context: list[ChatMessage] = await ctx.store.get(
            self.execution_history_key, default=[]
        )
        await memory.aput_messages(history_context)
        await ctx.store.set(self.execution_history_key, [])
        return output
