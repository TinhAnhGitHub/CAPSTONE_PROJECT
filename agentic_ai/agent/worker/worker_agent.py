from typing import Union, Callable, Awaitable, Sequence, cast
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
from agentic_ai.tools.type.factory import ToolOutputFormatter
from typing import Literal
from .prompt import MAKE_DECISION_PROMPT
class AgentDecision(Event):
    name: str
    decision:str
    reason:str


class AgentThinking(Event):
    name: str
    thinking_content: str


class ToolsOrCodeDecision(BaseModel):
    reason: str = Field(..., description="The reason why you obtain this decision")
    decision: Literal['tools', 'code']


EXECUTE_TOOL_NAME='execute'

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

    code_act_system_prompt: str = Field(
        ...,
        description="System prompt for code generation mode"
    )
    
    allow_parallel_tool_calls: bool = Field(
        default=True,
        description="If True, the agent will call multiple tools in parallel. If False, the agent will call tools sequentially.",
    )

    output_formatter: ToolOutputFormatter = Field(
        default_factory=ToolOutputFormatter,
        description="Formatter for tool outputs"
    )


    def __init__(
        self,
        execution_key:str,
        code_execute_fn: Union[Callable, Awaitable],
        code_act_system_prompt: str,
        name: str,
        description: str,
        system_prompt: str | None = None,
        tools: list[Union[BaseTool, Callable]] | None = None,
        tool_retriever: ObjectRetriever | None = None,
        llm: FunctionCallingLLM | None = None,
    ):
        """
        The initialization of the video agent
        - The code_act_system_prompt and the system prompt will be append together during the function calling
        """
    
        tools = tools or []
        tools.append(  
            FunctionTool.from_defaults(code_execute_fn, name='execute')  # type: ignore
        )
        self.execution_history_key = execution_key
        super().__init__(
                
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            tool_retriever=tool_retriever,
            can_handoff_to=None,
            llm=llm,
            code_act_system_prompt=code_act_system_prompt,
            code_execute_fn=code_execute_fn,
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

            tool_description += "\n  ...\n"
            tool_descriptions.append(tool_description)

        return "\n\n".join(tool_descriptions)
    
    
    
    async def _handle_tool_call(
        self,
        ctx: Context,
        current_llm_input: list[ChatMessage],
        tools: Sequence[BaseTool]
    ):
        response = await self.llm.astream_chat_with_tools(  # type: ignore
            tools=tools,
            chat_history=current_llm_input,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
        )

        last_chat_response = ChatResponse(message=ChatMessage())
        seen_thoughts = ""

        async for last_chat_response in response:
            thoughts = last_chat_response.additional_kwargs.get('thoughts','')
            if thoughts and thoughts != seen_thoughts:
                new_part = thoughts[len(seen_thoughts):]
                ctx.write_event_to_stream(
                    AgentThinking(
                        name=self.name,
                        thinking_content=new_part
                    )
                )

                seen_thoughts = thoughts
                continue

            tool_calls = self.llm.get_tool_calls_from_response(  # type: ignore
                last_chat_response, error_on_no_tool_call=False
            )
            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )
            ctx.write_event_to_stream(
                AgentStream(
                    delta=last_chat_response.delta or "",
                    response=last_chat_response.message.content or "",
                    tool_calls=tool_calls or [],
                    raw=raw,
                    current_agent_name=self.name,
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
        current_llm_input: list[ChatMessage]
    ):
        system_prompt_added = False
        for msg in current_llm_input:
            if msg.role.value == "system":
                system_prompt_added = True
                break
        
        if not system_prompt_added:
            code_system_msg = ChatMessage(
                role="system",
                content=self.code_act_system_prompt
            )
            current_llm_input = [code_system_msg] + current_llm_input
        

        response = await self.llm.astream_chat(current_llm_input)
        last_chat_response = ChatResponse(message=ChatMessage())
        
        full_response_text = ""
        seen_thoughts = ""

        async for last_chat_response in response:
            thoughts = last_chat_response.additional_kwargs.get('thoughts','')
            if thoughts and thoughts != seen_thoughts:
                new_part = thoughts[len(seen_thoughts):]
                ctx.write_event_to_stream(
                    AgentThinking(
                        name=self.name,
                        thinking_content=new_part
                    )
                )

                seen_thoughts = thoughts
                continue

            delta = last_chat_response.delta or ""
            full_response_text += delta

            raw = (
                last_chat_response.raw.model_dump()
                if isinstance(last_chat_response.raw, BaseModel)
                else last_chat_response.raw
            )

            ctx.write_event_to_stream(
                AgentStream(
                    delta=delta,
                    response=full_response_text,
                    tool_calls=[],
                    raw=raw,
                    current_agent_name=self.name,
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
                input=current_llm_input, current_agent_name=self.name
            )
        )

        sllm = self.llm.as_structured_llm(ToolsOrCodeDecision)
        chat_response = await sllm.achat(decision_input)
        raw_response = cast(ToolsOrCodeDecision, chat_response.raw)
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
                current_llm_input=current_llm_input
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
                
            await ctx.store.set(self.execution_history_key, history_context)
    

    async def finalize(
        self, ctx: Context, output: AgentOutput, memory: BaseMemory
    ) -> AgentOutput:
        history_context: list[ChatMessage] = await ctx.store.get(
            self.execution_history_key, default=[]
        )
        await memory.aput_messages(history_context)
        await ctx.store.set(self.execution_history_key, [])
        return output