from typing import Union, Callable, Awaitable, Sequence, cast, Annotated
import inspect
from llama_index.core.memory import BaseMemory
from llama_index.core.agent.workflow.base_agent import BaseWorkflowAgent
from llama_index.core.agent.workflow.workflow_events import (
    AgentInput,
    AgentOutput,
    AgentStream,
    ToolCallResult,
)
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection, LLM
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.workflow import Context
from llama_index.core.objects import ObjectRetriever
from llama_index.core.prompts import PromptTemplate

from .schema import ToolsOrCodeDecision


EXECUTE_TOOL_NAME='execute'


MAKE_DECISION_PROMPT: Annotated[
    PromptTemplate,
    "This prompt template will help the agent choosing to use the function directly, or to spawn a code which can execute more complex task"
] = PromptTemplate(
    """
    ...
    """
)
class WorkerCodeVideoContext(BaseModel):
    """
    A small context customize for the agent for interacting the environment
    """

    execution_history: list[ChatMessage] = Field(default_factory=list, description="The context history. The worker agent will iterate the result again and again, so this serive as a workspace")


class WorkerCodeVideoAgent(BaseWorkflowAgent):
    """
    A workflow agent that is specialized in generating code and execute a bunch of tools
    """

    code_execute_fn: Union[Callable, Awaitable] = Field(
        ...,
        description="The function to execute the code"
    )

    code_act_system_prompt: str  = Field(...)

    def __init__(
        self,
        code_execute_fn: Union[Callable, Awaitable],
        code_act_system_prompt: str,
        name: str,
        description: str,
        system_prompt: str | None = None,
        tools: list[Union[BaseTool, Callable]] | None = None,
        tool_retriever: ObjectRetriever | None = None,
        can_handoff_to: list[str] | None = None,
        llm: LLM | None = None,
    ):
        """
        The initialization of the video agent
        - The code_act_system_prompt and the system prompt will be append together during the function calling
        """
    
        tools = tools or []
        tools.append(  
            FunctionTool.from_defaults(code_execute_fn, name='execute')  # type: ignore
        )

        super().__init__(    
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            tool_retriever=tool_retriever,
            can_handoff_to=can_handoff_to,
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
    

    async def take_step(
        self, 
        ctx: Context,
        llm_input: list[ChatMessage],
        tools:  Sequence[BaseTool],
        memory: BaseMemory,
    )->AgentOutput:
        
        specific_context = cast(
            Context[WorkerCodeVideoContext], ctx
        )
        if not self.code_execute_fn:
            raise ValueError("code_execute_fn must be provided for the code execution")
        
        exe_his = await specific_context.store.get('execution_history', default=[])
        current_llm_input = [*llm_input, *exe_his]

        tool_descriptions = self._get_tool_descriptions(tools)


        make_decision_prompt = MAKE_DECISION_PROMPT.format(
            tool_descriptions=tool_descriptions
        )

        sllm = self.llm.as_structured_llm(ToolsOrCodeDecision)
        stream_output = sllm.stream_chat([current_llm_input])
        
        



        
