from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionAgent
from typing import Annotated, Any
from pydantic import BaseModel, Field
import json

from dotenv import load_dotenv
from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.google_genai import GoogleGenAI
load_dotenv(dotenv_path='/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/videodeepsearch/test/.env')

from event import EventHandler

generation_config = types.GenerateContentConfig(
    temperature=0.1,
    top_k=1,
    top_p=1,
)
LLM = GoogleGenAI(
    model='gemini-2.5-flash-lite',
    generation_config=generation_config
)



class LocalContext(BaseModel):
    task: str = Field(..., description="The task of the subagents")
    history: list[str] = Field(default_factory=list, description="The work history")


class SharedContext(BaseModel):
    shared_history: dict[str, list[str]] = Field(default_factory=dict, description="The history of all agents")


# method to modify LocalContext -> belong to worker agents
async def worker_add_work_task(
    external_store: Context,
    agent_name: str,
    result: str
) -> str:
    """
    Use this tool to persist the results.
    """

    try:
        async with external_store.store.edit_state() as ctx_state:
            local_context = LocalContext.model_validate(
                ctx_state[agent_name]
            )
            # add shit
            local_context.history.append(result)
            ctx_state[agent_name] = local_context.model_dump()
        return "Result add successfully"
    except Exception as e:
        return f"Error has been found: {str(e)}. Ignore the tool"

async def worker_persist_work_shared_context(
    external_store: Context,
    agent_name: str,
    session_id: str
) -> str:
    """
    Use this tool when you finish the task, and want to persist the result to the shared context
    """
    try:
        async with external_store.store.edit_state() as ctx_state:
            shared_context = SharedContext.model_validate(
                ctx_state[session_id]
            )
            local_context = LocalContext.model_validate(
                ctx_state[agent_name]
            )

            shared_context.shared_history[agent_name] = local_context.history
            ctx_state[session_id] = shared_context.model_dump()
        return "Your history has been shared to the shared context"
    except Exception as e:
        return f"Error has been found {str(e)}. Ignore the tool"


# method to view the history of the worker agent
async def orc_view_results(ctx: Context, session_id: str):
    shared_context_raw = await ctx.store.get(session_id)
    shared_context = SharedContext.model_validate(shared_context_raw)
    return json.dumps(
        shared_context.shared_history,
        indent=2,
        ensure_ascii=False
    )


async def create_worker_agent(
    ctx: Context,
    session_id: str,
    agent_task: Annotated[str, "Detail task for the agent"],
    agent_description: Annotated[str, "agent Description"],
    agent_name: Annotated[str, "The agent name"]
) -> str: 
    """
    use this tool to spawn worker agents
    """
    partial_params = {
        'agent_name': agent_name,
    }
    
    persist_params = {
        'agent_name': agent_name, 
        'session_id': session_id,
    }
    worker_tools = [
        FunctionTool.from_defaults(async_fn=worker_add_work_task, partial_params=partial_params),
        FunctionTool.from_defaults(async_fn=worker_persist_work_shared_context, partial_params=persist_params)
    ]

    agent_context = LocalContext(task=agent_task)
    async with ctx.store.edit_state() as ctx_state:
        ctx_state[agent_name] = agent_context.model_dump(mode='json')    

    agent =  FunctionAgent(
        name=agent_name,
        description=agent_description,
        system_prompt=(
        "You are a worker agent with a specific task to complete. Follow these steps EXACTLY:\n"
        "1. Analyze the task you've been given.\n"
        "2. Perform the research or analysis required.\n"
        "3. Call `worker_add_work_task` with your detailed findings.\n"
        "4. IMPORTANT: After adding your work, you MUST call `worker_persist_work_shared_context` to share your results.\n"
        "5. Once you've called both tools, your job is complete.\n\n"
        "DO NOT skip step 4. Always persist your results to the shared context."
    ),
        tools= worker_tools,
        llm=LLM,
        
    )


    worker_ctx = Context(agent)
    worker_ctx._state_store = ctx.store
    event_handler = EventHandler()
    handler = agent.run(user_msg=f"Here is the task: {agent_task}", ctx=ctx)
    async for ev in handler.stream_events():
        event_handler.handle_event(ev)
    result = await handler


    return result.response.content


def get_agent_as_tools(session_id: str) -> list[FunctionTool]:
    args = {
        'session_id': session_id
    }
    return [
        FunctionTool.from_defaults(async_fn=create_worker_agent, partial_params=args),
        FunctionTool.from_defaults(async_fn=orc_view_results, partial_params=args)
    ]


