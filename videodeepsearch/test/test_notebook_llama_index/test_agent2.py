from llama_index.core.agent import FunctionAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Event, StartEvent , StopEvent
from llama_index.core.workflow import Workflow
from llama_index.core.agent.workflow import AgentOutput
from llama_index.core.workflow.decorators import step
from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Any

load_dotenv(dotenv_path='/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/videodeepsearch/test/.env')

generation_config = types.GenerateContentConfig(
    temperature=0.1,
    top_k=1,
    top_p=1,
)


llm = GoogleGenAI(
    model='gemini-2.5-flash',
    generation_config=generation_config
)
session_id = "123"



CoreAgent = FunctionAgent(

)

class LocalContext(BaseModel):
    previous_answers: list[str]


class SharedContext(BaseModel):
    shared_context: dict[str, list]


async def add_poem_to_local_context(ctx: Context, answer: str, agent_name: str) -> bool:
    """
    Use this tool to persist into local context
    Must use after each version done 
    """
    try:
        async with ctx.store.edit_state() as ctx_state:
            state = LocalContext.model_validate(ctx_state[agent_name])
            state.previous_answers.append(answer)
        
        return True
    except Exception as e:
        raise e
    


async def persist_shared_context(ctx: Context, slicing: list[int], agent_name: str) -> bool:
    """
    Transfer the answer from local to shared context
    slicing: [0,1]  in python
            [0,3]
    """

    try:
        async with ctx.store.edit_state() as ctx_state:
            global_state = SharedContext.model_validate(ctx_state[session_id])
            local_context = LocalContext.model_validate(ctx_state[agent_name])

            global_state.shared_context[agent_name] = local_context.previous_answers[slicing]

        return True
    except Exception as e:
        raise e







async def agent1(ctx: Context, agent_name: str, poem_topic: str):
    prompt = f"Here is a poem topic: {poem_topic}, please generate them. You can generate any version you like (10 of them). and choose 3 of them to persist into shared context"

    agent = FunctionAgent(
        agent_name=agent_name,
        description="You are an agent that finish the task",
    tools=[
        FunctionTool(fn=add_poem_to_local_context, partial_params={'agent_name': agent_name}),
        FunctionTool
    
    ]
    )
    handler = agent.run(user_msg=prompt)
    async for event in handler.stream_events():
        print(event)
    
    answer = await handler
    answer_str = answer.response.content

    return answer_str





