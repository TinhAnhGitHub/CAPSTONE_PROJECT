import asyncio
import json
from llama_index.core.agent import FunctionAgent
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.workflow import Context, JsonSerializer
from dotenv import load_dotenv
from google.genai import types
from  pprint import pprint
from tools import get_agent_as_tools, SharedContext
from event import EventHandler



load_dotenv(dotenv_path='/home/tinhanhnguyen/Desktop/HK7/Capstone/CAPSTONE_PROJECT/videodeepsearch/test/.env')


generation_config = types.GenerateContentConfig(
    temperature=0.1,
    top_k=1,
    top_p=1,
)
LLM = GoogleGenAI(
    model='gemini-2.5-flash-lite',
    generation_config=generation_config
)




async def main():
    session_id = '123'
    
    orchestrator_agent = FunctionAgent(
        name='orchestration agent',
        system_prompt="From the task given, divide the task into smaller task, assign task to the worker agents. When the agents have done their tasks, please view the result, and return the final answer.",
        tools=get_agent_as_tools(session_id),
        llm=LLM
    )

    ctx = Context(orchestrator_agent)

    async with ctx.store.edit_state() as ctx_state:
        ctx_state[session_id] = SharedContext().model_dump()

    user_demand = "I want you to : Analyze the topic 'Reinforcement Learning'. Divide it into 3 subtasks, assign each to a worker, persist results, and combine them"

    handler = orchestrator_agent.run(
        user_msg=user_demand,
        ctx=ctx
    )
    event_handler = EventHandler()
    async for ev in handler.stream_events():
        event_handler.handle_event(ev)
    result = await handler

    # inspect context

    context_dict = ctx.to_dict(JsonSerializer())

    with open('./context.json', 'w') as f:
        json.dump(context_dict, f, indent=2, ensure_ascii=False)
    pprint(context_dict)

    return result.response.content


if __name__ == '__main__':
    asyncio.run(
        main=main()
    )