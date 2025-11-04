
import asyncio
from agent.workflow import VideoAgentWorkFlow
from dotenv import load_dotenv
from os import getenv

from tools.tools import single_tools
from agent.log import setup_logger, get_logger

import google.generativeai as genai
from llama_index.llms.gemini import Gemini

from agent.events import UserInputEvent, FinalResponseEvent, StopEvent


async def main(x =1):
    load_dotenv()
    setup_logger()
    logger = get_logger("workflow")
    genai.configure(api_key=getenv("GOOGLE_API_KEY")) 
    GEMINI_MODELS = (
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-thinking",
        "models/gemini-2.0-flash-thinking-exp-01-21",
        "models/gemini-2.0-flash-lite",
        "models/gemini-2.0-flash-lite-preview-02-05",
        "models/gemini-2.0-pro-exp-02-05",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.0-pro",
    )
    llm = Gemini(model=GEMINI_MODELS[0])

    wf= VideoAgentWorkFlow(
        llm=llm,
        context_tools=[],
        all_tools=single_tools,
        logger=get_logger("Workflow")
    )

    
    if x ==1:
        #e = UserInputEvent(input=input("Type the querry: "), chat_history=[])
        e = UserInputEvent(input="Find me the frame with a big white cat", chat_history=[])
        r = None
        handler = wf.run(start_event= e)
        count = 0

        async for ev in handler.stream_events(True):
            if ev: 
                print(f"Event {count}:\n {ev}")
            count+=1
            if isinstance(ev, FinalResponseEvent):
                r = ev
                break
    elif x ==2:
        #handler = wf.greeting_agent.run(user_msg=input("Type the querry: "), chat_history=[])
        handler = await wf.greeting_agent.run(user_msg="Find me the frame with a big white cat", chat_history=[])
        print("=== HANDLER TYPE ===\n", type(handler))
        print("\n=== HANDLER ===")
        count = 0
        '''
        async for ev in handler.stream_events(True):
            if ev: 
                print(f"Event {count}:\n {ev}")
            count+=1
            if isinstance(ev, StopEvent):
                r = ev
                break
        '''
        r = handler.raw
        print("Agent return")
    return r or ""


if __name__ == "__main__":
        x=int(input("Select test to run (1- workflow, 2- greeting agent): "))
        r =  asyncio.run(main(x))
        print("================== Final result ==================\n",r)   
        print(r)     



            
