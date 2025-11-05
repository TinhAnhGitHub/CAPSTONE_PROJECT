from agent.log import setup_logger, get_logger


# service
import asyncio
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

# llama
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool, ToolMetadata
import google.generativeai as genai


# workflow
from agent.workflow import VideoAgentWorkFlow, Context
from agent.events import UserInputEvent, FinalResponseEvent
from agent.state import AgentState
from agent.schema import Request
from core.app_state import Appstate

from tools.tools import single_tools

from chat_data import get_chat_history, save_chat_history


from dotenv import load_dotenv
from os import getenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    setup_logger()
    logger = get_logger("Service")
    genai.configure(api_key=getenv("GOOGLE_API_KEY")) # Do i Need this line ???
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
    logger.info("FastAPI service starting up...")
    llm = Gemini(model=GEMINI_MODELS[0])
    app.llm =llm
    app.app_state = Appstate()
    app.app_state.workflow = VideoAgentWorkFlow(
        llm=llm,
        context_tools=[],
        all_tools=single_tools,
        logger=get_logger("Workflow")
    )
    yield
    app.app_state.workflow = None


app = FastAPI(lifespan=lifespan)


async def run_query(user_id: str, session_id: str, query: str) -> str:
    h = get_chat_history(user_id, session_id)
  
    e = UserInputEvent(input=query, chat_history=h)
    """
    app.app_state.workflow = VideoAgentWorkFlow(
        llm=app.llm,
        context_tools=[],
        all_tools=single_tools,
    )
    """
    
    wf = app.app_state.workflow
    r = None
    handler = wf.run(start_event= e)
    async for ev in handler.stream_events():
        if isinstance(ev, FinalResponseEvent):
            r = ev.response
            break
    return r or ""

import traceback
@app.post("/query", response_model=FinalResponseEvent)
async def query(req: Request):
    try:
        res = await run_query(req.user_id, req.session_id, req.query)
        with open("response/print.log", "w") as f:
            traceback.print_exc(file=f)
        with open("response/ctx.txt", "w") as f:
            ctx = app.app_state.workflow.ctx 
            f.write(str(ctx.__dict__).replace("\\n", "\n"))
        return FinalResponseEvent(response=res)
    except Exception as e:
        with open("response/error.log", "w") as f:
            traceback.print_exc(file=f)
        with open("response/ctx.txt", "w") as f:
            ctx = app.app_state.workflow.ctx 
            f.write(str(ctx.__dict__).replace("\\n", "\n"))
        raise HTTPException(status_code=500, detail=str(e))
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)


