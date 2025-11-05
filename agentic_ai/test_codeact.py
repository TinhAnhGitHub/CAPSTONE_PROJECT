from llama_index.core.agent.workflow import CodeActAgent
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool, ToolMetadata, BaseTool
from typing import Any
from tools import single_tools, random_tool



# ---- STEP 3: Create code executor ----

def create_code_executor_for_worker(worker_tools: list[FunctionTool]):
    async def executor(code: str) -> Any:
        allowed_tools = {tool.metadata.name: tool.fn for tool in worker_tools}
        safe_globals = {"__builtins__": {"print": print, "range": range, "len": len}}
        safe_globals.update({tool.metadata.name: tool.fn for tool in worker_tools})
        safe_locals = allowed_tools.copy()
        print("\n--- Result ---")
        print(code)
        print("-----------------\n")
        
        async def _run_async_code():
            try:
                exec(
                    f"async def __worker_fn__():\n"
                    + "\n".join(f"    {line}" for line in code.splitlines()),
                    safe_globals,
                    safe_locals,
                )
                res = await safe_locals["__worker_fn__"]()
                
                return res
            except Exception as e:
                print(f"Execution error: {type(e).__name__}: {e}")
                return f"Execution error: {type(e).__name__}: {e}"

        try:           
            return await _run_async_code()
        except Exception as e:
            print(f"Worker runtime error: {type(e).__name__}: {e}")
            return f"Worker runtime error: {type(e).__name__}: {e}"

    return executor



import google.generativeai as genai
from dotenv import load_dotenv
import os
class Coder(CodeActAgent):
    def __init__(self, code_execute_fn, llm : Gemini,name = "code_act_agent", description = "A workflow agent that can execute code.", system_prompt = "", tools = []):
        super().__init__(code_execute_fn =code_execute_fn,
                         llm = llm, 
                         name = name, 
                         description= description, 
                         system_prompt=system_prompt, 
                         tools = tools)
    def __hash__(self):
        return hash(self.name)

BASE_FOLDER = "keyframes/"
Sys = f"""
You are a VideoQA Assistant that has access to tools that access to a series of n={len(os.listdir(BASE_FOLDER))} frames in a folder. Using the tools, answer the questions.
Normal pattern: 
- Pick a random number between 0 and n-1 using random_tool
- Get the caption then compare to query
- If this is not the frame you looking for, based on the previous retrieve frame to guess the possible frame
- Once targetted frame found, answer question all call appropriate tool to answer question

Rules:
1. You CANNOT write or execute raw Python code.
2. You MUST use the provided tools to perform any operation.
3. You CANNOT use 'import', 'open', 'int()', 'float()', or any other Python syntax.
4. You MUST NOT define functions or variables outside of tool calls.
5. All computation and reasoning happens by calling tools.
"""

async def main():

    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    GEMINI_MODELS = (
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-thinking",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash",
    )

    llm = Gemini(model=GEMINI_MODELS[0])

    
    code_execute_fn = create_code_executor_for_worker(single_tools+ random_tool)

    worker = Coder(
        llm=llm,
        name="math_worker",
        description="You are a VideoQA Assistant",
        system_prompt=Sys,
        tools=single_tools+ random_tool,
        code_execute_fn=code_execute_fn,
    )
    q = """
        There is a frame with a big cat and no human. What is the colour of the cat ?
    """
    handler = await worker.run(user_msg=q,chat_history=[])

    print("=== HANDLER TYPE ===\n", type(handler))
    print("\n=== HANDLER DIR ===\n", dir(handler))

    print("Agent Finish running")
    async for ev in handler.stream_events():
        print(">>> EVENT:", ev)
          
    print("Agent Finish Streaming")

    return handler.result()
    


import asyncio
if __name__ == "__main__":
    asyncio.run(main())


