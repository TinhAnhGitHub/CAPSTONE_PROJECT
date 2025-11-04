from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


# Define simple calculator tools
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


# Create function tools
add_tool = FunctionTool.from_defaults(fn=add_numbers)
multiply_tool = FunctionTool.from_defaults(fn=multiply_numbers)
divide_tool = FunctionTool.from_defaults(fn=divide_numbers)

# Initialize LLM

class test(FunctionAgent):
   def __hash__(self):
        return hash(self.name)
# Create FunctionAgent


# Test the agent with a complex calculation
import asyncio
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
from dotenv import load_dotenv
import os
async def test_function_agent():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    GEMINI_MODELS = (
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-thinking",
        "models/gemini-2.0-flash-lite",
        "models/gemini-1.5-flash",
    )

    llm = Gemini(model=GEMINI_MODELS[0])
    print("🔍 Testing FunctionAgent with Maxim observability...")

    query = "What is (15 + 25) multiplied by 2, then divided by 8?"

    print(f"\n📝 Query: {query}")

    agent = test(
        tools=[add_tool, multiply_tool, divide_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a helpful calculator assistant.
        Use the provided tools to perform mathematical calculations.
        Always explain your reasoning step by step.""",
    )

    # This will be automatically logged by Maxim instrumentation
    # FunctionAgent.run() is async, so we need to await it
    response = await agent.run(query)

    print(f"\n🤖 Response: {response}")
    print("\n✅ Check your Maxim dashboard for detailed trace information!")


# Run the async function
asyncio.run(test_function_agent())