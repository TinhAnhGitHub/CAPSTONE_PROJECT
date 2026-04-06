
from google.genai import types
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from llama_index.core.base.llms.types import ChatMessage
from pprint import pprint
load_dotenv()


llm = GoogleGenAI(
    model='gemini-2.5-flash',
)


class Hello(BaseModel):
    hello_response: str

input_msg = ChatMessage(role="user", content="Hello, give me a long poem on dinosaur")
sllm = llm.as_structured_llm(Hello)
stream_output = sllm.stream_chat([input_msg])

for partial_output in stream_output:
    pprint(partial_output.raw.dict())
    restaurant_obj = partial_output.raw