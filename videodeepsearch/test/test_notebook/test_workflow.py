import asyncio
from llama_index.core.workflow import Context
from llama_index.core.workflow import Event, StartEvent , StopEvent
from llama_index.core.workflow import Workflow
from llama_index.core.workflow.decorators import step



class FinalResponseEvent(StartEvent):
    def __init__(self, response: str, agent_name: str):
        super().__init__()
        self.response = response
        self.agent_name = agent_name


class AgentProgressEvent(Event):
    def __init__(self, agent_name: str, answer: str):
        super().__init__()
        self.agent_name = agent_name
        self.answer = answer


class MyWorkflow(Workflow):

    @step
    async def final_response(self, ev: FinalResponseEvent, ctx: Context)->StopEvent:
        final_resonse = ev.response
        agent_name = ev.agent_name

        async def _yield_response_streaming(final_resonse: str):
            for character in final_resonse:
                yield character

        res = ''
        async for character in _yield_response_streaming(final_resonse):
            res += character
            ctx.write_event_to_stream(
                AgentProgressEvent(agent_name=agent_name, answer=res)
            )

        return StopEvent(result=res)


# ---- Test Runner ----
async def test_final_response():
    wf = MyWorkflow()

    collected_progress = []


    ctx = Context(workflow=wf)
    text = (
        "This is a simulated large language model response. "
        "It keeps generating text continuously to mimic streaming output. "
        "Here's some more text for testing purposes, including punctuation, numbers (12345), "
        "and even emojis 😊 to ensure UTF-8 handling works correctly. "
    ) * 5 

    ev = FinalResponseEvent(response=text, agent_name="TestAgent")
    handler = wf.run(start_event=ev)
    collected_progress = []
    
    async for stream_event in handler.stream_events():
        if isinstance(stream_event, AgentProgressEvent):
            collected_progress.append((stream_event.agent_name, stream_event.answer))
            print(f"{stream_event.answer}", end='', flush=True)
    
    print()
    result = await handler
    print("Final StopEvent result:", result)
    print("Streamed Progress Updates:")
    for name, answer in collected_progress:
        print(f"{name}: {answer}")

# Run it
asyncio.run(test_final_response())
