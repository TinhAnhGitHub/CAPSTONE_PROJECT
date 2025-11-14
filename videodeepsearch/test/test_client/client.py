import json
import websockets
from typing import Optional

from llama_index.core.llms import ChatMessage
from event_handler import EventHandler
from videodeepsearch.agent.orc_events import (
    FinalEvent,
)
class WorkflowClient:
    def __init__(self, websocket_url:str, user_id: str, event_handler: EventHandler):
        self.websocket_url = websocket_url
        self.user_id = user_id
        self.event_handler = event_handler

    async def execute_workflow(
        self,
        user_demand: str,
        video_ids: list[str],
        chat_history: list[ChatMessage],
        session_id: str
    )-> list[ChatMessage]:
        final_response = None
        
        new_chat_history = None
        try:
            async with websockets.connect(
                self.websocket_url
            ) as websocket: 
                request_payload = {
                    "user_id": self.user_id,
                    "video_ids": video_ids,
                    "user_demand": user_demand,
                    "chat_history": [message.model_dump(mode='json') for message in chat_history],
                    # "session_id": session_id,
                }

                await websocket.send(json.dumps(request_payload))

                while True:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        if data.get("type") == "error":
                            err = data.get("error")
                            tb = data.get("traceback")
                            from rich.console import Console
                            Console().print("\n[bold red]✗ Server Error:[/bold red] " + str(err))
                            if tb:
                                Console().print(f"[dim]{tb}[/dim]")
                            break

                        if data.get("type") == "complete":
                            break

                        if data.get("type") == "workflow_event":
                            event_data = data.get("data", {})
                            response = self.event_handler.handle_event(event_data)
                            if isinstance(response, FinalEvent):
                                new_chat_history = response.chat_history
                      
                    
                    except websockets.exceptions.ConnectionClosed:
                        from rich.console import Console
                        Console().print("\n[yellow]⚠ WebSocket connection closed[/yellow]")
                        break
        except Exception as e:
            from rich.console import Console
            Console().print(f"\n[bold red]✗ WebSocket error:[/bold red] {e}")
            raise
        
        return new_chat_history #type:ignore
