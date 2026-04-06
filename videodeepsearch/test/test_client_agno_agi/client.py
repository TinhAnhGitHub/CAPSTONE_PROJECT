import json
import websockets

from agno.agent import Message
from event_handler import EventHandler

class WorkflowClient:
    def __init__(self, websocket_url:str, user_id: str, event_handler: EventHandler):
        self.websocket_url = websocket_url
        self.user_id = user_id
        self.event_handler = event_handler

    async def execute_workflow(
        self,
        user_demand: str,
        video_ids: list[str],
        chat_history: list[Message],
        session_id: str
    )-> list[Message]:
        

        
    
        async with websockets.connect(
            self.websocket_url
        ) as websocket: 
            request_payload = {
                "user_id": self.user_id,
                "video_ids": video_ids,
                "user_demand": user_demand,
                "chat_history": [message.model_dump(mode='json') for message in chat_history],
                "session_id": session_id,
            }

            await websocket.send(json.dumps(request_payload))

            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    self.event_handler.handle_event(data)
                    
                
                except websockets.exceptions.ConnectionClosed:
                    from rich.console import Console
                    Console().print("\n[yellow]⚠ WebSocket connection closed[/yellow]")
                    break
    
        
        return chat_history 
