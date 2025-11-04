import argparse
from typing import Any    
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()



global_events = []

@app.post('/api/ingestion/service/status/{video_id}')
async def receive_status_video(video_id: str, request: Request) :
    payload = await request.json()
    global_events.append({"video_id": video_id, "payload": payload})
    print(f"[MockBackEndTracker] video_id={video_id} | json_payload: {payload}")
    return {'ok': True}


@app.get('/events')
async def list_events():
   return global_events





    