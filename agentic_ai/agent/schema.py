
from typing import Any, Optional, List, Tuple
from pydantic import Field, BaseModel

class WorkerBluePrint(BaseModel):
   name: str = Field(..., description="The specific name of the agents")
   description: str = Field(..., description="The description detail of the agent. This description will refect its jobs, role, and how it should reasoning, perspective, to get the answer, based on the provided tools")
   task: str = Field(..., description="The specific task that it must complete")
   tools: list[str] = Field(..., description="The available tools that the system offer")
   max_iterations: int = Field(3, description="How many times that the agent have to try, before it give up :(  ")

class Img(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the image to be processed.")
    image_base64: Optional[str] = Field(None, description="Base64 encoded image data.")
    image_type: str

class Filter(BaseModel):
    video_url: Optional[List[str]] = []
    timestamp: Optional[Tuple]


class Request(BaseModel):
    user_id: str
    session_id: str
    query: str
    image: Optional[Img] = None
    filter: Optional[Filter] = None


class WorkersPlan(BaseModel):
    plan: list[WorkerBluePrint] = Field(default_factory=list,description="The plan for these agents. Should be around 1-3 agents only. 2 is a sweet spot.")
