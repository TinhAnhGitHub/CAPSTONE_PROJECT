
from pydantic import BaseModel


class Group(BaseModel):
    group_id: str

class Video(BaseModel):
    video_id: str

    
