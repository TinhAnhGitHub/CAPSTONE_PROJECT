from beanie import PydanticObjectId
from fastapi import APIRouter, UploadFile, File
from app.core.config import settings

from app.model.video import Video
from .socket import sio
router = APIRouter(prefix="/api/ingestion/service", tags=["ingestion"])

@router.post("/status/{video_id}")
async def ingestion_status(video_id: str, data: dict):
    status = data.get("status") #
    # find video and update status
    video = await Video.get(PydanticObjectId(video_id))
    video.status = status
    await video.save()
    
    # gửi socket frontend
    await sio.emit("ingestion_status", {"video_id": video_id, "status": status})
    return {"msg": "Status updated"}