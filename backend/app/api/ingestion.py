from beanie import PydanticObjectId
from fastapi import APIRouter, UploadFile, File

from app.model.video import Video
from .socket import sio
router = APIRouter(prefix="/api/ingestion/service", tags=["ingestion"])

@router.post("/status/{video_id}")
async def ingestion_status(video_id: str, data: dict):
    ingested_status = data.get("overall_percentage", 0)
    run_id = data.get("run_id", None)
    # find video and update status
    video = await Video.get(PydanticObjectId(video_id))
    video.ingested_status = ingested_status
    if video.run_id is None:
        video.run_id = PydanticObjectId(run_id)
        
    await video.save()

    # gửi socket frontend
    await sio.emit("ingestion_status", {"video_id": video_id, "ingested_status": ingested_status, "run_id": run_id})

    return {"msg": "Ingestion status updated"}
