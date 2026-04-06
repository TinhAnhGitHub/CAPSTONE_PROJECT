from beanie import Document
from app.core.config import settings
from beanie import PydanticObjectId


class SessionVideo(Document):
    session_id: PydanticObjectId
    video_id: PydanticObjectId
    selected: bool = False

    class Settings:
        name = settings.SESSION_VIDEO_COLLECTION_NAME
        indexes = ["session_id", "video_id", [("last_updated", -1)]]
