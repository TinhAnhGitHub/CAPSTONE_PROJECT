from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    """
    Core config of the whole application
    """

    UPLOAD_DIR: str = "uploads"

    # Mongo Settings:
    MONGO_URI: str = Field("mongodb://100.120.22.90:27017", description="MongoDB connection string")
    MONGO_DB: str = Field("mydatabase", description="Default database name")

    # chat settings
    CHAT_COLLECTION_NAME: str = Field("chat_history", description="default chat collection names")
    # user settings
    USER_COLLECTION_NAME: str = Field("users", description="default user collection name")
    # group settings
    GROUP_COLLECTION_NAME: str = Field("groups", description="default group collection name")
    # video settings
    VIDEO_COLLECTION_NAME: str = Field("videos", description="default video collection name")
    SESSION_VIDEO_COLLECTION_NAME: str = Field(
        "session_videos", description="default session video collection name"
    )
    CHAT_MESSAGE_COLLECTION_NAME: str = Field(
        "chat_messages", description="default chat message collection name"
    )

    # GOOGLE AUTH CLIENT settings
    GOOGLE_OAUTH_CLIENT_ID: str = Field(..., description="Google OAuth Client ID")
    GOOGLE_OAUTH_CLIENT_SECRET: str = Field(..., description="Google OAuth Client Secret")

    # MINIO settings for file storage
    MINIO_PUBLIC_ENDPOINT: str = Field("100.113.186.28:9000", description="MinIO server endpoint")

    MINIO_ACCESS_KEY: str = Field("minioadmin", description="MinIO access key")
    MINIO_SECRET_KEY: str = Field("minioadmin", description="MinIO secret key")
    # MINIO_SECURE: bool = Field(False, description="Use secure connection (HTTPS) for MinIO")
    # media url 
    MEDIA_URL_BASE: str = Field("http://localhost:8011/media/", description="Media URL base")

    # Redis / Celery settings
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection URL (broker + result backend)")

    # Ingestion pipeline service URLs
    INGESTION_SERVICE_URL: str = Field("http://100.113.186.28:8050", description="Video pipeline ingestion endpoint")
    INGESTION_CANCEL_URL: str = Field("http://100.113.186.28:8000", description="Video pipeline management/cancel endpoint")

    # Server settings
    HOST: str = Field("0.0.0.0", description="Server host")
    PORT: int = Field(8011, description="Server port")

    class Config:
        env_file = ".env"   
        env_file_encoding = "utf-8"


settings = AppSettings() #type: ignore
