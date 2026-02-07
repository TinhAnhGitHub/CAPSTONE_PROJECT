from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field


class MinioConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="MINIO_", case_sensitive=False, extra="ignore"
    )
    port: str
    host: str
    user: str
    password: str
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    secure: bool = Field(
        default=False, description="Whether to use HTTPS when contacting MinIO"
    )

    @computed_field
    @property
    def endpoint(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.host}:{self.port}"
