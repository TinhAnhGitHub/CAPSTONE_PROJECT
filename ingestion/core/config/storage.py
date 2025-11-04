"""
Contain the configuration related to the persistent Storage
1. Postgre
2. Minio client
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field,  computed_field

class MinioSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='MINIO_',
        case_sensitive=False,
        extra='ignore'
    )
    port: str
    host: str
    user:str
    password: str
    access_key: str = Field(default="minioadmin", description="MinIO access key")
    secret_key: str = Field(default="minioadmin", description="MinIO secret key")
    secure: bool = Field(default=False, description="Whether to use HTTPS when contacting MinIO")

    @computed_field
    @property
    def endpoint(self) -> str:
        scheme = "https" if self.secure else "http"
        return f"{scheme}://{self.host}:{self.port}"

class PostgreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='POSTGRE_',
        case_sensitive=False,
        extra='ignore'
    )
    database_url: str

class MilvusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_prefix='MILVUS_',
        case_sensitive=False,
        extra='ignore'
    )

    host: str = Field(default="standalone", description="Milvus host")
    port: int = Field(default=19530, description="Milvus gRPC port")
    user: str | None = Field(default=None, description="Milvus username (optional)")
    password: str | None = Field(default=None, description="Milvus password (optional)")
    db_name: str = Field(default="default", description="Milvus database name")
    time_out: float = 30.0
    
    @computed_field
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"



minio_settings = MinioSettings() #type:ignore
postgre_settings = PostgreSettings() #type:ignore
milvus_settings = MilvusSettings()  # type: ignore



