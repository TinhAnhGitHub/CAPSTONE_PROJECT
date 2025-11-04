from pydantic import Field

from shared.config import ServiceConfig, LogLevel


class ASRServiceConfig(ServiceConfig):
    """Runtime configuration for the ASR service."""

    model_name: str = Field(..., description="Path to the Chunkformer model checkpoint directory")
    temp_dir: str = Field(default="./tmp/asr", description="Directory for intermediate audio artifacts")

    default_chunk_size: int = Field(default=64, ge=1)
    default_left_context: int = Field(default=128, ge=0)
    default_right_context: int = Field(default=128, ge=0)
    default_total_batch_duration: int = Field(default=900, ge=1)
    default_sample_rate: int = Field(default=16000, ge=8000)
    default_num_extraction_workers: int = Field(default=1, ge=1, le=4)
    default_num_asr_workers: int = Field(default=1, ge=1, le=4)


    log_level: LogLevel = Field(LogLevel.INFO)
    log_format: str = Field("console",)
    log_retention: str = Field("30 days",)
    log_file:  str= Field("./logs/app.log",)
    log_rotation: str = Field("100 MB",)

    @property
    def MODEL_PATH(self) -> str:  
        return self.model_name

    @property
    def TEMP_DIR(self) -> str:  
        return self.temp_dir


asr_service_config = ASRServiceConfig()  # type: ignore
