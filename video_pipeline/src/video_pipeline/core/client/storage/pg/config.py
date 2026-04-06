from pydantic import Field, BaseModel


class PgConfig(BaseModel):
    database_url: str
    pool_size: int = Field(20)
    max_overflow: int = Field(40)
    pool_timeout: int = Field(30)
    pool_recycle: int = Field(1800)
