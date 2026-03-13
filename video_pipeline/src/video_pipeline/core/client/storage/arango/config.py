"""ArangoDB storage client configuration."""

from pydantic import BaseModel, Field


class ArangoConfig(BaseModel):
    """Configuration for ArangoDB client."""

    host: str = Field(default="http://localhost:8529")
    database: str = Field(default="video_kg")
    username: str = Field(default="root")
    password: str = Field(default="")
    timeout: int = Field(default=30)


class ArangoIndexConfig(BaseModel):
    """Configuration for ArangoDB vector index."""

    type: str = Field(default="vector")
    name: str
    fields: list[str]
    dimension: int = Field(default=384)
    metric: str = Field(default="cosine")
    n_lists: int = Field(default=10)
    n_probe: int = Field(default=10)
    training_iterations: int = Field(default=40)


class ArangoInvertedIndexConfig(BaseModel):
    """Configuration for ArangoDB inverted index."""

    type: str = Field(default="inverted")
    name: str
    collection: str
    fields: list[dict]
    stored_values: list[str] | None = None
    optimize_top_k: list[str] | None = None