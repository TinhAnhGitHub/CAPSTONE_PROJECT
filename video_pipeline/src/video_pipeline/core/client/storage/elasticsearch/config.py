"""Elasticsearch configuration settings."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ElasticsearchSettings(BaseModel):
    """Elasticsearch configuration settings."""

    host: str = Field(default="localhost")
    port: int = Field(default=9200)
    user: str | None = Field(default=None)
    password: str | None = Field(default=None)
    use_ssl: bool = Field(default=False)
    verify_certs: bool = Field(default=True)
    index_name: str = Field(default="video_ocr_docs")
    embedding_dim: int = Field(default=768)  # MMBertClient produces 768-dim embeddings
    timeout: int = Field(default=30)

    @property
    def url(self) -> str:
        """Build Elasticsearch URL."""
        scheme = "https" if self.use_ssl else "http"
        return f"{scheme}://{self.host}:{self.port}"

    def get_client_kwargs(self) -> dict[str, Any]:
        """Get kwargs for AsyncElasticsearch."""
        kwargs: dict[str, Any] = {
            "hosts": [self.url],
            "timeout": self.timeout,
        }
        if self.user and self.password:
            kwargs["basic_auth"] = (self.user, self.password)
        if self.use_ssl:
            kwargs["verify_certs"] = self.verify_certs
        return kwargs