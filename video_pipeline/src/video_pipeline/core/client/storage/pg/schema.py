from datetime import datetime
from uuid import uuid4
from pydantic import BaseModel, Field

from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import Column, String, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ArtifactMetadata(BaseModel):
    """
    Pydantic model representing the metadata for an artifact stored in the database.
    This captures essential details about the artifact, its origin, and associated data.
    """

    artifact_id: str = Field(
        ...,
        description="Unique identifier for the artifact (e.g., SHA-256 hash or UUID).",
    )
    artifact_type: str = Field(
        ...,
        description="The type of the artifact, such as 'video', 'json', 'image', or 'caption'. Determines the structure and purpose.",
    )
    user_id: str = Field(..., description="User id associated")
    minio_url: str = Field(
        ...,
        description="Full S3/Minio URL to the artifact file (e.g., 's3://bucket/path/to/file').",
    )
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="UTC timestamp when the artifact metadata was created/inserted into the database.",
    )
    artifact_metadata: dict = Field(..., description="related metadata")
    lineage_parents: list[str] = Field(default_factory=list)


class ArtifactSchema(Base):
    __tablename__ = "artifacts_application"

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True, index=True)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    minio_url: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now(), index=True
    )
    artifact_metadata: Mapped[dict] = mapped_column("metadata", JSON, nullable=True)


class ArtifactLineageSchema(Base):
    """Track artifact lineage relationships"""

    __tablename__ = "artifact_lineage_application"

    id = Column(String(128), primary_key=True, default=lambda: uuid4().hex)
    parent_artifact_id = Column(
        String(128),
        ForeignKey("artifacts_application.artifact_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    child_artifact_id = Column(
        String(128),
        ForeignKey("artifacts_application.artifact_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=datetime.now)
