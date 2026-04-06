from beanie import Document
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

from app.schemas.application import (
    KeyframeScore,
    CaptionSearch,
    KeyframeSearch,
    OCRSearch,
)


class KeyframeModel(Document):
    identification: int = Field(
        ..., description="A unique identifier for the keyframe."
    )
    group_id: str = Field(..., description="The group ID of the keyframe.")
    video_id: str = Field(..., description="The video ID associated with the keyframe.")
    keyframe_id: str = Field(..., description="The unique ID of the keyframe.")
    tags: list[str] | None = None

    class Settings:
        collection = "keyframes"
        indexes = [
            "group_id",
            "video_id",
            "keyframe_id",
            {"fields": ["identification"], "unique": True},
        ]


class ChatHistory(Document):
    """
    Represents a chat or question history item.
    """

    question_filename: str = Field(
        ..., description="The name/identifier of the question or search."
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When the history item was created."
    )

    return_images: list[KeyframeScore] = Field(
        ..., description="Search images associated with this history."
    )

    keyframe_search_text: KeyframeSearch | None = Field(
        None, description="The keyframe search text"
    )
    caption_search_text: CaptionSearch | None = Field(
        None, description="The caption search text"
    )
    ocr: OCRSearch | None = Field(None, description="List of OCR for matching")

    rerank: Literal["rrf", "weigted"] | None = Field(
        None, description="Enable if both caption, keyframe search and OCR"
    )
    weights: list[float] | None = Field(
        None,
        description="Weighted of visual embedding. Caption search will be (1-keyframe embedding)",
    )


from pydantic import BaseModel, Field
from typing import Literal
from typing_extensions import Annotated, Union


class KeyframeInstance(BaseModel):
    group_id: str
    video_id: str
    keyframe_id: str
    identification: int = Field(
        ...,
        description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection",
    )
    tags: list[str] | None = Field(
        None, description="List of tags associated with the keyframe"
    )
    ocr: list[str] | None = Field(
        None, description="List of OCR texts associated with the keyframe"
    )


class KeyframeScore(KeyframeInstance):
    score: float


class MilvusSearchRequestInput(BaseModel):
    """
    Input schema for Milvus vector search requests.
    """

    embedding: list[float] = Field(
        ..., description="The embedding vector to search for."
    )
    top_k: int = Field(..., description="The number of top similar items to retrieve.")


class MilvusSearchResponseItem(BaseModel):
    """
    Response item schema for Milvus vector search results.
    """

    identification: int = Field(
        ...,
        description="The identification of the keyframe, corresponding to the index of the embeddings in the Milvus Collection",
    )
    score: float = Field(..., description="The similarity score of the retrieved item.")


class TagInstance(BaseModel):
    tag_name: str
    tag_score: float


class CaptionSearch(BaseModel):
    type_search: str = "caption_search"
    caption_search_text: str = Field(..., description="The keyframe search text")
    mode: Literal["rrf", "weighted"]
    weighted: float | None = Field(
        None,
        description="The weighted if using weighted, of the embedding. The bm25 will be (1 - embedding_weight)",
    )
    tag_boost_alpha: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Tag boost alpha, if 0.0 then it will not be used",
    )


class KeyframeSearch(BaseModel):
    type_search: str = "keyframe_search"
    keyframe_search_text: str = Field(..., description="The keyframe search text")
    tag_boost_alpha: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Tag boost alpha, if 0.0 then it will not be used",
    )


class OCRSearch(BaseModel):
    type_search: str = "ocr_search"
    list_ocr: str = Field(..., description="List of OCR")


# class EventOrder(BaseModel):
#     """
#     Event Text, chunked from the original text, with order to indicate the sequence.
#     """
#     order: int
#     event_text: str


class EventSearch(BaseModel):
    keyframe_search: KeyframeSearch | None = Field(None, description="Keyframe search")
    caption_search: CaptionSearch | None = Field(None, description="Caption search")
    ocr_search: OCRSearch | None = Field(None, description="OCR search")
    event_order: int = Field(..., description="The event order")


class EventHit(BaseModel):
    search_setting: EventSearch = Field(..., description="Search setting")
    video_id: str
    keyframe_id: str
    group_id: str
    score: float
