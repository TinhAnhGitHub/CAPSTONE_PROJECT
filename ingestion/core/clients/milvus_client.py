from __future__ import annotations
from pymilvus import (
    DataType,
    FieldSchema,
    CollectionSchema,
)
from loguru import logger
from pydantic import BaseModel
from core.clients.base import BaseMilvusClient



class ImageEmbeddingMilvusClient(BaseMilvusClient):

    @property
    def embedding_field(self):
        return 'embedding'

    def get_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name='id', dtype=DataType.VARCHAR, is_primary=True, max_length=128, auto_id=False),
            FieldSchema(name=self.embedding_field, dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
            FieldSchema(
                name="related_video_name",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="related_video_id",
                dtype=DataType.VARCHAR,
                max_length=128
            ),
            FieldSchema(
                name="segment_index",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="minio_url",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name='user_bucket',
                dtype=DataType.VARCHAR,
                max_length=512,
            )
        ]

        return CollectionSchema(
            fields=fields,
            description=self.config.description
        )

    async def exists(
        self,
        id_: str,
        related_video_id: str,
        user_bucket: str
    ):
        filter_expr = (
            f'id == "{id_}" '
            f'and related_video_id == "{related_video_id}" '
            f'and user_bucket == "{user_bucket}"'
        )
        return await self.record_exists(filter_expr)
    



class TextCaptionEmbeddingMilvusClient(BaseMilvusClient):
    
    @property
    def embedding_field(self):
        return 'embedding'

    def get_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
                auto_id=False
            ),
            FieldSchema(
                name=self.embedding_field,
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension
            ),
            FieldSchema(
                name="frame_index",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="related_video_name",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="related_video_id",
                dtype=DataType.VARCHAR,
                max_length=128
            ),
            FieldSchema(
                name="caption",
                dtype=DataType.VARCHAR,
                max_length=10_000
            ),
            FieldSchema(
                name="caption_minio_url",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name='user_bucket',
                dtype=DataType.VARCHAR,
                max_length=512,
            )
        ]

        
        
        return CollectionSchema(
            fields=fields,
            description=self.config.description
        )

    async def exists(
        self,
        id_: str,
        related_video_id: str,
        user_bucket: str
    ):
        filter_expr = (
            f'id == "{id_}" '
            f'and related_video_id == "{related_video_id}" '
            f'and user_bucket == "{user_bucket}"'
        )
        return await self.record_exists(filter_expr)
    

class SegmentCaptionEmbeddingMilvusClient(BaseMilvusClient):
    """Client for segment caption embedding vectors"""

    # AppState attribute name that holds this client's MilvusCollectionConfig
    collection_config_state_attr = "text_segment_caption_milvus_config"

    @property
    def embedding_field(self):
        return 'embedding'

    def get_schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
                auto_id=False
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.config.dimension
            ),
            FieldSchema(
                name="start_frame",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="end_frame",
                dtype=DataType.INT64
            ),
            FieldSchema(
                name="related_video_name",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="related_video_id",
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name="caption",
                dtype=DataType.VARCHAR,
                max_length=10_000
            ),
            FieldSchema(
                name="segment_caption_minio_url",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name='user_bucket',
                dtype=DataType.VARCHAR,
                max_length=512,
            )

        ]

        return CollectionSchema(
            fields=fields,
            description=self.config.description
        )
    
    async def exists(
        self,
        id_: str,
        related_video_id: str,
        user_bucket: str
    ):
        filter_expr = (
            f'id == "{id_}" '
            f'and related_video_id == "{related_video_id}" '
        f'and user_bucket == "{user_bucket}"'
        )
        return await self.record_exists(filter_expr)