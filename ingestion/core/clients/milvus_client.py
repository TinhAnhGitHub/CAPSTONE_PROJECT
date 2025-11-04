from __future__ import annotations
from pymilvus import (
    DataType,
    FieldSchema,
    Function,
    FunctionType,
    CollectionSchema,
)
from typing import cast
from core.clients.base import BaseMilvusClient

class ImageMilvusClient(BaseMilvusClient):

    @property
    def visual_embedding_field(self):
        return "visual_embedding_field"

    @property
    def caption_embedding_field(self):
        return "caption_embedding_field"

    @property
    def caption_sparse_embedding_field(self):
        return "caption_sparse_embedding_field"

    @property
    def caption_text_field(self):
        return "image_caption"


    def get_schema(self) -> CollectionSchema:
        function = None
        fields = [
            FieldSchema(
                name='id', 
                dtype=DataType.VARCHAR, 
                is_primary=True, 
                max_length=128, 
                auto_id=False
            ),
            FieldSchema(
                name="related_video_id",
                dtype=DataType.VARCHAR,
                max_length=128
            ),
            FieldSchema(
                name="image_minio_url",
                dtype=DataType.VARCHAR,
                max_length=512
            ),
            FieldSchema(
                name='user_bucket',
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name='frame_index',
                dtype=DataType.INT64
            ),
            FieldSchema(
                name='timestamp',
                dtype=DataType.VARCHAR,
                max_length=512,
            )
        ]

        if self.visual_index_config:
            fields.append(
                FieldSchema(
                    name=self.visual_embedding_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.visual_index_config.dimension
                )
            )
        if self.caption_dense_index_config:
            fields.append(
                FieldSchema(
                    name=self.caption_embedding_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.caption_dense_index_config.dimension
                )
            )
        
        if self.caption_sparse_embedding_field and self.caption_text_field:
            fields.append(
                FieldSchema(
                    name=self.caption_sparse_embedding_field,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                )
            )
            fields.append(
                FieldSchema(
                    name=self.caption_text_field,
                    dtype=DataType.VARCHAR,
                    max_length=60_000,
                    enable_analyzer=True,
                )
            )
            function = Function(
                name="text_bm25_emb",
                input_field_names=[self.caption_text_field],  
                output_field_names=[self.caption_sparse_embedding_field],  
                function_type=FunctionType.BM25
            )
            

        return CollectionSchema(
            fields=fields,
            functions=[function]
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
    """Client for segment caption embedding vectors, with dense + sparse schema and BM25 function"""

    @property
    def visual_embedding_field(self):
        return None  

    @property
    def caption_embedding_field(self):
        return "caption_embedding_field"

    @property
    def caption_sparse_embedding_field(self):
        return "caption_sparse_embedding_field"

    @property
    def caption_text_field(self):
        return "segment_caption"

    def get_schema(self) -> CollectionSchema:
        function = None
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                is_primary=True,
                max_length=128,
                auto_id=False,
            ),
            FieldSchema(
                name="related_video_id",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="start_frame",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="end_frame",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="start_time",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="end_time",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="segment_caption_minio_url",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
            FieldSchema(
                name="user_bucket",
                dtype=DataType.VARCHAR,
                max_length=512,
            ),
        ]

        if self.caption_dense_index_config:
            fields.append(
                FieldSchema(
                    name=self.caption_embedding_field,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.caption_dense_index_config.dimension,
                )
            )

        if self.caption_sparse_embedding_field and self.caption_text_field:
            fields.append(
                FieldSchema(
                    name=self.caption_sparse_embedding_field,
                    dtype=DataType.SPARSE_FLOAT_VECTOR,
                    
                )
            )
            fields.append(
                FieldSchema(
                    name=self.caption_text_field,
                    dtype=DataType.VARCHAR,
                    max_length=10_000,
                    enable_analyzer=True,
                )
            )
            function = Function(
                name="segment_caption_bm25_emb",
                input_field_names=cast(str, self.caption_text_field),
                output_field_names=cast(str, self.caption_sparse_embedding_field),
                function_type=FunctionType.BM25,
            )

        return CollectionSchema(fields=fields, functions=[function] if function else [])

    async def exists(
        self,
        id_: str,
        related_video_id: str,
        user_bucket: str,
    ):
        filter_expr = (
            f'id == "{id_}" '
            f'and related_video_id == "{related_video_id}" '
            f'and user_bucket == "{user_bucket}"'
        )
        return await self.record_exists(filter_expr)
