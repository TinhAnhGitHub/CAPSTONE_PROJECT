from .base import BaseMilvusClient
from .schema import ImageFilterCondition, ImageMilvusResponse, SegmentCaptionMilvusResponse, SegmentCaptionFilterCondition
from typing import cast
from pymilvus import AsyncMilvusClient,AnnSearchRequest,WeightedRanker
from pymilvus import Function, FunctionType

class ImageMilvusClient(BaseMilvusClient[ImageMilvusResponse]):
    def __init__(
            self, 
            uri:str,
            collection_name:str, 
            visual_param: dict,
            caption_param:dict,
            sparse_param:dict,
            visual_ann_field: str = "visual_embedding_field", 
            caption_ann_field:str  = "caption_embedding_field", 
            sparse_field:str = "caption_sparse_embedding_field"
        ):
        super().__init__(
            uri=uri,
            collection=collection_name
        )
        
        self.visual_param = visual_param
        self.caption_param = caption_param
        self.sparse_param = sparse_param
        self.visual_ann_field = visual_ann_field
        self.caption_ann_field = caption_ann_field
        self.sparse_field = sparse_field

    @staticmethod
    def _hit_to_item(hit) -> ImageMilvusResponse:
        fields = hit.fields if hasattr(hit, "fields") else hit.entity["fields"]
        try:
            return ImageMilvusResponse(
                    id=str(fields["id"]),
                    related_video_id=str(fields["related_video_id"]),
                    image_minio_url=str(fields["image_minio_url"]),
                    user_bucket=str(fields["user_bucket"]),
                    timestamp=str(fields['timestamp']),
                    frame_index=int(fields['frame_index']),
                    image_caption=str(fields['image_caption']),
                    score=float(hit.score),
                )
        except Exception as e:
            missing = e.args[0]
            raise KeyError(f"Missing expected field '{missing}' in Milvus hit: {fields}")
    
    def visual_dense_request(
        self,
        data: list[list[float]],
        limit:int,
        expr:str | None
        
    ):
        return AnnSearchRequest(
            data=data,
            anns_field=self.visual_ann_field,
            param=self.visual_param,
            limit=limit,
            expr=expr
        )

    def caption_dense_request(
        self,
        data: list[list[float]],
        limit:int,
        expr:str | None
    ):
        return AnnSearchRequest(
            data=data,
            anns_field=self.caption_ann_field,
            param=self.caption_param,
            limit=limit,
            expr=expr
        )

    def caption_sparse_request(
        self,
        data: list[str],
        limit:int,
        expr:str | None
    ):
        return AnnSearchRequest(
            data=data,
            anns_field=self.sparse_field,
            param=self.sparse_param,
            limit=limit,
            expr=expr
        )


    async def search_combination(
        self,
        requests: list[AnnSearchRequest],
        limit:int,
        weight: list[float]
    ) -> list[ImageMilvusResponse]:
        assert len(requests) == len(weight), "The...."

        rerank = Function(
            name='weight',
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "weighted", 
                "weights": weight,
                "norm_score": True  
            }
        )
        client = cast(AsyncMilvusClient, self.client)

        result = await client.hybrid_search(
            collection_name=self.collection,
            reqs=requests,
            ranker=WeightedRanker(*weight),
            limit=limit,
            output_fields=['*']
        )
        return self._from_hit_to_response(result)

class SegmentCaptionImageMilvusClient(BaseMilvusClient[SegmentCaptionMilvusResponse]):
    def __init__(
        self,
        uri: str,
        collection_name: str,
        dense_param: dict,
        sparse_param: dict,
        dense_field: str = "caption_embedding_field",
        sparse_field: str = "caption_sparse_embedding_field"
    ):
        super().__init__(
            uri=uri,
            collection=collection_name
        )
        self.dense_param = dense_param
        self.sparse_param = sparse_param
        self.dense_field = dense_field
        self.sparse_field = sparse_field
    
    @staticmethod
    def _hit_to_item(hit) -> SegmentCaptionMilvusResponse:
        fields = hit.fields if hasattr(hit, "fields") else hit.entity["fields"]
        try:
            return SegmentCaptionMilvusResponse(
                id=str(fields['id']),
                start_frame=int(fields['start_frame']),
                end_frame=int(fields['end_frame']),
                start_time=str(fields['start_time']),
                end_time=str(fields['end_time']),
                related_video_id=str(fields['related_video_id']),
                segment_caption=str(fields['caption']),
                segment_caption_minio_url=str(fields['segment_caption_minio_url']),
                user_bucket=str(fields['user_bucket']),
                score=float(hit.score)
            )
        except KeyError as e:
            raise KeyError(f"Missing expected field in Milvus hit: {e}") from e

    def dense_request(
        self,
        data: list[list[float]],
        limit: int,
        expr: str | None
    ):
        return AnnSearchRequest(
            data=data,
            anns_field=self.dense_field,
            param=self.dense_param,
            limit=limit,
            expr=expr
        )
    
    def sparse_request(
        self,
        data: list[str],
        limit: int,
        expr: str | None
    ):
        return AnnSearchRequest(
            data=data,
            anns_field=self.sparse_field,
            param=self.sparse_param,
            limit=limit,
            expr=expr
        )
    
    async def search_combination(
        self,
        requests: list[AnnSearchRequest],
        limit: int,
        weight: list[float]
    ) -> list[SegmentCaptionMilvusResponse]:
        assert len(requests) == len(weight), (
            "The number of requests and weights must match."
        )

        rerank = Function(
            name='weight',
            input_field_names=[],
            function_type=FunctionType.RERANK,
            params={
                "reranker": "weighted",
                "weights": weight,
                "norm_score": True
            }
        )
        client = cast(AsyncMilvusClient, self.client)

        result = await client.hybrid_search(
            collection_name=self.collection,
            reqs=requests,
            ranker=WeightedRanker(*weight),
            limit=limit
        )

        if not result:
            return []
        return self._from_hit_to_response(result)
