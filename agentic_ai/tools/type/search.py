from typing import Annotated, cast
import base64
from agentic_ai.tools.schema.artifact import ImageObjectInterface, SegmentObjectInterface
from agentic_ai.tools.clients.milvus.client import ImageMilvusClient, ImageFilterCondition, SegmentCaptionFilterCondition, SegmentCaptionImageMilvusClient

from agentic_ai.tools.clients.external.encode_client import ExternalEncodeClient
from ingestion.prefect_agent.service_image_embedding.schema import ImageEmbeddingRequest
from ingestion.prefect_agent.service_text_embedding.schema import TextEmbeddingRequest
from agentic_ai.tools.clients.minio.client import StorageClient

# from agentic_ai.tools.clients.postgre.client import PostgresClient
# from ingestion.core.artifact.schema import ImageCaptionArtifact
# from agentic_ai.tools.clients.minio.client import StorageClient

from .registry import tool_registry
from .helper import extract_s3_minio_url

@tool_registry.register(
    category='search',
    tags=["visual", "semantic", "image"],
    dependencies=[
        "visual_milvus_client",
        "external_client",
        "minio_client",
    ]
)
async def get_images_from_visual_query(
    visual_query: Annotated[
        str,
        "A visually descriptive natural-language query (e.g., 'a red sports car on a wet street at night'). "
        "Avoid non-visual elements such as names, numbers, or abstract concepts."
    ],
    top_k: Annotated[int, "Number of top-matching images to retrieve based on similarity score."],
    # Runtime-Agent run Injection
    list_video_id: Annotated[list[str], "List of video IDs to search within (auto-provided at runtime)."],
    user_id: Annotated[str, "User identifier for context or permissions (auto-provided)."],
    #Runtime-Dependencies Injection
    visual_milvus_client: Annotated[ImageMilvusClient, "Milvus client for semantic retrieval (auto-provided)"],
    external_client: Annotated[ExternalEncodeClient, "External client (auto-provided)"],
) ->  list[ImageObjectInterface]:
    """
    Retrieve visually similar images based on a **visual query**.
    This function performs a semantic search over a visual embedding index (e.g., Milvus)
    using a descriptive visual query, such as *"a sunset over mountains with orange clouds"*.

    (auto-provided): Ignore this parameters
    """

    embedding_request = ImageEmbeddingRequest(
        text_input=[visual_query],
        image_base64=None,
        metadata={}
    )

    response = await external_client.encode_visual_text(request=embedding_request)
    query_embedding = cast(list[list[float]], response.text_embeddings) 
    
    filter_condition = ImageFilterCondition(
        related_video_id=list_video_id,
        user_bucket=user_id
    )

    request = visual_milvus_client.visual_dense_request(
        data=query_embedding,
        limit=top_k,
        expr=filter_condition.to_expr()
    )

    milvus_response = await visual_milvus_client.search_combination(
        requests=[request],
        limit=top_k,
        weight=[1.0]
    )

    result: list[ImageObjectInterface] = []
    for resp in milvus_response:
        image = ImageObjectInterface(
            related_video_id=resp.related_video_id,
            frame_index=resp.frame_index,
            timestamp=resp.timestamp,
            caption_info=resp.image_caption,
            minio_path=resp.image_minio_url,
            score=resp.score,
            reference_query_image=None,
            query=visual_query
        )
        result.append(image)
    

    return result


@tool_registry.register(
    category='search',
    tags=["caption", "semantic", "image"],
    dependencies=[
        "visual_milvus_client",
        "external_client",
        "minio_client",
    ]
)
async def get_images_from_caption_query(
    caption_query: Annotated[
        str,
        "A descriptive text query that semantically aligns with image captions (e.g., 'a person surfing during sunset'). "
        "Use this for retrieving images based on caption embeddings rather than raw visual content."
    ],
    top_k_each_request: Annotated[int, "Number of top-matching images to retrieve based on caption embedding similarity."],
    top_k_final:Annotated[int, "Number of top-matching images to retrieve based on caption embedding similarity."],
    weights: Annotated[tuple[float,float] | None, "If enable, will use the sparse BM25 for hybrid search"],
    
    list_video_id: Annotated[list[str], "List of video IDs to restrict the search domain (auto-provided)."],
    user_id: Annotated[str, "User identifier for context or permissions (auto-provided)."],
    
    visual_milvus_client: Annotated[ImageMilvusClient, "Milvus client for caption-based embedding search (auto-provided)."],
    external_client: Annotated[ExternalEncodeClient, "External encoding client for generating caption embeddings (auto-provided)."],
) -> list[ImageObjectInterface]:
    """
    Retrieve images semantically related to a **caption-based query**.

    This function performs a semantic search using **caption embeddings** instead of raw image embeddings.
    The query is encoded into the same embedding space as stored image captions,
    allowing text-to-image retrieval that aligns with descriptive annotations rather than pure visual features.
    """

    embedding_request = TextEmbeddingRequest(
        texts=[caption_query],
        metadata={}
    )
    response = await external_client.encode_text(request=embedding_request)
    query_embedding = cast(list[list[float]], response.embeddings)

    filter_condition = ImageFilterCondition(
        related_video_id=list_video_id,
        user_bucket=user_id
    )
    reqs = [
        visual_milvus_client.caption_dense_request(
            data=query_embedding,
            limit=top_k_each_request,
            expr=filter_condition.to_expr()
        )
    ]
    
    if weights:
        reqs.append(
            visual_milvus_client.caption_sparse_request(
                data=[caption_query],
                limit=top_k_each_request,
                expr=filter_condition.to_expr()
            )
        )

    milvus_response = await visual_milvus_client.search_combination(
        requests=reqs,
        limit=top_k_final,
        weight=list(weights) if weights else [1.0]
    )

    result = []
    for resp in milvus_response:
        text_object = ImageObjectInterface(
            related_video_id=resp.related_video_id,
            frame_index=resp.frame_index,
            timestamp=resp.timestamp,
            caption_info=resp.image_caption,
            minio_path=resp.image_minio_url,
            score=resp.score,
            reference_query_image=None,
            query=caption_query
        )
        result.append(text_object)
    return result



@tool_registry.register(
    category='search',
    tags=["event", "semantic", "segment"],
    dependencies=[
        "segment_milvus_client",
        "external_client",
    ]
)
async def get_segments_from_event_query(
    event_query: Annotated[
        str,
        "An event-level query (e.g., 'a person starts running', 'a car accident occurs', 'a soccer player scores a goal')."
    ],
    top_k_each_request: Annotated[int, "Number of top results per subquery (dense/sparse)."],
    top_k_final: Annotated[int, "Final number of results to return after hybrid ranking."],
    weights: Annotated[tuple[float, float] | None, "Weights for hybrid dense/sparse reranking. If None, dense-only search."],

    list_video_id: Annotated[list[str], "Video IDs to constrain the search domain (auto-provided)."],
    user_id: Annotated[str, "User identifier (auto-provided)."],
    segment_milvus_client: Annotated[SegmentCaptionImageMilvusClient, "Milvus client for segment-level caption retrieval (auto-provided)."],
    external_client: Annotated[ExternalEncodeClient, "External encoding client for generating text embeddings (auto-provided)."],
) -> list[SegmentObjectInterface]:
    """
    Retrieve temporally localized video segments that semantically match a **natural-language event query**.
    This performs dense or hybrid (dense + sparse) embedding search over segment-level captions.
    """

    embedding_request = TextEmbeddingRequest(
        texts=[event_query],
        metadata={}
    )
    response = await external_client.encode_text(request=embedding_request)
    query_embedding = cast(list[list[float]], response.embeddings)

    filter_condition = SegmentCaptionFilterCondition(
        related_video_id=list_video_id,
        user_bucket=user_id
    )

    reqs = [
        segment_milvus_client.dense_request(
            data=query_embedding,
            limit=top_k_each_request,
            expr=filter_condition.to_expr()
        )
    ]
    if weights:
        reqs.append(
            segment_milvus_client.sparse_request(
                data=[event_query],
                limit=top_k_each_request,
                expr=filter_condition.to_expr()
            )
        )

    milvus_response = await segment_milvus_client.search_combination(
        requests=reqs,
        limit=top_k_final,
        weight=list(weights) if weights else [1.0]
    )

    result = []
    for resp in milvus_response:
        segment = SegmentObjectInterface(
            related_video_id=resp.related_video_id,
            start_frame_index=resp.start_frame,
            end_frame_index=resp.end_frame,
            start_time=resp.start_time,
            end_time=resp.end_time,
            caption_info=resp.segment_caption,
            score=resp.score,
            segment_caption_query=event_query
        )
        result.append(segment)

    return result



@tool_registry.register(
    category='search',
    tags=["multimodal", "visual", "caption", "image"],
    dependencies=[
        "visual_milvus_client",
        "external_client",
    ]
)
async def get_images_from_multimodal_query(
    visual_query: Annotated[
        str,
        "A multimodal query combining visual and textual semantics (e.g., 'a rainy street with a yellow taxi at night')."
    ],
    caption_query:  Annotated[
        str,
        "A multimodal query combining visual and textual semantics (e.g., 'a rainy street with a yellow taxi at night')."
    ],
    top_k_each_request: Annotated[int, "Top results from each modality before reranking."],
    top_k_final: Annotated[int, "Final number of multimodal results to return."],
    weights: Annotated[tuple[float, float, float], "Weights for [visual, caption_dense, caption_sparse] reranking."],

    list_video_id: Annotated[list[str], "Restrict search to these videos (auto-provided)."],
    user_id: Annotated[str, "User identifier (auto-provided)."],
    visual_milvus_client: Annotated[ImageMilvusClient, "Milvus client for multimodal image retrieval (auto-provided)."],
    external_client: Annotated[ExternalEncodeClient, "External encoding client for generating embeddings (auto-provided)."],
) -> list[ImageObjectInterface]:
    """
    Perform **multimodal image retrieval** using combined:
      - Visual embeddings (image encoder)
      - Caption dense embeddings (text encoder)
      - Caption sparse (BM25 or keyword) search
    """

    visual_embed_req = ImageEmbeddingRequest(text_input=[visual_query], image_base64=None, metadata={})
    text_embed_req = TextEmbeddingRequest(texts=[caption_query], metadata={})

    visual_resp = await external_client.encode_visual_text(request=visual_embed_req)
    text_resp = await external_client.encode_text(request=text_embed_req)

    visual_emb = cast(list[list[float]], visual_resp.image_embeddings)
    caption_emb = cast(list[list[float]], text_resp.embeddings)

    filter_condition = ImageFilterCondition(
        related_video_id=list_video_id,
        user_bucket=user_id
    )

    reqs = [
        visual_milvus_client.visual_dense_request(
            data=visual_emb,
            limit=top_k_each_request,
            expr=filter_condition.to_expr(),
        ),
        visual_milvus_client.caption_dense_request(
            data=caption_emb,
            limit=top_k_each_request,
            expr=filter_condition.to_expr(),
        ),
        visual_milvus_client.caption_sparse_request(
            data=[caption_query],
            limit=top_k_each_request,
            expr=filter_condition.to_expr(),
        ),
    ]

    milvus_response = await visual_milvus_client.search_combination(
        requests=reqs,
        limit=top_k_final,
        weight=list(weights),
    )

    result = []
    for resp in milvus_response:
        img = ImageObjectInterface(
            related_video_id=resp.related_video_id,
            frame_index=resp.frame_index,
            timestamp=resp.timestamp,
            caption_info=resp.image_caption,
            minio_path=resp.image_minio_url,
            score=resp.score,
            query=[visual_query,caption_query],
            reference_query_image=None
        )
        result.append(img)
    return result


@tool_registry.register(
    category="search",
    tags=["visual", "semantic", "image", "similarity"],
    dependencies=[
        "visual_milvus_client",
        "external_client",
        "minio_client",
    ]
)
async def find_similar_images_from_image(
    reference_image: ImageObjectInterface,
    top_k: int,
    list_video_id: Annotated[list[str], "Restrict search to these videos (auto-provided)."],
    user_id: Annotated[str, "User identifier (auto-provided)."],
    visual_milvus_client: Annotated[ImageMilvusClient, "Milvus client for multimodal image retrieval (auto-provided)."],
    minio_client:  StorageClient,
    external_client: Annotated[ExternalEncodeClient, "External encoding client for generating embeddings (auto-provided)."],
)-> list[ImageObjectInterface]:

    minio_path = reference_image.minio_path
    bucket_name, object_name = extract_s3_minio_url(minio_path)
    image_bytes = minio_client.get_object(bucket=bucket_name, object_name=object_name)
    if image_bytes is None:
        raise ValueError()

    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    embedding_request = ImageEmbeddingRequest(
        text_input=None,
        image_base64=[img_b64],
        metadata={}
    )

    response = await external_client.encode_visual_text(request=embedding_request)
    query_embedding = cast(list[list[float]], response.image_embeddings)

    filter_condition = ImageFilterCondition(
        related_video_id=list_video_id,
        user_bucket=user_id
    )
    request = visual_milvus_client.visual_dense_request(
        data=query_embedding,
        limit=top_k,
        expr=filter_condition.to_expr()
    )

    milvus_response = await visual_milvus_client.search_combination(
        requests=[request],
        limit=top_k,
        weight=[1.0]
    )
    result: list[ImageObjectInterface]= []
    for resp in milvus_response:
        image = ImageObjectInterface(
            related_video_id=resp.related_video_id,
            frame_index=resp.frame_index,
            timestamp=resp.timestamp,
            caption_info=resp.image_caption,
            minio_path=resp.image_minio_url,
            score=resp.score,
            query=None,
            reference_query_image=reference_image
        )
        result.append(image)
    

    return result
