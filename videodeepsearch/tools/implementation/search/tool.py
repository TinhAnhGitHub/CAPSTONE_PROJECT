"""
videodeepsearch/tools/implementation/search/tool.py
"""
from typing import cast
import base64

from ingestion.prefect_agent.service_image_embedding.schema import ImageEmbeddingRequest
from ingestion.prefect_agent.service_text_embedding.schema import TextEmbeddingRequest

from videodeepsearch.tools.clients.milvus.client import (
    ImageFilterCondition, 
    SegmentCaptionFilterCondition,  
)

from videodeepsearch.tools.base.schema import ImageInterface, SegmentInterface
from videodeepsearch.tools.helpers import extract_s3_minio_url
from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.middleware.output import output_image_results, output_segment_results
from videodeepsearch.tools.base.middleware.input import input_simple_images_middleware

from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_WORKER_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles
from videodeepsearch.agent.definition import WORKER_AGENT


from videodeepsearch.core.app_state import (
    get_external_client,
    get_visual_image_client,
    get_storage_client,
    get_segment_milvus,
)

from .arg_alias import (
    VisualQuery, 
    TopK,
    TopKEach, 
    TopkFinal,
    CaptionQuery,
    EventQuery,
    DenseSparseWeight,
    MultiModalWeight   
)


@tool_registry.register(
    group_doc_name=GroupName.SEARCH_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
    output_middleware=output_image_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['list_video_id', 'user_id']
)
async def get_images_from_visual_query(
    visual_query: VisualQuery,
    top_k: TopK,

    # automated added at runtime
    list_video_id: list[str],
    user_id: str
) -> list[ImageInterface]:
    """
    Search for images using visual similarity based on a natural-language description. Encodes the input query into a CLIP-style visual embedding and performs dense similarity search across indexed video frames.
    
    **When to use:**
    - Searching for specific visual appearance (objects, people, scenes, compositions)
    - Query describes what the image LOOKS LIKE, not what's happening
    - Query is in English and visually descriptive
    - Need to find all frames matching a visual pattern across videos

    **When NOT to use:**
    - Query describes events/actions (use get_segments_from_event_query instead)
    - Need to combine visual + textual signals (use get_images_from_multimodal_query)
    - Searching for similar images to a reference image (use find_similar_images_from_image)

    **Typical workflow:**
    1. Optionally enhance query with enhance_visual_query
    2. Call this tool with enhanced or raw visual query
    3. Receive DataHandle with summary
    4. Use view tool result with handle id. 
    5. If results are good, persist evidence, else retry

    
    """
    external_client = get_external_client()
    visual_milvus_client = get_visual_image_client()

    
    embedding_request = ImageEmbeddingRequest(
        text_input=[visual_query],
        image_base64=None,
        metadata={}
    )
    response = await external_client.encode_visual_text(request=embedding_request)
    query_embedding = cast(list[list[float]], response.text_embeddings) 

    filter_condition = ImageFilterCondition(related_video_id=list_video_id, user_bucket=user_id)
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
    return milvus_response


@tool_registry.register(
    group_doc_name=GroupName.SEARCH_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
    output_middleware=output_image_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['list_video_id', 'user_id']
)
async def get_images_from_caption_query(
    caption_query: CaptionQuery,
    top_k_each_request: TopKEach,
    top_k_final: TopkFinal,
    weights: DenseSparseWeight,
    
    list_video_id:list[str],
    user_id: str,
) -> list[ImageInterface]:
    """
    Search images by Vietnamese caption text using semantic embeddings.

    Performs caption-based semantic search over image captions using Vietnamese 
    text embeddings. Supports hybrid retrieval combining dense (semantic) and 
    sparse (keyword) matching with weighted fusion. Best for finding images based 
    on what's described in Vietnamese captions.

    **When to use:**
    - Query is in Vietnamese describing image content
    - Need semantic understanding of caption text, not just visual appearance
    - Searching for events/actions/concepts described in captions
    - Want to combine dense semantic + sparse keyword matching (hybrid search)
    - Caption describes the "story" or "meaning" of the scene

    **When NOT to use:**
    - Query is in English about visual appearance (use get_images_from_visual_query), or translate to Vietnamese.
    - Where other search functions are more suitable.

    **Typical workflow:**
    1. Optionally enhance query with enhance_textual_query
    2. Call this tool with Vietnamese caption query
    3. Receive DataHandle with summary
    4. Inspect results with tool that receive handle id
    5. Optionally verify with tool that retrieve transcript context
    6. Persist evidence if matches are good

    """

    external_client = get_external_client()
    visual_milvus_client = get_visual_image_client()

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

    if weights is not None:
        if len(weights) != 2:
            raise ValueError("weights must contain exactly two elements: [dense_weight, sparse_weight]")
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
        weight=list(weights) if weights is not None else [1.0]
    )

    return milvus_response




@tool_registry.register(
    group_doc_name=GroupName.SEARCH_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
    output_middleware=output_segment_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['list_video_id', 'user_id']
)
async def get_segments_from_event_query(
    event_query: EventQuery,
    top_k_each_request: TopKEach,
    top_k_final: TopkFinal,
    weights: DenseSparseWeight,

    list_video_id: list[str],
    user_id: str,
) -> list[SegmentInterface]:
    """
    Search for video segments by Vietnamese event/scene description.

    Retrieves video segments (multi-frame sequences) based on Vietnamese event 
    descriptions using caption embeddings. Returns segments ranked by semantic 
    similarity to the query. Best for finding continuous actions, events, or 
    scenes described in Vietnamese.

    **When to use:**
    - Query describes an EVENT, ACTION, or SCENE (not a single frame)
    - Need continuous video segments (multiple frames) vs individual images
    - Query is in Vietnamese about what's HAPPENING in the video
    - Looking for temporal sequences (conversations, actions, events)
    - Need semantic understanding of segment captions

    **When NOT to use:**
    - Looking for specific frames

    **Typical workflow:**
    1. Optionally enhance query with enhance_textual_query
    2. Call this tool with Vietnamese event description
    3. Receive DataHandle with segment summary
    4. Inspect results 
    5. Optionally Verify with transcript
    6. Use video navigator  to adjacent segments if needed
    7. Persist evidence for matching segments

    
    """

    segment_milvus_client = get_segment_milvus()
    external_client = get_external_client()
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
    if weights is not None:
        if len(weights) != 2:
            raise ValueError("weights must contain exactly two elements: [dense_weight, sparse_weight]")
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
        weight=list(weights) if weights is not None else [1.0]
    )
    return milvus_response
    
@tool_registry.register(
    group_doc_name=GroupName.SEARCH_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
    output_middleware=output_image_results,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['list_video_id', 'user_id']
)
async def get_images_from_multimodal_query(
    visual_query: VisualQuery,
    caption_query:  CaptionQuery,
    top_k_each_request: TopKEach,
    top_k_final:TopkFinal,
    weights: MultiModalWeight,

    list_video_id: list[str],
    user_id: str,
) -> list[ImageInterface]:
    """
    Search images using combined visual + caption signals (multimodal fusion).

    Performs multimodal retrieval by fusing three search signals: (1) CLIP visual 
    embedding from English query, (2) Vietnamese caption dense embedding, and 
    (3) Vietnamese caption sparse (keyword) matching. Enables finding images that 
    match both visual appearance AND semantic description.

    **When to use:**
    - Need to match BOTH visual appearance AND semantic meaning
    - Query has both visual descriptors (English) and event/context (Vietnamese)
    - Want maximum precision by combining multiple signals
    - User query contains both "what it looks like" and "what's happening"
    - Complex queries requiring multi-faceted matching

    **When NOT to use:**
    - Query is purely visual (use get_images_from_visual_query)
    - Query is purely semantic/event-based (use get_images_from_caption_query)
    - Simple queries where single-modal search suffices
    - Searching for segments vs frames (use get_segments_from_event_query)

    **Typical workflow:**
    1. Split user query into visual (English) and semantic (Vietnamese) components
    2. Optionally enhance with enhance_visual_query + enhance_textual_query
    3. Call this tool with both queries
    4. Receive DataHandle with fused results
    5. Inspect with worker_view_results
    6. Verify matches satisfy both visual AND semantic criteria
    7. Persist high-confidence matches

    - weights: [visual_weight, caption_dense_weight, caption_sparse_weight]
      * [0.5, 0.3, 0.2]: Balanced (visual-focused)
      * [0.33, 0.33, 0.34]: Equal weighting across modalities
      * [0.4, 0.5, 0.1]: Semantic-focused with visual boost

    """

    external_client = get_external_client()
    visual_milvus_client = get_visual_image_client()

    visual_embed_req = ImageEmbeddingRequest(text_input=[visual_query], image_base64=None, metadata={})
    text_embed_req = TextEmbeddingRequest(texts=[caption_query], metadata={})

    visual_resp = await external_client.encode_visual_text(request=visual_embed_req)
    text_resp = await external_client.encode_text(request=text_embed_req)

    visual_emb = cast(list[list[float]], visual_resp.text_embeddings)
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

    if len(weights) != 3:
        raise ValueError("weights must contain exactly three elements: [visual_weight, caption_dense_weight, caption_sparse_weight]")

    milvus_response = await visual_milvus_client.search_combination(
        requests=reqs,
        limit=top_k_final,
        weight=list(weights),
    )

    return milvus_response


# @tool_registry.register(
#     group_doc_name=GroupName.SEARCH_GROUP,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.SEMANTIC_SEARCHER,
#     output_middleware=output_image_results,
#     input_middleware=input_simple_images_middleware
# )
# async def find_similar_images_from_image(
#     image_interface: ImageInterface,
#     top_k:TopK, 
#     # automated added at runtime
#     list_video_id: list[str], # ignore
#     user_id: str # ignore
# ) -> list[ImageInterface]:
    
#     """
#     Retrieves visually similar images from a reference image.

#     1. Load the reference image from MinIO.
#     2. Encode it into a visual embedding.
#     3. Perform a dense visual similarity search in Milvus.
#     4. Rank results by visual similarity.
#     5. Return the top matching images.
#     """
#     external_client = get_external_client()
#     visual_milvus_client = get_visual_image_client()
#     minio_client = get_storage_client()

#     bucket_name, object_name = extract_s3_minio_url(image_interface.minio_path)
#     image_bytes = minio_client.get_object(bucket=bucket_name, object_name=object_name)

#     if image_bytes is None:
#         raise ValueError(f"The image in the path: {image_interface.minio_path} does not exists due to error: {str(image_bytes)}")
    
#     img_b64 = base64.b64encode(image_bytes).decode("utf-8")

#     embedding_request = ImageEmbeddingRequest(text_input=None, image_base64=[img_b64], metadata={})
#     response = await external_client.encode_visual_text(request=embedding_request)
#     query_embedding = cast(list[list[float]], response.image_embeddings)

#     filter_condition = ImageFilterCondition(
#         related_video_id=list_video_id,
#         user_bucket=user_id
#     )
#     request = visual_milvus_client.visual_dense_request(
#         data=query_embedding,
#         limit=top_k,
#         expr=filter_condition.to_expr()
#     )

#     milvus_response = await visual_milvus_client.search_combination(
#         requests=[request],
#         limit=top_k,
#         weight=[1.0]
#     )
#     return milvus_response

