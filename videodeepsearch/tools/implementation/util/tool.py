"""
videodeepsearch/tools/implementation/util/tool.py
Some helpful utility tools
"""
from typing import cast
from ingestion.core.artifact.schema import ASRArtifact
from ingestion.prefect_agent.service_asr.core.schema import  ASRResult

from videodeepsearch.core.app_state import (
    get_storage_client,
    get_postgres_client
)

from videodeepsearch.tools.helpers import (
    extract_s3_minio_url, 
    time_to_seconds,
    time_range_overlap,
    convert_time_datetime_to_frame
)

from videodeepsearch.tools.clients.milvus.client import ( 
    ImageInterface,
    SegmentInterface
)

from videodeepsearch.tools.clients.milvus.client import (
    SegmentInterface
)
from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.middleware.output import output_image_results, output_segment_results
from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_WORKER_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles

from .arg_alias import (
    WindowSeconds,
)

from .str_template import SNIPPET, ASR_TOKEN_TEMPLATE
from videodeepsearch.agent.definition import WORKER_AGENT


@tool_registry.register(
    group_doc_name=GroupName.CONTEXT_RETRIEVE_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.TRANSCRIPT_ANALYZER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT]
)
async def get_related_asr_from_video_id(
    segment_related_video_id: str,
    segment_start_time: str,
    segment_end_time: str,
    window_seconds: WindowSeconds,
)-> str:
    """
    Retrieve spoken words (ASR transcript) around a video segment for context.

    Extracts the audio transcript within a time window surrounding a segment, 
    providing spoken context to verify or enrich visual findings. Useful for 
    grounding visual evidence with what was being said at that moment.

    **When to use:**
    - Found a promising segment/image but need verbal context
    - User query mentioned dialogue, speech, or audio events
    - Want to verify if visual match aligns with spoken content
    - Need to distinguish between similar-looking scenes via dialogue

    **When NOT to use:**
    - User query is purely visual (no mention of speech/audio)
    - Segment is from a video with no audio/ASR data
    - Already have sufficient evidence without audio context
    - Window would extend beyond video boundaries

    **Typical workflow:**
    1. Get segment from search 
    2. Inspect segment details 
    3. Call this tool to see what was spoken during that segment
    4. Use ASR context to confirm or refute relevance
    """
    postgres_client = get_postgres_client()
    minio_client = get_storage_client()
    related_video_id = segment_related_video_id

    start_time = segment_start_time
    end_time = segment_end_time

    video_artifact = await postgres_client.get_artifact(
        artifact_id=related_video_id
    )
    if video_artifact is None:
        raise ValueError(f"Video id: {related_video_id} does not exist")
    
    video_asr_artifacts = await postgres_client.get_children_artifact(
        artifact_id=related_video_id,
        filter_artifact_type=[ASRArtifact.__name__]
    ) 
    if not video_asr_artifacts:
        raise ValueError(f"The video id {related_video_id} might not exist.")

    video_asr_artifact = video_asr_artifacts[0]
    
    video_fps = cast(float, video_artifact.artifact_metadata['fps'])
    duration = cast(str, video_artifact.artifact_metadata['duraion'])
    
    segment_start = time_to_seconds(start_time)
    segment_end = time_to_seconds(end_time)
    video_duration = time_to_seconds(duration)

    if segment_start < 0:
        raise ValueError(f"start_time must be >= 0, got: {start_time}")

    if segment_end > video_duration:
        raise ValueError(
            f"end_time ({end_time}) exceeds video duration ({duration})"
        )

    if segment_start >= segment_end:
        raise ValueError(
            f"start_time ({start_time}) must be < end_time ({end_time})"
        )


    bucket_name, object_name = extract_s3_minio_url(video_asr_artifact.minio_url)

    minio_object = cast(dict, minio_client.read_json(bucket=bucket_name, object_name=object_name))
    asr_object = ASRResult.model_validate(minio_object)
    asr_tokens = asr_object.tokens


    
    window_start = segment_start - window_seconds
    window_end = segment_end + window_seconds

    snippet_tokens = [
        t for t in asr_tokens
        if time_range_overlap(
            start_input=window_start,
            end_input=window_end,
            start_range=time_to_seconds(t.start),
            end_range=time_to_seconds(t.end),
            iou=0.2
        )
    ]

    context = []
    for token in snippet_tokens:
        context_token = ASR_TOKEN_TEMPLATE.format(token=token)
        context.append(context_token.strip())
    
    return_snippet = SNIPPET.format(
        segment_start_time=start_time,
        segment_end_time=end_time,
        segment_start_frame=convert_time_datetime_to_frame(time=start_time, fps=video_fps),
        segment_end_frame=convert_time_datetime_to_frame(time=end_time, fps=video_fps),
        window_seconds=window_seconds,
        context="\n\n".join(context)
    )

    return return_snippet


@tool_registry.register(
    group_doc_name=GroupName.CONTEXT_RETRIEVE_GROUP,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.TRANSCRIPT_ANALYZER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT]
)
async def get_related_asr_from_image(
    image_related_video_id: str,
    image_timestamp: str,
    window_seconds: WindowSeconds,
)->str:
    """
    Retrieve spoken words (ASR transcript) around a specific image/frame.
    
    Extracts audio transcript within a time window surrounding an image's timestamp,
    providing spoken context to verify or enrich visual findings. Similar to
    get_related_asr_from_video_id but works with single frames instead of segments.

    **When to use:**
    - Found a promising image but need verbal context around that moment
    - User query mentioned dialogue, speech, or audio events at specific frames
    - Want to verify if visual match aligns with spoken content
    - Need to distinguish between similar-looking frames via dialogue
    
    **When NOT to use:**
    - User query is purely visual (no mention of speech/audio)
    - Image is from a video with no audio/ASR data
    - Already have sufficient evidence without audio context
    - Working with segments (use get_related_asr_from_video_id instead)

    **Typical workflow:**
    1. Get image from search 
    2. Inspect image details 
    3. Call this tool to see what was spoken at that frame's timestamp
    4. Use ASR context to confirm or refute relevance
    5. Persist as evidence if ASR supports visual match

    """
    postgres_client = get_postgres_client()
    minio_client = get_storage_client()


    parent_video_id = image_related_video_id

    video_asr_artifacts = await postgres_client.get_children_artifact(
        artifact_id=parent_video_id,
        filter_artifact_type=[ASRArtifact.__name__]
    )  

    video_artifact = await postgres_client.get_artifact(
        artifact_id=image_related_video_id
    )
    if video_artifact is None:
        raise ValueError(f"Video id: {image_related_video_id} does not exist")
    
  
    if not video_asr_artifacts:
        raise ValueError(f"The video id {image_related_video_id} might not exist.")
    
    
    video_asr_artifact = video_asr_artifacts[0]
    video_fps = cast(float, video_artifact.artifact_metadata['fps'])
    bucket_name, object_name = extract_s3_minio_url(video_asr_artifact.minio_url)
    minio_object = cast(dict, minio_client.read_json(bucket=bucket_name, object_name=object_name))
    asr_object = ASRResult.model_validate(minio_object)

    asr_tokens = asr_object.tokens

    timestamp = image_timestamp

    ts_center = time_to_seconds(timestamp)
    window_start = ts_center - window_seconds
    window_end = ts_center + window_seconds
    
    snippet_tokens = [
        t for t in asr_tokens
        
        if time_range_overlap(
            start_input=window_start,
            end_input=window_end,
            start_range=time_to_seconds(t.start),
            end_range=time_to_seconds(t.end),
            iou=0.2
        )
    ]

    snippet = """
    Here are the asr captured around the image at timestamp/frame_index: {image_timestamp}/{image_frame_index} around the window seconds: {window_seconds}
    {context}
    The asr might have irrelevant events/context, so just focus on the related asr segments!
    The context around:{context}
    """

    context = []
    for token in snippet_tokens:
        context_token = f"""
        ---------------------
        Start time/index: {token.start}/{token.start_frame} - End time/index: {token.end}/{token.end_frame}
        ASR content: {token.text}
        ---------------------
        """
        context.append(context_token)
    
    frame_index = convert_time_datetime_to_frame(time=timestamp, fps=video_fps)
    return_snippet = snippet.format(
        image_timestamp=timestamp,
        image_frame_index=frame_index,
        window_seconds=window_seconds,
        context='\n\n'.join(context)
    )

    return return_snippet

