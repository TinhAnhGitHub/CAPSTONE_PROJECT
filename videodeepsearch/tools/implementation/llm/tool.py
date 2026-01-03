"""
videodeepsearch/tools/implementation/llm/tool.py
This contains the tools for query expansion, query enhancement
"""
from typing import Annotated, cast
from pydantic import BaseModel, Field

from videodeepsearch.core.app_state import get_llm_instance
from .prompt import CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT, CAPTION__ENHANCEMENT_PROMPT
from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName

from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_WORKER_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles

from videodeepsearch.agent.definition import WORKER_AGENT


class EVQResponse(BaseModel):
    """Enhance visual query response"""
    resp: list[str] = Field(..., description="The list of enhanced response")

@tool_registry.register(
    group_doc_name=GroupName.UTILITY,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.QUERY_ANALYZER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT]
)
async def enhance_visual_query(
    raw_query: Annotated[
        str,
        "The user's original visual query. This may describe objects, actions, scenes, or any visual intent "
        "the user wants to search for in the video. The query must be in English"
    ],
    variants: Annotated[
        list[str],
        (
            "EXTREME-DETAIL perspective variations to generate. Each variant should be a highly-specific "
            "viewpoint/visual-demand description i.e. precise camera perspective, framing, scale, motion, "
            "and occlusion details. Each variant should represent an extremely detailed reinterpretation of the original content, emphasizing meaningful visual divergence. Variants may include changes in spatial perspective (e.g., different viewing angles, camera positions, scale shifts), alterations in scene composition, modifications of lighting or atmosphere, and nuanced adjustments to texture, depth, geometry, or structural layout—while still preserving the core semantic identity of the original."
        )
    ]
) -> str:    
    """
    Generate detailed CLIP-optimized variations of a short visual query
    
    When to use:
    - User query is vague or under-specified (< 10 words)
    - Need multiple search angles for comprehensive retrieval
    - Before calling any tools related to visual semantic search.
    - To improve recall when initial search returns bad results.

    When not to use:
    - Query is related to some sort of events. This tool only enhance visual stuff.
    - If the query contain any keywords related to events, name, places,... then use the enhance_textual_query(), in Vietnamese
    - Query describes events/actions rather than visual appearance (use enhance_textual_query)
    - For segment/scene-level queries (use enhance_textual_query)

    """


    llm_as_tools = get_llm_instance()
    prompt = CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT.format(raw_query=raw_query, variants=variants)
    sllm = llm_as_tools.as_structured_llm(EVQResponse)
    response = await sllm.acomplete(prompt)
    return '\n'.join(cast(EVQResponse, response.raw).resp)

@tool_registry.register(
    group_doc_name=GroupName.UTILITY,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.QUERY_ANALYZER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT]
)
async def enhance_textual_query(
    raw_query: Annotated[str, "Câu truy vấn gốc của người dùng (văn bản hoặc ý định sự kiện). Bằng Tiếng Việt"],
    variants: Annotated[list[str], "EXTREME-DETAIL: Danh sách các phương án biến thể cần sinh ra. Mỗi biến thể phải mô tả cực kỳ chi tiết về góc nhìn/camera, bố cục, tỉ lệ, chuyển động, độ che khuất, và toàn bộ yêu cầu thị giác liên quan. Mỗi biến thể là một diễn giải lại nội dung gốc với mức độ khác biệt có ý nghĩa, nhấn mạnh thay đổi về phối cảnh (ví dụ: góc nhìn khác, vị trí camera, độ xa/gần), cấu trúc bố cục, ánh sáng-không khí, cũng như tinh chỉnh chất liệu, độ sâu, hình học hoặc bố trí không gian—nhưng vẫn giữ nguyên bản sắc ngữ nghĩa cốt lõi của nội dung gốc."
],
) -> str:
    """
    Generate rich Vietnamese semantic variations for event/scene queries.

    Takes a Vietnamese query describing events, actions, or scene-level meaning 
    and expands it into multiple detailed variations optimized for caption/segment 
    embedding retrieval. Unlike enhance_visual_query (which focuses on visual 
    appearance), this tool emphasizes semantic context and narrative structure.

    **When to use:**
    - Query is in Vietnamese, if in english, translate to Vietnamese.
    - Query describes events, actions, or scene-level meaning.
    - It also could work with visual appearance, with image captioning.
    - Before calling get_images_from_caption_query or get_segments_from_event_query
    - Need to improve recall for semantic/event-based searches, or word-level caption visual.
    - Query is ambiguous or short 

    **When NOT to use:**
    - Query describes only visual appearance without events (use enhance_visual_query)
    - Query is already detailed and specific (> 20 words)
    - For pure visual similarity search (use enhance_visual_query)

    **Typical workflow:**
    1. Receive user's Vietnamese query about events/scenes
    2. Call this tool with 3-5 semantic perspective variants
    3. Use returned enhanced queries with get_images_from_caption_query or get_segments_from_event_query
    4. Inspect results with worker_view_results
    5. Optionally verify with ASR context using get_related_asr_from_video_id

    

    """


    llm_as_tools = get_llm_instance()
    prompt = CAPTION__ENHANCEMENT_PROMPT.format(raw_query=raw_query, variants=variants)
    sllm = llm_as_tools.as_structured_llm(EVQResponse)
    response = await sllm.acomplete(prompt)
    return '\n'.join(cast(EVQResponse, response.raw).resp)