"""LLM Toolkit for query enhancement and generation.

This toolkit provides LLM-powered tools for:
- Visual query enhancement (CLIP-optimized variations in English)
- Textual query enhancement (Vietnamese semantic variations)

Similar to HyDE but with a twist: generates multiple perspective variations
that get searched individually, then aggregated for better recall.

All tools return ToolResult for unified interface.
"""

from __future__ import annotations

from typing import Any

from agno.tools import Toolkit, tool
from agno.tools.function import ToolResult
from pydantic import BaseModel, Field


class EnhancedQueryResponse(BaseModel):
    """Response model for query enhancement tools."""

    queries: list[str] = Field(
        ...,
        description="List of enhanced query variations",
    )


# =============================================================================
# Prompt Templates
# =============================================================================

CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT = """
You are an expert in CLIP prompt engineering and multimodal retrieval.
Your goal is to create contrastively effective visual prompts that maximize alignment
between text and image embeddings.

Guidelines:
- Use natural language templates that match CLIP's training distribution
  (e.g., "a photo of…", "a painting of…", "a detailed picture of…").
- Include relevant visual modifiers such as lighting, color, texture, composition,
  and emotional tone.
- Add descriptive context: background, setting, and object relationships.
- Generate *multiple* prompt variations that capture distinct but semantically
  consistent views of the same concept.
- Avoid domain-unfamiliar jargon unless explicitly required.
- Make sure each prompt is under 25 words and coherent.
- The output must be in English.

Original query: {raw_query}

Perspective variants to generate:
{variants}

Output: A list of optimized CLIP-style prompts suitable for contrastive retrieval.
Return each prompt on a new line.
"""

CAPTION_ENHANCEMENT_PROMPT = """
Bạn là một chuyên gia mô tả hình ảnh và tăng cường truy vấn ngữ nghĩa.
Hãy tạo các câu mô tả ngắn gọn nhưng chi tiết và sinh động cho nội dung được mô tả dưới đây,
nhằm phục vụ cho việc truy xuất hoặc embedding ngữ nghĩa.

Hướng dẫn:
- Tập trung vào *những điểm nổi bật hoặc khác biệt*: đối tượng, mối quan hệ, hành động, bối cảnh.
- Bao gồm *thuộc tính thị giác*: màu sắc, hình dạng, kết cấu, không khí, tâm trạng.
- Ưu tiên mô tả khách quan, tránh nhận xét chủ quan.
- Không dùng các cụm dư thừa như "Bức ảnh này cho thấy..." hay "Đây là hình ảnh của...".
- Mỗi câu mô tả nên ngắn gọn, dưới 30 từ.
- Đầu ra phải bằng **tiếng Việt**.

Truy vấn gốc: {raw_query}

Các biến thể góc nhìn cần sinh ra:
{variants}

Đầu ra: Danh sách các câu mô tả tối ưu cho truy xuất ngữ nghĩa.
Mỗi câu trên một dòng mới.
"""


class LLMToolkit(Toolkit):
    """Toolkit for LLM-powered query enhancement.

    Provides tools for enhancing search queries with multiple
    perspective variations, improving retrieval recall.

    Similar to HyDE (Hypothetical Document Embeddings) but generates
    query variations instead of hypothetical documents.

    All tools return ToolResult for unified interface.
    """

    def __init__(
        self,
        llm_client: Any,  # agno LLM model instance
        name: str = "LLM Enhancement Tools",
    ):
        """Initialize the LLMToolkit.

        Args:
            llm_client: agno LLM model instance (e.g., Gemini, Claude, OpenAIChat)
            name: Toolkit name
        """
        self.llm = llm_client
        super().__init__(name=name)

    @tool(
        description=(
            "Generate CLIP-optimized visual query variations for image search. "
            "Takes a short visual query and expands it into multiple detailed "
            "variations optimized for visual embedding retrieval. "
            "Use BEFORE visual search tools to improve recall."
        ),
        instructions=(
            "Use when: query describes visual appearance (objects, scenes, colors, compositions), "
            "query is vague or under-specified (< 10 words), "
            "need multiple search angles for comprehensive retrieval. "
            "Query MUST be in English. "
            "NOT for event/action queries - use enhance_textual_query instead."
        ),
    )
    async def enhance_visual_query(
        self,
        raw_query: str,
        variants: list[str],
    ) -> ToolResult:
        """Generate detailed CLIP-optimized variations of a visual query.

        Takes a short visual query describing objects, scenes, or visual patterns
        and expands it into multiple detailed variations optimized for CLIP-style
        visual embedding retrieval.

        Best for queries about WHAT THINGS LOOK LIKE (appearance, not events).

        Args:
            raw_query: The user's original visual query in English.
                      Describes objects, actions, scenes, or visual intent.
            variants: Perspective variations to generate. Each variant should be
                     a specific viewpoint/visual-demand description emphasizing
                     different aspects (camera angle, lighting, composition, etc.)

        Returns:
            ToolResult with enhanced query variations (one per line)
        """
        variants_str = "\n".join(f"- {v}" for v in variants)

        prompt = CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT.format(
            raw_query=raw_query,
            variants=variants_str,
        )

        try:
            response = await self.llm.arun(prompt)

            # Parse response - each line is a query
            content = response.content if hasattr(response, "content") else str(response)
            enhanced_queries = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            result_content = (
                f"Generated {len(enhanced_queries)} visual query variations:\n\n"
                + "\n".join(f"{i+1}. {q}" for i, q in enumerate(enhanced_queries))
            )

            return ToolResult(content=result_content)

        except Exception as e:
            return ToolResult(content=f"Error enhancing visual query: {e}")

    @tool(
        description=(
            "Generate Vietnamese semantic query variations for caption/event search. "
            "Takes a Vietnamese query describing events, actions, or scenes "
            "and expands it into multiple detailed semantic variations. "
            "Use BEFORE caption/segment search tools to improve recall."
        ),
        instructions=(
            "Use when: query is in Vietnamese (translate if English), "
            "query describes events, actions, or scene-level meaning, "
            "query is ambiguous or short. "
            "NOT for pure visual appearance queries - use enhance_visual_query instead. "
            "Best paired with: get_images_from_caption_query_mmbert, get_segments_from_event_query_mmbert"
        ),
    )
    async def enhance_textual_query(
        self,
        raw_query: str,
        variants: list[str],
    ) -> ToolResult:
        """Generate rich Vietnamese semantic variations for event/scene queries.

        Takes a Vietnamese query describing events, actions, or scene-level meaning
        and expands it into multiple detailed variations optimized for caption/segment
        embedding retrieval.

        Unlike enhance_visual_query (which focuses on visual appearance), this tool
        emphasizes semantic context and narrative structure.

        Args:
            raw_query: The user's original query in Vietnamese.
                      Describes events, actions, scenes, or semantic intent.
                      If in English, translate to Vietnamese first.
            variants: Perspective variations to generate. Each variant should
                     describe a different semantic angle or narrative perspective.

        Returns:
            ToolResult with enhanced query variations (one per line)
        """
        variants_str = "\n".join(f"- {v}" for v in variants)

        prompt = CAPTION_ENHANCEMENT_PROMPT.format(
            raw_query=raw_query,
            variants=variants_str,
        )

        try:
            response = await self.llm.arun(prompt)

            content = response.content if hasattr(response, "content") else str(response)
            enhanced_queries = [
                line.strip()
                for line in content.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]

            result_content = (
                f"Generated {len(enhanced_queries)} textual query variations:\n\n"
                + "\n".join(f"{i+1}. {q}" for i, q in enumerate(enhanced_queries))
            )

            return ToolResult(content=result_content)

        except Exception as e:
            return ToolResult(content=f"Error enhancing textual query: {e}")


__all__ = ["LLMToolkit"]