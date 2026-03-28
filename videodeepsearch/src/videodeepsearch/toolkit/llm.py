"""LLM Toolkit for query enhancement and generation.

This toolkit provides LLM-powered tools for:
- Visual query enhancement (CLIP-optimized variations)
- Textual query enhancement (semantic variations)

Similar to HyDE but with a twist: generates multiple perspective variations
that get searched individually, then aggregated for better recall.

All tools return ToolResult for unified interface.
"""

from __future__ import annotations

from typing import Any

from agno.models.base import Model
from agno.tools import Toolkit, tool
from agno.models.message import Message
from agno.tools.function import ToolResult
from agno.models.openrouter import OpenRouter
from pydantic import BaseModel, Field


class EnhancedQueryResponse(BaseModel):
    """Response model for query enhancement tools."""

    queries: list[str] = Field(
        ...,
        description="List of enhanced query variations",
    )


VISUAL_ENHANCE_SYSTEM_PROMPT ="""
You are an expert in multimodal embedding optimization for Qwen-VL models.

Your goal is to generate semantically precise and visually grounded descriptions
that improve alignment between text and image embeddings.

Guidelines:
- Focus on clear, concrete descriptions of objects, attributes, and relationships.
- Describe what is visibly present, not artistic style or abstract interpretation.
- Include details such as color, shape, position, actions, and environment.
- Use simple, direct language without poetic or stylistic phrasing.
- Avoid templates like "a photo of..." unless necessary.
- Each prompt should represent a distinct semantic perspective of the same concept.
- Keep each prompt under 25 words.
- Ensure all outputs are in English.
"""



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

CAPTION_ENHANCEMENT_SYSTEM_PROMPT = """
You are an expert in image description and semantic query enhancement.
Create concise but detailed and vivid descriptions for the content described below,
for the purpose of semantic retrieval or embedding.

Guidelines:
- Focus on *notable or distinctive features*: subjects, relationships, actions, context.
- Include *visual attributes*: color, shape, texture, atmosphere, mood.
- Prioritize objective descriptions, avoid subjective commentary.
- Do not use redundant phrases like "This image shows..." or "This is an image of...".
- Each description should be concise, under 30 words.
- Output must be in **English**.
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
        llm_client: OpenRouter, 
        name: str = "LLM Enhancement Tools",
    ):
        """Initialize the LLMToolkit.

        Args:
            llm_client: agno LLM model instance (e.g., Gemini, Claude, OpenRouter)
            name: Toolkit name
        """
        self.llm = llm_client
        super().__init__(
            name=name,
            tools=[
                self.enhance_visual_query,
                self.enhance_textual_query,
            ],
        )

    async def _invoke_llm(self, system_prompt: str, prompt: str) -> str:
        user_messages = [
            Message(
                role='user',
                content=prompt
            )
        ]
        assistant_message = Message(role='assistant', content=system_prompt)
        response = await self.llm.ainvoke(messages=user_messages, assistant_message=assistant_message)
        
        if hasattr(response, "content"):
            return response.content #type:ignore
        if hasattr(response, "messages") and response.messages: #type:ignore
            for msg in reversed(response.messages): #type:ignore
                if getattr(msg, "role", None) == "assistant":
                    return getattr(msg, "content", str(msg))
        return str(response)

    @tool(
        description=(
            "Generate CLIP-optimized visual query variations for image search. "
            "Takes a short visual query and expands it into multiple detailed "
            "variations optimized for visual embedding retrieval. "
            "Use BEFORE visual search tools to improve recall.\n\n"
            "Typical workflow - Query enhancement for visual search:\n"
            "  1. This tool - generate multiple visual query variations\n"
            "  2. search.get_images_from_qwenvl_query - search with each variation\n"
            "  3. Aggregate results from multiple searches for better recall\n"
            "  4. utility.get_related_asr_from_image - get context for findings\n\n"
            "When to use:\n"
            "  - Query describes visual appearance (objects, scenes, colors, compositions)\n"
            "  - Query is vague or under-specified (< 10 words)\n"
            "  - Need multiple search angles for comprehensive retrieval\n\n"
            "Related tools:\n"
            "  - enhance_textual_query: For event/action/semantic queries (not visual)\n"
            "  - search.get_images_from_qwenvl_query: Target tool for enhanced visual queries\n"
            "  - search.get_images_from_caption_query_mmbert: Alternative visual search\n\n"
            "Args:\n"
            "  raw_query (str): The user's original visual query in English. Describes objects, "
            "actions, scenes, or visual intent (REQUIRED)\n"
            "  variants (list[str]): Perspective variations to generate. Each variant should be "
            "a specific viewpoint/visual-demand description (REQUIRED)"
        ),
        instructions=(
            "Use when: query describes visual appearance (objects, scenes, colors, compositions), "
            "query is vague or under-specified (< 10 words), "
            "need multiple search angles for comprehensive retrieval.\n\n"
            "Query MUST be in English.\n\n"
            "Best paired with: search.get_images_from_qwenvl_query, search.get_images_from_caption_query_mmbert (use enhanced queries for search). "
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
        
        user_prompt = f"""
        Original query: {raw_query}

        Perspective variants to generate:
        {variants_str}

        Output:
        A list of semantically grounded descriptions optimized for Qwen-VL embeddings.
        Return each prompt on a new line.
        """

        try:
            content = await self._invoke_llm(
                system_prompt=VISUAL_ENHANCE_SYSTEM_PROMPT,
                prompt=user_prompt
            )

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
            "Generate semantic query variations for caption/event search. "
            "Takes a query describing events, actions, or scenes "
            "and expands it into multiple detailed semantic variations. "
            "Use BEFORE caption/segment search tools to improve recall.\n\n"
            "Typical workflow - Query enhancement for semantic search:\n"
            "  1. This tool - generate multiple semantic query variations\n"
            "  2. search.get_images_from_caption_query_mmbert - search images with variations\n"
            "  3. Or search.get_segments_from_event_query_mmbert - search segments with variations\n"
            "  4. Aggregate results from multiple searches for better recall\n\n"
            "When to use:\n"
            "  - Query describes events, actions, or scene-level meaning\n"
            "  - Query is ambiguous or short\n"
            "  - Need multiple semantic angles for comprehensive retrieval\n\n"
            "Related tools:\n"
            "  - enhance_visual_query: For visual appearance queries (not events)\n"
            "  - search.get_images_from_caption_query_mmbert: Image search with captions\n"
            "  - search.get_segments_from_event_query_mmbert: Segment search with events\n"
            "  - search.get_audio_from_query_dense: Audio search with spoken content\n\n"
            "Args:\n"
            "  raw_query (str): The user's original query. Describes events, actions, scenes, "
            "or semantic intent (REQUIRED)\n"
            "  variants (list[str]): Perspective variations to generate. Each variant should "
            "describe a different semantic angle or narrative perspective (REQUIRED)"
        ),
        instructions=(
            "Use when: query describes events, actions, or scene-level meaning, "
            "query is ambiguous or short.\n\n"
            "Best paired with: search.get_images_from_caption_query_mmbert, search.get_segments_from_event_query_mmbert (use enhanced queries for search). "
            "NOT for pure visual appearance queries - use enhance_visual_query instead."
        ),
    )
    async def enhance_textual_query(
        self,
        raw_query: str,
        variants: list[str],
    ) -> ToolResult:
        """Generate rich semantic variations for event/scene queries.

        Takes a query describing events, actions, or scene-level meaning
        and expands it into multiple detailed variations optimized for caption/segment
        embedding retrieval.

        Unlike enhance_visual_query (which focuses on visual appearance), this tool
        emphasizes semantic context and narrative structure.

        Args:
            raw_query: The user's original query.
                      Describes events, actions, scenes, or semantic intent.
            variants: Perspective variations to generate. Each variant should
                     describe a different semantic angle or narrative perspective.

        Returns:
            ToolResult with enhanced query variations (one per line)
        """
        variants_str = "\n".join(f"- {v}" for v in variants)

        user_prompt = f"""
        Original query: {raw_query}

            Perspective variants to generate:
            {variants_str}

            Output: A list of optimized descriptions for semantic retrieval.
            Each description on a new line.
        """
        try:
            content = await self._invoke_llm(system_prompt=CAPTION_ENHANCEMENT_SYSTEM_PROMPT, prompt=user_prompt)

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
