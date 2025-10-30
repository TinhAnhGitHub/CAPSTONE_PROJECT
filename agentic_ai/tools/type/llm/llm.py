"""
This will contain a bunch of agent as tools
"""
from typing import Annotated, cast
from llama_index.core.llms import LLM
from llama_index.core.llms import ChatMessage
from pydantic import BaseModel, Field
from .prompt import CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT, CAPTION__ENHANCEMENT_PROMPT, CAPTION_WITH_ASR_FOCUS_PROMPT
from ..registry import tool_registry
from ..helper import extract_s3_minio_url
from agentic_ai.tools.schema.artifact import ImageObjectInterface
from agentic_ai.tools.clients.minio.client import StorageClient
from llama_index.core.base.llms.types import ImageBlock, TextBlock
from ..util import get_related_asr_from_image

class EVQResponse(BaseModel):
    """Enhance visual query response"""
    resp: list[str] = Field(..., description="The list of enhanced response")


@tool_registry.register(
    category="Prompt/Enhancement",
    tags=["enhancement", "visual", "query", "contrastive"],
    dependencies=["llm_as_tools"]
)
async def enhance_visual_query(
    raw_query: Annotated[str, "Original user query (visual intent)"],
    variants: Annotated[list[str], "List of desired visual variations"],
    llm_as_tools: LLM,
) -> list[str]:
    """Enhance a visual query using contrastive prompt engineering."""
    prompt = CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT.format(raw_query=raw_query, variants=variants)
    sllm = llm_as_tools.as_structured_llm(EVQResponse)
    response = await sllm.acomplete(prompt)
    return cast(EVQResponse, response.raw).resp


@tool_registry.register(
    category="Prompt/Enhancement",
    tags=["enhancement", "caption", "semantic", "query"],
    dependencies=["llm_as_tools"]
)
async def enhance_textual_query(
    raw_query: Annotated[str, "Original user query (textual or event intent)"],
    variants: Annotated[list[str], "List of semantic or contextual variations"],
    llm_as_tools: LLM,
) -> list[str]:
    """Enhance a textual query for richer embedding or captioning."""
    prompt = CAPTION__ENHANCEMENT_PROMPT.format(raw_query=raw_query, variants=variants)
    sllm = llm_as_tools.as_structured_llm(EVQResponse)
    response = await sllm.acomplete(prompt)
    return cast(EVQResponse, response.raw).resp

   


@tool_registry.register(
    category="Caption/New Image",
    tags=["caption", "image", "multimodal", "asr", "focus", "retrieval"],
    dependencies=["llm_as_tools", "minio_client"]
)

async def caption_new_image(
    raw_inference_image: ImageObjectInterface,
    related_asr:str | None,
    focus_prompt: str,
    llm_as_tools: LLM,
    minio_client: StorageClient
)->str :
    minio_path = raw_inference_image.minio_path
    bucket, object_name = extract_s3_minio_url(minio_path)
    image_bytes = minio_client.get_object(bucket=bucket, object_name=object_name)

    image_block = ImageBlock(image=image_bytes)
    full_prompt = CAPTION_WITH_ASR_FOCUS_PROMPT.format(
        related_asr=related_asr,
        focus_prompt=focus_prompt
    )
    text_block = TextBlock(text=full_prompt)

    chat_message = ChatMessage(
        content=[image_block, text_block]
    )
    response = await llm_as_tools.achat([chat_message])
    return response.message.content if response.message.content else "No caption due to error, please try again"

