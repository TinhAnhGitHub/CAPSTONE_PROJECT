"""
This will contain a bunch of agent as tools
"""
from typing import Annotated, cast
from llama_index.core.llms import LLM
from pydantic import BaseModel, Field
from .prompt import CONTRASTIVE_VISUAL_ENHANCEMENT_PROMPT, CAPTION__ENHANCEMENT_PROMPT
from ..registry import tool_registry


class EVQResponse(BaseModel):
    """Enhance visual query response"""
    resp: list[str] = Field(..., description="The list of enhanced response")


@tool_registry.register(
    category='Prompt Enhance',
    tags=["enhance", "visual"],
    dependencies=[
        "llm_as_tools",
    ]
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
    category='Prompt Enhance',
    tags=["enhance", "caption"],
    dependencies=[
        "llm_as_tools",
    ]
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





# async def rerank_window_artifacts_by_llm(
#     artifacts: Annotated[List[Union[ImageObjectInterface, SegmentObjectInterface]], "Artifacts from window fetch."],
#     relevance_query: Annotated[str, "Query to rerank by (e.g., 'action scene')."],
#     top_k: Annotated[int, "Keep top N after rerank.", default=5],
#     external_client: Annotated[ExternalEncodeClient, "For query embedding."],
# ) -> List[Union[ImageObjectInterface, SegmentObjectInterface]]:
#     """
#     Rerank artifacts by cosine sim to query embedding (textual/caption focus). Rerank by LLM
#     """
   

   