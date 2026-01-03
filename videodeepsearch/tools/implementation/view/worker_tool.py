"""
# videodeepsearch/tools/implementation/view/tool.py
This contains a set of tools for the agents to view the results,...
"""
from typing import Annotated, Literal
from pydantic import Field
from llama_index.core.workflow import Context

from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.middleware.data_handle import DataHandle
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.doc_template.bundle_template import (
    VIDEO_EVIDENCE_WORKER_BUNDLE , VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE
)

from videodeepsearch.tools.base.types import BundleRoles
from videodeepsearch.tools.base.middleware.arg_doc import HANDLE_ID_ANNOTATION
from videodeepsearch.agent.context.worker_context import SmallWorkerContext
from videodeepsearch.agent.context.orc_context import OrchestratorContext

from .arg_doc import (
    SLICING_ANNOTATION,
    parse_slicing
)

from videodeepsearch.agent.definition import WORKER_AGENT


async def retrieve_tool_persist_result(
    ctx: Context,
    agent_name_key: str,
    handle_id: str
) -> DataHandle:
    ctx_dict = await ctx.store.get(agent_name_key)
    if ctx_dict is None:
        raise ValueError(f"No state found for agent: {agent_name_key}")

    local_agent_context = SmallWorkerContext.model_validate(ctx_dict)
    result_store = local_agent_context.raw_result_store
    if result_store is None:
        raise ValueError(f"No result store found for agent: {agent_name_key}" )

    data_handle = result_store.retrieve(handle_id=handle_id)
    if data_handle is None:
        raise ValueError(f"Handle tool id {handle_id} not found in ResultStore of agent: {agent_name_key}")

    return data_handle


@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.WORKER_RESULT_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['agent_name']
)
async def worker_view_all_data_handle(
    ctx: Context, 
    agent_name_key: str,   
) -> str:
    """
    Inspect detailed results from all tool calls.

    **When to use:**
    - If you somehow forgot all the result stored. just use this.
    - Receiving all data handle id that tool's output

    """
    ctx_dict = await ctx.store.get(agent_name_key)
    if ctx_dict is None:
        raise ValueError(f"No state found for agent: {agent_name_key}")

    local_agent_context = SmallWorkerContext.model_validate(ctx_dict)
    result_store = local_agent_context.raw_result_store
    if result_store is None:
        raise ValueError(f"No result store found for agent: {agent_name_key}" )

    return result_store.get_all_handle_str()
    

@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.WORKER_RESULT_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['agent_name']
)
async def worker_view_results(
    ctx: Context, # partial params, passed during runtime, ignore
    agent_name_key: str, # partial params, passed during runtime, ignore
    handle_id: HANDLE_ID_ANNOTATION,
    slicing: SLICING_ANNOTATION | None = None,
) -> str:
    """
    Inspect detailed results from a previous search tool call.

    Retrieves the full data stored under a DataHandle and formats it for review. 
    Results are sorted by score (descending) if they come from search tools, showing detailed representations 
    of each item including timestamps, video IDs, scores, captions, and storage paths.
    If they from  video navigator tools, then they are sorted based on timestamp. PLease review the video navigator doc.

    **When to use:**
    - After receiving a DataHandle from any tool.
    - Need to verify quality of search results before persisting evidence
    - Want to inspect specific items by index or range
    - Checking if results actually match the query intent

    **When NOT to use:**
    - To get summary statistics (use worker_view_statistics instead)
    - To access results from other workers (use worker_request_related_evidence)
    - Before calling a search tool (nothing to view yet)
    - When you unsure that whether or not you use any tool yet?

    **Typical workflow:**
    1. Call a tool that return DataHandle (get_images_from_visual_query, etc.)
    2. Receive DataHandle in response
    3. Use this tool with the handle_id to inspect results
    4. Optionally use slicing to view specific items
    5. Decide whether to persist as evidence
    

    
    """
    data_handle = await retrieve_tool_persist_result(ctx, agent_name_key=agent_name_key, handle_id=handle_id)
    raw_data = data_handle.get_data()
    tool_call = data_handle.tool_used
    tool_name = tool_call.tool_name #type:ignore
    tool_kwargs = tool_call.tool_kwargs #type:ignore

    if len(raw_data) == 0:
        return "No results to display"
    
    if slicing is not None:
        parsed_slicing = parse_slicing(slicing)
        if isinstance(parsed_slicing, list):
            sliced_data = [raw_data[i] for i in parsed_slicing if i < len(raw_data)]
        else:
            sliced_data = raw_data[parsed_slicing]

    else:
        sliced_data = raw_data
    
    formatter_cls = type(
        raw_data[0]
    )

    return formatter_cls.quick_format(
        tool_name,
        tool_kwargs,
        handle_id,
        sliced_data #type:ignore
    )

@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.WORKER_RESULT_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['agent_name']
)
async def worker_view_my_evidence(
    ctx: Context,
    agent_name_key: str,
) -> str:
    """
    View all evidence you've persisted so far in this session.  Useful when you want to look back the evidences you have gathered.

    **When to use:**
    - You want to use this tool before you finish your work
    - You want to see what evidences you have peristed

    
    """
    ctx_dict = await ctx.store.get(agent_name_key)
    local_ctx = SmallWorkerContext.model_validate(ctx_dict)
    
    if not local_ctx.evidences:
        return "No evidence persisted yet."
    
    lines = [f"Your Evidence Summary ({len(local_ctx.evidences)} items)\n"]
    for i, ev in enumerate(local_ctx.evidences, 1):
        lines.append(f"{i}. \n")
        lines.append(str(ev))
 
    return "\n".join(lines)




@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.WORKER_RESULT_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['agent_name']
)
async def worker_view_statistics(
    ctx: Context,
    agent_name_key: str,
    handle_id: HANDLE_ID_ANNOTATION,
    group_by: Annotated[
        Literal['video_id', 'score_bucket'],
        Field(description='How to group these items for statistics')
    ] = 'video_id'
):
    """
    View aggregated statistics and distribution for search results.

    Analyzes search results to show distributions, counts, and statistical summaries 
    grouped by video, score ranges, or time buckets. Useful for understanding result 
    patterns before detailed inspection or evidence selection.

    **When to use:**
    - Want high-level overview before detailed inspection
    - Need to understand result distribution across videos
    - Checking score ranges to set confidence thresholds
    - Identifying temporal patterns in matches
    - Quick validation of search quality without viewing all items

    
    **Typical workflow:**
    1. Call search tool → receive DataHandle
    2. Call this tool for statistical overview
    3. Analyze distribution and decide inspection strategy
    4. Use to inspect specific ranges
    5. Persist evidence based on statistical insights
    """
    data_handle = await retrieve_tool_persist_result(ctx, agent_name_key, handle_id)
    raw_data = data_handle.get_data()
    tool_call = data_handle.tool_used
    tool_name = tool_call.tool_name #type:ignore
    tool_kwargs = tool_call.tool_kwargs #type:ignore

    
    if len(raw_data) == 0:
        return "No results to display"
    
    formatter_cls = type(
        raw_data[0]
    )

    return formatter_cls.statistic_format(
        tool_name = tool_name,
        tool_kwargs = tool_kwargs,
        handle_id=handle_id,
        items=raw_data,
        group_by=group_by,
    )





# @tool_registry.register(
#     group_doc_name=GroupName.VIEW_RESULT,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.WORKER_RESULT_INSPECTOR,
#     output_middleware=None,
#     input_middleware=None
# )
# async def worker_request_related_evidence(
#     ctx: Context,
#     session_id: str,
#     video_ids: list[str],
#     accept_min_score: Annotated[float, Field(description="filter min score", le=1.0, ge=0.1)]
# ) -> str:
#     """
#     Query evidence from other workers related to specific videos and confidence level.

#     Searches through evidence submitted by all workers in the current round to find 
#     items related to target videos with confidence scores above a threshold. Enables 
#     workers to build on others' findings and avoid duplicate investigation.

#     **When to use:**
#     - About to investigate videos that others may have already analyzed
#     - Want to see what high-confidence evidence exists for specific videos
#     - Building comprehensive evidence by combining multiple workers' findings
#     - Checking if your task overlaps with completed workers

#     **Typical workflow:**
#     1. Receive task mentioning specific videos
#     2. Call this tool to find existing high-confidence evidence for those videos
#     3. Review what others found to avoid duplicate work
#     4. Focus investigation on gaps or complementary aspects
#     5. Submit your evidence 

#     """
#     orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
#     relevance_evidences = []
#     for worker_result in orc_ctx.history_worker_results[-1]:
#         for evidence in worker_result.evidences:
#             if any(vid in evidence.related_video_ids for vid in video_ids) and evidence.confidence_score >= accept_min_score:
#                 relevance_evidences.append(str(evidence))
            
#     separator = "\n" + "=" * 50 + "\n"
#     return separator.join(relevance_evidences)
    


