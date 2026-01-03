"""
videodeepsearch/tools/implementation/view/orc_tool.py
Allow the orchestrator agent to view/inspect its past/current context
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

from videodeepsearch.agent.definition import ORCHESTRATOR_AGENT


# @tool_registry.register(
#     group_doc_name=GroupName.VIEW_RESULT,
#     bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
#     bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
#     output_middleware=None,
#     input_middleware=None,
#     belong_to_agents=[ORCHESTRATOR_AGENT],
#     ignore_params=['session_id']
# )
# async def orc_view_work_histories(
#     ctx: Context,
#     session_id: str,
#     slicing: SLICING_ANNOTATION
# ) -> str :
#     """
#     View synthesized summaries from past rounds/sessions (orchestrator only).
#     - User refers to previous queries or findings
#     - Need context from earlier investigation rounds
#     - Building on past analysis for follow-up queries
#     - Checking what was concluded in prior sessions
#     - slicing: Python list indexing to select summaries
#       * -1: Current summary.
#       * -2: Most recent summary
#       * slice(-3, None): Last 2 summaries
#       * slice(0, 5): First 5 summaries
#     """

#     orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
#     parsed_slicing = parse_slicing(slicing)
    
#     if isinstance(parsed_slicing, list):
#         sliced_data = [orc_ctx.summarize_works[i] for i in parsed_slicing if i < len(orc_ctx.summarize_works)]
#     else:
#         sliced_data = orc_ctx.summarize_works[parsed_slicing]

#     try:
#         return sliced_data if isinstance(sliced_data, str) else '\n\n'.join(sliced_data)
#     except Exception as e:
#         raise ValueError(f"Maybe you over indexing, analyze the error: {str(e)}")
    
@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT],
    ignore_params=['session_id']
)
async def orc_view_current_evidences_by_worker(
    ctx: Context,
    session_id: str,
    agent_worker_name: str,
):
    """
    View evidence submitted by a specific worker in current round.
    - Worker has completed task
    - Need to review specific worker's findings before synthesis
    - Checking if worker successfully completed assigned objectiv
    """
    try:
        orc_context = await ctx.store.get(session_id)
        orc_context: OrchestratorContext = OrchestratorContext.model_validate(orc_context)
        return orc_context.get_agent_evidence(agent_worker_name=agent_worker_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to view evidence for worker '{agent_worker_name}' in session '{session_id}': {str(exc)}"
        ) from exc


@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT],
    ignore_params=['session_id']
)
async def orc_view_past_evidences_by_worker(
    ctx: Context,
    session_id: str,
    agent_worker_name: Annotated[str | None, "If agent name is specified, only the evidences produced by that agent are returned. If None, then all evidences will return. Be wisely in your choice."],
    slicing: SLICING_ANNOTATION
) -> str:
    """
    View evidence submitted by a specific worker in the history.
    - When you want to check for evidences in the past, that might be helpful for your planning and orchestration. Maybe the evidences in the past might be helpful 
    - Need to review specific worker's findings before synthesis. 
    - Can be optionally used, but might come in handy.
    """

    orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
    parsed_slicing = parse_slicing(slicing)
    
    if isinstance(parsed_slicing, list):
        sliced_data = [orc_ctx.history_worker_results[i] for i in parsed_slicing if i < len(orc_ctx.history_worker_results)]
    else:
        sliced_data = orc_ctx.history_worker_results[parsed_slicing]

    result = []
    if agent_worker_name:
        for session_evidence in sliced_data:
            for worker_result in session_evidence:
                if worker_result.worker_name == agent_worker_name: #type:ignore
                    result.extend(worker_result.evidences) #type:ignore
    
    return '\n'.join([str(res) for res in result])



@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT],
    ignore_params=['session_id']
)
async def orc_view_all_agent_in_the_past(
    ctx: Context,
    session_id: str,
) -> str:
    """
    This tool allow the orchestrator to view all the worker agents exists in the whole process (from the past to the present)
    - Use when you want to make sure the worker agent that you have orchestrated until now.
    - Allow you to retriefve the right name, in order to look at the agent's evidences (if they have)
    """
    orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
    return orc_ctx.get_all_worker_agent_name()
    


@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT]
)
async def orc_view_video_context(
    ctx: Context,
    session_id: str,
    video_id: str
) -> str:
    """
    Allow you to view the video context, given an video id.
    This tool is useful when the user asks follow-up questions that *appear* to be
    related to a video you have recently processed. If the given `video_id` has been
    handled befor the tool will return any stored context associated with that video. 
    
    Typical uses:
    - Checking whether a follow-up question refers to a previously processed video.
    - Reusing extracted context (metadata, summaries, OCR results, etc.) to avoid
      redundant processing.
    - Improving consistency in multi-turn conversations involving multiple videos.
    """
    orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
    is_video_context = orc_ctx.video_context.get(video_id)
    if is_video_context is None:
        return f"Video id: {video_id} does not existed in the video context yet. You might want to use other tools to get the available video that have context"
    return is_video_context.generate_context()


@tool_registry.register(
    group_doc_name=GroupName.VIEW_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_WORK_INSPECTOR,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT],
    
)
async def orc_view_get_all_video_ids_context(
    ctx: Context,
    session_id: str,
) -> str:
    """
    Return all video_ids that the context have acummulated to this present. Use it if you want to know that are the available video_ids
    """
    orc_ctx = OrchestratorContext.model_validate(await ctx.store.get(session_id))
    return f"Here are the video_id(s): {list(orc_ctx.video_context.keys())}"
