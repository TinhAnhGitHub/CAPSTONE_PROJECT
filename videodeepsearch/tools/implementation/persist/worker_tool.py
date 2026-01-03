"""
videodeepsearch/tools/implementation/persist/worker_tool.py
This contains a bunch of tools for the agent to modify the context (local worker)
"""
from typing import Sequence, Annotated, cast
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCall
from pydantic import Field

from videodeepsearch.tools.base.middleware.arg_doc import HANDLE_ID_ANNOTATION
from videodeepsearch.agent.context.worker_context import SmallWorkerContext
from videodeepsearch.agent.context.orc_context import OrchestratorContext, WorkerResult
from videodeepsearch.agent.context.worker_context import EvidenceItem

from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.doc_template.bundle_template import (
    VIDEO_EVIDENCE_WORKER_BUNDLE 
)
from videodeepsearch.tools.base.types import BundleRoles


from ..view.arg_doc import SLICING_ANNOTATION, parse_slicing

from videodeepsearch.tools.base.schema import BaseInterface
from videodeepsearch.agent.definition import WORKER_AGENT


SUMMARY_STR = """
"A detailed final report from the worker agent to the orchestrator. 
The summary must explicitly address each of the following questions:

1. Task Understanding  
   - How did you interpret the task?  
   - What was the goal, requirement, or expected output?  
   - What constraints or important conditions did you identify?

2. Execution Process  
   - Which tools did you use, and why?  
   - What was the sequence of steps you performed?  
   - How did each step contribute to solving the task?

3. Evidence Handling  
   - What evidence did you generate, retrieve, or aggregate?  
   - Why is this evidence relevant or correct?  
   - How did you decide what to store in the shared context?

4. Outcome Evaluation  
   - Were you able to complete the task fully?  
   - If yes, what is the final result?  
   - If not, what blocked you or what information was missing?

5. Reflection & Recommendations  
   - What challenges did you encounter?  
   - What improvements or next steps do you recommend for the orchestrator?  
   - Any additional insights the orchestrator should be aware of?"
"""



@tool_registry.register(
    group_doc_name=GroupName.PERSIST_RESULT,
    bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
    bundle_role_key=BundleRoles.WORKER_EVIDENCE_MANAGER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[WORKER_AGENT],
    ignore_params=['agent_key_name']
)
async def worker_persist_evidence(
    ctx: Context,
    agent_name_key: str,
    handle_id: HANDLE_ID_ANNOTATION,
    confidence_score: Annotated[int, Field(description="""- confidence_score: Your confidence level (1-10 scale)
      * 9-10: Definitive match, no doubt
      * 7-8: Strong match with minor caveats
      * 5-6: Moderate match (rarely persist these)
      * < 5: Weak match (don't persist)""", ge=1, le=10)],
    claims: Annotated[str, """- claims: WHY these items are evidence (be specific and clear)
      * Explain what you see that matches the task
      * Reference specific details (timestamps, visual features, etc.)
      * Note any caveats or uncertainties"""],
    slicing: SLICING_ANNOTATION,
) -> str:
    
    """
    Mark specific search results as high-confidence evidence for your task.

    Selects items from a previous search result and saves them to your evidence 
    collection with a confidence score and explanatory claims. Evidence persists 
    across tool calls and will be included in your final report to the orchestrator.
    Use this to build your case incrementally as you find supporting items.

    **When to use:**
    - Found results that strongly support your task objective (confidence 7-10)
    - After inspecting results with to verify match quality
    - Ready to commit specific items as evidence 
    - Want to accumulate evidence from multiple searches before final submission
    - Items clearly match task requirements with good scores

    **When NOT to use:**
    - Results don't clearly match task objective (confidence < 7)
    - Haven't verified results with worker_view_results yet
    - As final step (use worker_mark_evidence to finish and submit task)
    - Results are ambiguous or require more investigation

    **Typical workflow:**
    1. Call search tool → receive DataHandle
    2. Call tool to inspect DataHandle → inspect matches in detail
    3. Identify high-confidence items.
    4. Call THIS TOOL to persist those items as evidence
    5. Repeat searches and evidence collection as neede

    """
    ctx_dict = await ctx.store.get(agent_name_key)
    if ctx_dict is None:
        raise ValueError(f"No state found for agent: {agent_name_key}")

    local_agent_context = SmallWorkerContext.model_validate(ctx_dict)
    result_store = local_agent_context.raw_result_store
    if result_store is None:
        raise ValueError(f"No result store found for agent: {agent_name_key}" )

    data_handle = result_store.retrieve(handle_id)
    if data_handle is None:
        raise ValueError(f"Handle id: {handle_id} not found in the result store of agent {agent_name_key}")


    results_data = data_handle.get_data()

    if len(results_data) == 0:
        raise ValueError("")
    
    if slicing is not None:
        parsed_slicing = parse_slicing(slicing)
        if isinstance(parsed_slicing, list):
            sliced_data = [results_data[i] for i in parsed_slicing if i < len(results_data)]
        else:
            sliced_data = results_data[parsed_slicing]

    else:
        sliced_data = results_data
    
    if not isinstance(sliced_data, Sequence):
        sliced_data = [sliced_data]

    related_video_ids = list(set([
        r.related_video_id for r in sliced_data
    ]))


    tool_call = ToolCall(
        tool_name=data_handle.tool_used.tool_name, #type:ignore
        tool_kwargs=data_handle.tool_used.tool_kwargs, #type:ignore
        tool_id=data_handle.tool_used.tool_id #type:ignore
    )
    evidence = EvidenceItem(
        source_worker_name=agent_name_key,
        source_tool_call=tool_call,
        artifacts=sliced_data,
        confidence_score=confidence_score,
        related_video_ids=related_video_ids,
        claims=claims
    )

    local_agent_context.evidences.append(evidence)

    async with ctx.store.edit_state() as ctx_state:
        ctx_state[agent_name_key] = local_agent_context

    return  (
        f"✓ Persisted {len(sliced_data)} items as evidence "
        f"(confidence: {confidence_score}/10, videos: {related_video_ids})"
    )

# @tool_registry.register(
#     group_doc_name=GroupName.PERSIST_RESULT,
#     bundle_spec=VIDEO_EVIDENCE_WORKER_BUNDLE,
#     bundle_role_key=BundleRoles.WORKER_EVIDENCE_MANAGER,
#     output_middleware=None,
#     input_middleware=None,
#     return_direct=True
# )
# async def worker_mark_evidence(
#     ctx: Context,
#     summary: Annotated[str, SUMMARY_STR], 
#     agent_name: str,
#     session_id: str
# ) -> str:
#     """
#     Submit all accumulated evidence and complete your task (FINAL STEP - TERMINAL).

#     Packages your entire work session—all persisted evidence, tool results, reasoning 
#     process, and conclusions—into a WorkerResult that's sent to the orchestrator. 
#     This is a TERMINAL action: after calling this, your work is done and control 
#     returns to the orchestrator. THIS MUST BE YOUR LAST TOOL CALL.

#      **When to use:**
#     - Completed your assigned task objective to the best of your ability
#     - Accumulated sufficient evidence, or you just can't finish the task even trying
#     - Ready to report findings to orchestrator with comprehensive summary
#     - No more searches, investigations, or evidence collection needed
#     - THIS MUST BE YOUR ABSOLUTE LAST TOOL CALL
#     -  Task is impossible/unclear (still call this, but explain why in summary)

#     **When NOT to use:**
#     - Still exploring or need more searches to complete task
#     -  Haven't persisted any evidence yet
#     -  Want to contribute findings but continue working
#     -  As an intermediate step (this ENDS your work)
#     """
#     ctx_dict = await ctx.store.get(agent_name)
#     if ctx_dict is None:
#         raise ValueError(f"No state found for agent: {agent_name}")

#     local_agent_context = SmallWorkerContext.model_validate(ctx_dict)
    
#     all_evidences = local_agent_context.evidences
#     task_objective = local_agent_context.task_objective
#     worker_chat_history = local_agent_context.chat_history
#     result_stored = local_agent_context.raw_result_store
#     worker_name = local_agent_context.worker_agent_name

#     worker_result = WorkerResult(
#         worker_name=worker_name,
#         task_objective=task_objective,
#         worker_chat_history=worker_chat_history,
#         raw_result_store=result_stored,
#         evidences=all_evidences,
#         result_summary=summary
#     )

#     async with  ctx.store.edit_state() as ctx_state:
#         global_shared_context = OrchestratorContext.model_validate(ctx_state[session_id])
#         global_shared_context.add_to_latest_worker_results(worker_result)
#         ctx_state[session_id] = global_shared_context.model_dump(mode='json')    
#     return summary





