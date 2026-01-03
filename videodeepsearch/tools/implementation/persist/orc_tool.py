"""
videodeepsearch/tools/implementation/persist/orc_tool.py
This contains a bunch of tools for the agent to modify the context ( orchestrator)
"""
from typing import Annotated
from llama_index.core.workflow import Context

from videodeepsearch.agent.context.orc_context import OrchestratorContext

from videodeepsearch.tools.base.registry import tool_registry
from videodeepsearch.tools.base.doc_template.group_doc import GroupName
from videodeepsearch.tools.base.doc_template.bundle_template import VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE
from videodeepsearch.tools.base.types import BundleRoles

from videodeepsearch.agent.definition import ORCHESTRATOR_AGENT


@tool_registry.register(
    group_doc_name=GroupName.PERSIST_RESULT,
    bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
    bundle_role_key=BundleRoles.ORCHESTRATOR_EVIDENCE_MANAGER,
    output_middleware=None,
    input_middleware=None,
    belong_to_agents=[ORCHESTRATOR_AGENT],
    ignore_params=['session_id']
)
async def update_video_context_orc_agent(
    ctx: Context,
    session_id: str,
    video_id: Annotated[str, "The video id. This should usually coming from the related_video_id in those tool results from the worker agents"],
    new_finding: Annotated[str, "Any new things from the video that you want to note down."]
) -> str:
    """
    Add new finding to shared video context (orchestrator only).
    
    Updates the shared video context with new insights discovered during synthesis.
    Allows orchestrator to maintain cumulative knowledge about each video across
    multiple rounds and workers.

    **When to use:**
    - Synthesizing worker findings and want to update video knowledge
    - Discovered new insight about a video from cross-worker analysis
    - Building cumulative video profiles across conversation rounds
    - Before writing final report.

    **Typical workflow (Orchestrator):**
    1. Review all worker submissions 
    2. Synthesize cross-worker findings
    3. Call this tool to update video context with new insights
    4. Repeat for each video with new findings
    5. Write final report.
    """
    try:
        async with  ctx.store.edit_state() as ctx_state:
            global_shared_context = OrchestratorContext.model_validate(ctx_state[session_id])
            global_shared_context.update_video_context(video_id, new_finding)
            ctx_state[session_id] = global_shared_context.model_dump(mode='json')    
        return f"Video id {video_id} context has been updated successfully."
    except Exception as e:
        return f"Error: {e}. Please do not use this tool anymore"
    
# @tool_registry.register(
#     group_doc_name=GroupName.PERSIST_RESULT,
#     bundle_spec=VIDEO_EVIDENCE_ORCHESTRATOR_BUNDLE,
#     bundle_role_key=BundleRoles.ORCHESTRATOR_EVIDENCE_MANAGER,
#     output_middleware=None,
#     input_middleware=None,
#     return_direct=True,
#     belong_to_agents=[ORCHESTRATOR_AGENT],
#     ignore_params=['session_id']
# )
# async def orc_synthesize_final_findings(
#     ctx: Context,
#     session_id: str,
#     user_demand: Annotated[str, "This is the user demand, you need to note it down by your own words"],
#     synthesize_result_summary: Annotated[str, "Synthesize the summary into a detail report. Note about the process of making plan, divide works, monitor results,.... and how was the process?"]
# ) -> str:
#     """
#     Write final comprehensive report synthesizing all worker findings (orchestrator only).
    
#     Creates the final deliverable report combining evidence from all workers,
#     cross-worker synthesis, process explanation, and conclusions. This is the
#     orchestrator's final output to the user. This is your final tool

#     **When to use:**
#     - All workers have completed and submitted evidence
#     - Reviewed all worker submissions
#     - Synthesized findings and updated video contexts
#     - Ready to deliver final answer to user
#     - THIS IS ORCHESTRATOR'S FINAL TOOL CALL
    
#     **When NOT to use:**
#     - Workers still running (wait for all to complete)
#     - Haven't reviewed worker evidence yet
    
#     **Typical workflow (Orchestrator):**
#     2. Wait for all workers to finish
#     3. Review each worker's submission
#     4. Update video contexts with synthesized insights
#     5. Call THIS TOOL with comprehensive report
#     """
#     async with ctx.store.edit_state() as ctx_state:
#         orc_ctx = OrchestratorContext.model_validate(ctx_state[session_id])

#         final = f"""
#         User demand: {user_demand}

#         Work summary: {synthesize_result_summary}
#         """
        
        
#         orc_ctx.summarize_works.append(final)
        
    
#         ctx_state[session_id] = orc_ctx.model_dump(mode='json')

    return final