from enum import Enum
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.llms import MessageRole


class GroupName(str, Enum):
    SEARCH_GROUP = 'SEARCH_GROUP'
    CONTEXT_RETRIEVE_GROUP = 'CONTEXT_RETRIEVE_GROUP'
    VIEW_RESULT = "VIEW_RESULT"
    PERSIST_RESULT = "PERSIST_RESULT"
    UTILITY = "UTILITY"



GROUPNAME_TO_TEMPLATE = {
    'SEARCH_GROUP': RichPromptTemplate(
        """
        {% chat role="system" %}
        **Group**: SEARCH_GROUP
        **Description**: Tools in this group perform semantic search across video-derived artifacts. They interpret visual or text-based queries to retrieve the most relevant items, such as images, keyframes, or video segments based on visual or semantic similarity.
        **Usage**: Use these tools for any semantic search. They should be the priority tools for searching. All search tools return DataHandles that must be inspected with VIEW_RESULT tools.
        **Tools**:
        {{tool_usage}}
        {% endchat %} 
        """
    ),
    
    'CONTEXT_RETRIEVE_GROUP': RichPromptTemplate(
        """
        {% chat role="system" %}
        **Group**: CONTEXT_RETRIEVE_GROUP
        **Description**: Tools for retrieving audio context (ASR transcripts) around video segments or images. These tools fetch spoken words within a time window to verify or enrich visual findings with what was being said at that moment.
        **Usage**: Use after finding promising visual results to ground them in audio context. Essential for queries involving dialogue, speech, or audio events. Helps distinguish between similar-looking scenes via spoken content.
        **When to use**:
        - User query mentions dialogue, speech, or audio events
        - Need to verify visual matches against spoken content
        - Want to distinguish similar scenes using audio cues
        **Tools**:
        {{tool_usage}}
        {% endchat %} 
        """
    ),
    
    'VIEW_RESULT': RichPromptTemplate(
        """
        {% chat role="system" %}
        **Group**: VIEW_RESULT
        **Description**: Tools for inspecting, analyzing, and querying results from previous tool calls. These tools work with DataHandles to view detailed results, statistics, accumulated evidence, and cross-worker findings.
        **Usage**: Use immediately after receiving DataHandles from search/navigation tools. Essential for verifying result quality before persisting evidence. Workers use these to inspect their own results and query others' evidence. Orchestrators use these to monitor worker progress and review submissions.

        **Note**: This tools can be used to inspect the previous conversation. For example, if the chat is multi-turn, and the user ask about something related to the previous sessssion, then some of the tool can be use as well.

 
        **Tools**:
        {{tool_usage}}
        {% endchat %} 
        """
    ),
    
    'PERSIST_RESULT': RichPromptTemplate(
        """
        {% chat role="system" %}
        **Group**: PERSIST_RESULT
        **Description**: Tools for persisting evidence, updating shared context, and marking task completion. Workers use these to save high-confidence findings and submit final reports. Orchestrators use these to update video-level context and write synthesis reports.
        **Usage**: Workers persist evidence incrementally as they find supporting items, then call worker_mark_evidence to finish. Orchestrators update video context during synthesis, then call orc_synthesize_final_findings to deliver final report.
        **Critical notes**:
        - worker_mark_evidence is TERMINAL - it ends worker execution
        - orc_synthesize_final_findings is the orchestrator's final tool call
        - Only persist evidence with confidence ≥7/10
        **Tools**:
        {{tool_usage}}
        {% endchat %} 
        """
    ),
    
    'UTILITY': RichPromptTemplate(
        """
        {% chat role="system" %}
        **Group**: UTILITY
        **Description**: Utility tools for query enhancement, video navigation, and frame extraction. Includes tools to expand queries for better retrieval, navigate through video timelines (hop between segments/frames), and extract raw frame images.
        **Usage**: 
        - Query enhancement: Use before search to improve recall (enhance_visual_query, enhance_textual_query)
        - Video navigation: Use after finding matches to explore surrounding context (get_segments, get_image)
        - Frame extraction: Use when you need to visually inspect specific time windows
        **Typical patterns**:
        - Enhance query → search → inspect → navigate adjacent content
        - Find segment → hop backward/forward → verify continuity
        - Extract frames → view directly for detailed analysis
        **Tools**:
        {{tool_usage}}
        {% endchat %} 
        """
    )
}