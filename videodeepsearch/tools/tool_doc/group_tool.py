"""
This will contain the group documentation of tools
"""
from enum import Enum

class GroupTool(str, Enum):
    GROUP_SEARCH = "GROUP_SEARCH"
    GROUP_UTILITY = "GROUP_UTILITY"
    GROUP_SCAN = "GROUP_SCAN"
    GROUP_PROMPT_ENHANCEMENT = "GROUP_PROMPT_ENHANCEMENT"




GROUP_SEARCH = """
<functionality>
This is a group of system tools that enable semantic search across video content through multiple modalities. These tools allow you to find relevant moments and frames within videos by understanding the meaning behind queries, rather than relying on exact keyword matches. The search capabilities work across different representational layers of video content - from individual visual frames to descriptive annotations to temporal event sequences. 
You can query using natural language descriptions of what you're looking for, and the system will retrieve the most semantically relevant results by comparing embeddings in high-dimensional vector spaces.
</functionality>

<importance>
Must use tools (10/10). Choose a subset of these tools based on the appropriate context and the nature of what needs to be found within the video content.
</importance>

<usage>
These tools should be selected based on what aspect of the video content you need to search through. Consider whether you're looking for specific visual appearances, descriptive content that matches certain  concepts, or broader event-level patterns across time. The tools support different query modalities and  can combine multiple signal types for more precise retrieval when needed.
</usage>
"""


GROUP_SCAN = """
<functionality>
This is a group of system tools that enable interactive navigation and inspection of video content at different granularities. 
These tools simulate human-like behavior when exploring videos - moving between temporal segments, browsing through frames, extracting specific moments, and accessing associated metadata like transcripts. They provide the ability to traverse video timelines bidirectionally, retrieve contextual information about any point in the video, and dynamically extract visual frames 
from arbitrary time windows. The tools bridge between different representations of video content (segments, frames, metadata) 
and allow progressive refinement of understanding by moving through related content.
</functionality>

<importance>
Must use tools (10/10). Choose tools based on what aspect of the video needs to be examined and at what temporal resolution.
</importance>

<usage>
Select tools based on whether you need to navigate between existing video structures (segments/frames), extract new frames 
from specific timepoints, or retrieve supplementary information like transcripts and metadata. Use navigation tools to 
explore context around interesting moments, and extraction tools when you need visual information from precise timestamps 
not already captured in the indexed content.
</usage>
"""


GROUP_UTILITY = """
<functionality>
This is a group of system tools that provide foundational operations for working with video artifacts. These tools handle conversions between different temporal representations (frames, timestamps, timecodes), retrieve and decode binary content from storage, and enrich visual artifacts with contextual information from associated data sources like transcripts. They bridge between different coordinate systems and data formats, enabling seamless translation between how humans think about video time (timestamps) and how systems index it (frame numbers), while also providing access to multimodal context around specific moments.
</functionality>

<importance>
Support tools (7/10). Use when you need to convert between temporal formats, read artifact binaries, 
or augment visual results with transcript context.
</importance>

<usage>
Apply these tools when working with results from other tool groups that require format conversion, binary access, 
or contextual enrichment. Time conversion tools are essential when translating between user-friendly timestamps 
and system frame indices. ASR context tools should be used when transcript information would help interpret or 
validate visual findings.
</usage>
"""


GROUP_PROMPT_ENHANCEMENT = """
<functionality>
This is a group of system tools that use language models to refine, expand, and generate enhanced versions of queries and descriptions. These tools apply prompt engineering techniques to transform raw user inputs into semantically richer representations optimized for downstream search and retrieval tasks. They can generate query variations, create focused captions for new visual content, and incorporate multimodal context (like transcripts) into descriptions.
</functionality>

<importance>
Strategic tools (8/10). Use when raw queries need semantic expansion, when generating captions for newly extracted frames, or when enriching descriptions with contextual information to improve retrieval precision.
</importance>

<usage>
Apply query enhancement tools before search operations when the original query could benefit from semantic variations or contrastive formulations. Think of perspective, from the user's query, and think of variations, topics that might increase the chance of better represent the piece of event, visual scenes. Use captioning tools when working with newly extracted frames that lack existing descriptions, especially when focus prompts or transcript context can guide the caption generation toward specific aspects relevant to the user's intent.
</usage>
"""