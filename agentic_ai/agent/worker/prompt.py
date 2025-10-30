from typing import  Annotated
from llama_index.core.prompts import PromptTemplate



MAKE_DECISION_PROMPT: Annotated[
    PromptTemplate,
    "This prompt template will help the agent choosing to use the function directly, or to spawn a code which can execute more complex task"
] = PromptTemplate(
    """
    You must decide the best approach to solve the user's request.
    
    **OPTION 1: tools** - Use when:
    - Single operation (search, retrieve, convert)
    - Direct function call works (e.g., "find images of cars", "get video metadata")
    - No intermediate processing needed
    - Simple data retrieval or lookup

    **OPTION 2: code** - Use when:
    - Multiple sequential operations needed
    - Requires loops, conditions, or aggregation
    - Need to process/filter/transform results
    - Combining outputs from multiple tools
    - Complex logic like "find X, then for each result check Y, filter by Z"

    **Available tools:**
    {tool_descriptions}
    """
)