from .md import single_caption, single_detect, single_query
from llama_index.core.tools import FunctionTool, ToolMetadata


single_tools = [
    FunctionTool(
        fn=single_caption,
        metadata=ToolMetadata(
            name="single_caption",
            description="Get the caption of a single frame given its path or index"
        )
    ),
    FunctionTool(
        fn=single_detect,
        metadata=ToolMetadata(
            name="single_detect",
            description="Detect a specific object in a single frame given its path or index and object name"
        )
    ),
    FunctionTool(
        fn=single_query,
        metadata=ToolMetadata(
            name="single_query",
            description="Answer a question about a single frame given its path or index and a question string"
        )
    )
]

complex_tools = [
    
]
__all__ = single_tools + complex_tools



import random

def generate_random_number(low: float = 0, high: float = 1) -> float:
    """Generate a random number between 'low' and 'high'."""
    return random.uniform(low, high)

random_tool = [FunctionTool(
    fn=generate_random_number,
    metadata=ToolMetadata(
        name="generate_random_number",
        description="Generate a random number between 'low' and 'high' (inclusive)."
    ),
)]

def visual_content(tools: list[FunctionTool] = single_tools):
    ctx= []
    for tool in tools:
        tool_info = {
            "name": tool.metadata.name,
            "description": tool.metadata.description
        }
        if hasattr(tool, "fn") and hasattr(tool.fn, "__annotations__"):
            tool_info["parameters"] = {
                k: str(v.__name__) if hasattr(v, "__name__") else str(v)
                for k, v in tool.fn.__annotations__.items()
                if k != "return"
            }

        ctx.append(tool_info)
    return ctx

get_visual_tools = FunctionTool(fn = visual_content,
                                metadata =ToolMetadata(
                                    name="get_visual_tools",
                                    description="Get the list of tools that can be used to retrieve visual content from videos."
                                ))

get_context_tools = [get_visual_tools]
get_all_tools  = single_tools