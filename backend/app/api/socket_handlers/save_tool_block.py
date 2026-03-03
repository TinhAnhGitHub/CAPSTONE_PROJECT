from app.model.session_message import (
    ToolsBlock,
)

def save_tool_block(accum):
    tool_message_block = ToolsBlock(steps=accum.tools_accum)
    accum.ai_message_blocks.append(tool_message_block)
    accum.tools_accum = []
    # global_session_tasks[session_id]["accum"] = accum