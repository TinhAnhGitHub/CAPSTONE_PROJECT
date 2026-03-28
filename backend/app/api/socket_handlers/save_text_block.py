from app.model.session_message import (
    TextBlock,
)


def save_text_block(accum):
    ai_message_block = TextBlock(text=accum.text_accum)
    accum.ai_message_blocks.append(ai_message_block)
    accum.text_accum = ""
    # global_session_tasks[session_id]["accum"] = accum
