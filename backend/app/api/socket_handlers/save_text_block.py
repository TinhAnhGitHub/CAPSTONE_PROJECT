from app.model.session_message import (
    TextBlock,
)


def save_text_block(session_id, accum, global_session_tasks):
    ai_message_block = TextBlock(text=accum.accum)
    accum.ai_message_blocks.append(ai_message_block)
    global_session_tasks[session_id]["accum_blocks"].append(
        ai_message_block
    )
    accum.accum = ""
    global_session_tasks[session_id]["accum"] = ""
