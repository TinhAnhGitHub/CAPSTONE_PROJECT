from app.model.session_message import (
    ThinkingBlock,
)


def save_thinking_block( accum):
    thinking_message_block = ThinkingBlock(steps=accum.thinking_accum)
    accum.ai_message_blocks.append(thinking_message_block)
    accum.thinking_accum = []
