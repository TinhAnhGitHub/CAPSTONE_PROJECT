from typing import Sequence

from beanie import PydanticObjectId

from llama_index.core.base.llms.types import MessageRole
from datetime import datetime

from uuid import uuid4

from app.model.chat_history import ChatHistory
from app.model.session_message import SessionMessage
from llama_index.core.base.llms.types import MessageRole, ContentBlock


class ChatService:
    async def add_message(
        self, session_id: str | None, user_id: str | None, message: SessionMessage
    ):

        if session_id is None:
            session_id = str(PydanticObjectId())

        message.session_id = PydanticObjectId(session_id)

        chat_history = await ChatHistory.get(PydanticObjectId(session_id))
        if chat_history:
            await message.insert()
            chat_history.last_updated = datetime.now()
            await chat_history.save()
        else:
            new_chat_his = ChatHistory(
                id=PydanticObjectId(session_id), user_id=PydanticObjectId(user_id),
            )
            await new_chat_his.insert()
            await message.insert()

        return session_id

    async def get_chat_history(self, session_id: str) -> ChatHistory | None:
        return await ChatHistory.get(PydanticObjectId(session_id))

    async def get_all_session_ids(self) -> list[ChatHistory]:
        return await ChatHistory.find_all().to_list()

    async def delete_chat_history(self, session_id: str) -> bool:
        chat_history = await ChatHistory.get(PydanticObjectId(session_id))
        if chat_history:
            await chat_history.delete()
            return True
        return False

    @staticmethod
    def blocks_to_message(role: MessageRole, blocks: Sequence[ContentBlock], **kwargs):
        return SessionMessage(role=role, blocks=list(blocks), additional_kwargs=kwargs)
