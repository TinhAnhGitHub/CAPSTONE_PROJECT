
import io, json
from typing import List
from minio import Minio
from llama_index.core.llms import ChatMessage

MINIO_ENDPOINT = "localhost:9000"
MINIO_BUCKET = "chat-history"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"

client = Minio(MINIO_ENDPOINT, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

def get_chat_history(user_id:str, session_id: str) -> List[ChatMessage]:
    """load chat history from minio"""
    return []
    try:
        res = client.get_object(MINIO_BUCKET, f"{user_id}_{session_id}.json")
        data = res.read().decode("utf-8")
        j = json.loads(data)
        return [ChatMessage(**m) for m in j]
    except Exception:
        return []

def save_chat_history(session_id: str, chat_history: List[ChatMessage]):
    """save chat history to minio"""
    return []
    try:
        d = json.dumps([m.dict() for m in chat_history]).encode("utf-8")
        client.put_object(
            MINIO_BUCKET,
            f"{session_id}.json",
            io.BytesIO(d),
            len(d),
            content_type="application/json"
        )
    except Exception:
        pass
