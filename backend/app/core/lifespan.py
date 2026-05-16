from contextlib import asynccontextmanager
from fastapi import FastAPI
from beanie import init_beanie
from minio import Minio
import threading

from motor.motor_asyncio import AsyncIOMotorClient
from llama_index.core.llms import LLM
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import MockLLM
import socketio

from app.core.config import settings
from app.model.chat_history import ChatHistory
from app.service.agent import Agent
from app.service.chat import ChatService
from app.model.user import User
from app.service.user import UserService
from app.service.minio import MinioService
from app.model.group import Group
from app.model.session_video import SessionVideo
from app.model.video import Video
from app.model.session_message import SessionMessage
from app.worker.celery_app import celery_app


class AppState:
    """Global application state"""

    def __init__(self):
        self.mongo_client: AsyncIOMotorClient = None  # type: ignore

        self.agent: Agent = None  # type: ignore
        self.chat_service: ChatService = None  # type: ignore
        self.user_service: UserService = None  # type: ignore
        self.minio_service: MinioService = None  # type: ignore
        self.sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins="*",
            logger=True,
            engineio_logger=True,
        )


app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up application...")
    app_state.mongo_client = AsyncIOMotorClient(settings.MONGO_URI)
    database = app_state.mongo_client[settings.MONGO_DB]
    await init_beanie(database=database, document_models=[ChatHistory, User, Group, Video, SessionVideo, SessionMessage])  # type: ignore
    print("✓ MongoDB and Beanie initialized")

    try:
        minio_client = Minio(
            endpoint=settings.MINIO_PUBLIC_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )

        minio_service = MinioService(minio_client)

        print("✓ MinIO initialized")
    except Exception as e:
        print(f"✗ MinIO initialization failed: {e}")

    llm = MockLLM(max_tokens=2)
    app_state.agent = Agent(llm=llm)
    app_state.chat_service = ChatService()
    app_state.minio_service = minio_service
    app_state.user_service = UserService(minio_service, app_state.sio)
    print("✓ Services initialized")

    # ── Embedded Celery worker ────────────────────────────────────────────────
    # Runs in a background daemon thread so you don't need a separate terminal.
    # daemon=True means it shuts down automatically when FastAPI exits.
    # pool=threads avoids spawning child processes inside uvicorn's process.
    worker = celery_app.Worker(
        loglevel="info",
        concurrency=4,
        pool="threads",       # thread pool — safe to embed inside uvicorn
        without_gossip=True,  # reduce Redis chatter
        without_mingle=True,  # skip worker sync on startup (faster boot)
        without_heartbeat=False,
    )
    worker_thread = threading.Thread(target=worker.start, daemon=True, name="celery-worker")
    worker_thread.start()
    print("✓ Celery worker started (embedded, thread pool, concurrency=4)")
    # ─────────────────────────────────────────────────────────────────────────

    yield

    print("Shutting down application...")
    worker.stop()
    print("✓ Celery worker stopped")
    if app_state.mongo_client:
        app_state.mongo_client.close()
    print("✓ Application shutdown complete")
