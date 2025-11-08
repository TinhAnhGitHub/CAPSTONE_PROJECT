# app/main.py
import socketio


sio = socketio.AsyncServer(
    async_mode="asgi", cors_allowed_origins="*", logger=True, engineio_logger=True
)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from app.core.lifespan import lifespan
from app.core.config import settings
from app.api import chat
from app.api import user
from app.api import ingestion

from app.api.socket import sio


app = FastAPI(
    title="Chat Agent API",
    description="FastAPI application with chat agents and real-time messaging",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app_with_sockets = socketio.ASGIApp(sio, other_asgi_app=app)


os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.include_router(chat.router)
app.include_router(user.router)
app.include_router(ingestion.router)


@app.get("/")
async def root():
    return {
        "message": "Chat Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/socket.io/",
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "llm_model": "mock"}


if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app_with_sockets", host=settings.HOST, port=settings.PORT, reload=True
        )
    except KeyboardInterrupt as e:
        ...
