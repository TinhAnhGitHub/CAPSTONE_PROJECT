from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from videodeepsearch.api.stream import router as workflow_router
from videodeepsearch.core.lifespan import lifespan


app = FastAPI(
    title="Video Deep Search Agent",
    description="Streaming workflow API for the index-then-act video understanding pipeline.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(workflow_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "video-agent-workflow"}
