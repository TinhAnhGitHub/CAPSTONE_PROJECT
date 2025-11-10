from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
from videodeepsearch.api.stream import router as workflow_router
from videodeepsearch.api.health import router as health_router
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
app.include_router(health_router)


@app.get("/")
async def root():
    return {"status": "ok", "service": "video-agent-workflow"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8050,
        reload=True,
        log_level="debug",
    )
