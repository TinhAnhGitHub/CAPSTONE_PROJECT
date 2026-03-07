from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from video_pipeline.api.lifespan import lifespan
from video_pipeline.api.routers.health import router as health_router
from video_pipeline.api.routers.upload import router as upload_router

app = FastAPI(
    title="Video Pipeline API",
    description=(
        "Submit videos to the Prefect processing pipeline and monitor deployment health. "
        "All processing happens asynchronously via the Prefect worker pool."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="")
app.include_router(health_router, prefix="")


@app.get("/")
async def root() -> dict:
    return {
        "service": "Video Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/uploads",
            "health": "/api/health",
            "prefect_health": "/api/health/prefect",
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):  # noqa: ANN001
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


def main() -> None:
    """Entry point registered in pyproject.toml [project.scripts]."""
    uvicorn.run(
        "video_pipeline.api.app:app",
        host="0.0.0.0",
        port=8050,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
