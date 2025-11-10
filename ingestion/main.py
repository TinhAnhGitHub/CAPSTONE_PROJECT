import os
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse 

from core.lifespan import lifespan
from fastapi.middleware.cors import CORSMiddleware
from core.settings import get_settings
from loguru import logger
from api.management import router as management_router
from api.upload import router as upload_router
from api.health import router as health_router

app = FastAPI(
    title="Video Processing Orchestration API",
    description=(
        "Complete video processing pipeline with ML inference, "
        "embedding generation, and vector database integration"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.include_router(management_router)
app.include_router(upload_router)
app.include_router(health_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {
        "service": "Video Processing Orchestration API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload": "/api/uploads",
            "management": "/api/management",
            "health": "/api/uploads/health"
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
