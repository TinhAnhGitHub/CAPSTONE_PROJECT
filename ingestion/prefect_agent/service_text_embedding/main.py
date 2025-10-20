import uvicorn
from fastapi import Depends, FastAPI
from fastapi.responses import Response
from loguru import logger
import sys
import os


os.environ.setdefault("HF_HUB_ENABLE_XET", "0")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
ROOT_DIR = os.path.abspath(
    os.path.join(
        __file__, '../..'
    )
)
print(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)


from service_text_embedding.core.api import router
from service_text_embedding.core.config import text_embedding_config
from service_text_embedding.core.dependencies import get_service
from service_text_embedding.core.lifespan import lifespan

app = FastAPI(
    title="Text Embedding Service",
    description="Generate vector representations for text using configurable backends",
    version=text_embedding_config.service_version,
    lifespan=lifespan,
)

app.include_router(router, prefix="/text-embedding", tags=["text-embedding"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": text_embedding_config.service_name}


@app.get("/metrics")
async def metrics(service=Depends(get_service)) -> Response:
    service.update_system_metrics()
    return Response(
        content=service.metrics.get_metrics(),
        media_type=service.metrics.get_content_type(),
    )


if __name__ == "__main__":
    logger.info(
        "starting_text_embedding_service",
        host=text_embedding_config.host,
        port=text_embedding_config.port,
        version=text_embedding_config.service_version,
    )
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=text_embedding_config.port,
        reload=True,
        log_level="info",
    )
