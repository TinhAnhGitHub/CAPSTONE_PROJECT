from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
import httpx
from app.core.config import settings

router = APIRouter(prefix="/media", tags=["media"])

@router.get("/{bucket}/{file_path:path}")
async def proxy_media(bucket: str, file_path: str, request: Request):
    # if bucket not in ["videos", "thumbnails", "avatars"]:
    #     raise HTTPException(status_code=400, detail="Invalid bucket")
        
    minio_url = f"http://{settings.MINIO_PUBLIC_ENDPOINT}/{bucket}/{file_path}"
    
    # Forward relevant headers for media streaming
    headers = {}
    for k, v in request.headers.items():
        k_lower = k.lower()
        if k_lower in ("range", "if-match", "if-none-match", "if-modified-since", "if-unmodified-since"):
            headers[k] = v
            
    # 5 seconds connect timeout, no read timeout for streaming
    timeout = httpx.Timeout(5.0, read=None)
    client = httpx.AsyncClient(timeout=timeout)
    
    req = client.build_request("GET", minio_url, headers=headers)
    response = await client.send(req, stream=True)
    
    if response.status_code == 404:
        await client.aclose()
        raise HTTPException(status_code=404, detail="Media not found")
        
    async def stream_generator():
        try:
            async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()
            
    resp_headers = {}
    for k, v in response.headers.items():
        k_lower = k.lower()
        if k_lower in ("content-type", "content-length", "content-range", "accept-ranges", "etag", "last-modified"):
            resp_headers[k] = v
            
    return StreamingResponse(
        stream_generator(),
        status_code=response.status_code,
        headers=resp_headers
    )
