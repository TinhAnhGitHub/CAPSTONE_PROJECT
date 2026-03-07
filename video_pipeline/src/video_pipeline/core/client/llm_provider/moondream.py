"""
Moondream Vision API Client
1. Standard API: real-time caption & detect (async, aiohttp)
2. Batch API: bulk processing via multipart upload for large-scale workloads
"""
from tqdm.asyncio import tqdm
import asyncio
import base64
import json
import time
from pathlib import Path
import aiohttp
from loguru import logger
from pydantic import BaseModel

class MoondreamConfig(BaseModel):
    api_key: str
    base_url: str = "https://api.moondream.ai/v1"
    model: str = "moondream3-preview"
    timeout: int = 120
    batch_poll_interval: int = 10  # seconds between status polls
    batch_chunk_size: int = 50 * 1024 * 1024  # 50 MB

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class CaptionResult(BaseModel):
    request_id: str | None = None
    caption: str


class DetectResult(BaseModel):
    request_id: str | None = None
    objects: list[BoundingBox]


class BatchStatus(BaseModel):
    id: str
    status: str  # chunking | processing | completed | failed
    progress: dict | None = None
    outputs: list[dict] | None = None
    error: dict | None = None


class MoondreamClient:
    def __init__(self, config: MoondreamConfig):
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.model = config.model
        self.timeout = config.timeout
        self.batch_poll_interval = config.batch_poll_interval
        self.batch_chunk_size = config.batch_chunk_size
        self._session: aiohttp.ClientSession | None = None


    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-Moondream-Auth": self.api_key}
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


    @staticmethod
    def encode_image(image_path: str | Path) -> str:
        """Return a data URI string for an image file."""
        path = Path(image_path)
        ext = path.suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
            "gif": "image/gif",
        }.get(ext, "image/png")
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def encode_image_bytes(data: bytes, mime: str = "image/jpeg") -> str:
        """Return a data URI string from raw bytes."""
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:{mime};base64,{b64}"


    async def caption(
        self,
        image_url: str,
        length: str = "normal",
    ) -> CaptionResult | None:
        """
        Generate a caption for a single image.

        Args:
            image_url: data-URI (use encode_image / encode_image_bytes)
            length: "short" | "normal" | "long"
        """
        try:
            session = await self._get_session()
            payload = {
                "image_url": image_url,
                "length": length,
                "stream": False,
            }
            async with session.post(
                f"{self.base_url}/caption",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Moondream caption HTTP {resp.status}: {body}")
                    return None
                data = await resp.json()
                return CaptionResult(
                    request_id=data.get("request_id"),
                    caption=data["caption"],
                )
        except Exception:
            logger.exception("Moondream caption failed")
            return None

    async def detect(
        self,
        image_url: str,
        object_name: str,
    ) -> DetectResult | None:
        """
        Detect objects in a single image.

        Args:
            image_url: data-URI
            object_name: what to detect (e.g. "person", "car")
        """
        try:
            session = await self._get_session()
            payload = {
                "image_url": image_url,
                "object": object_name,
            }
            async with session.post(
                f"{self.base_url}/detect",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(f"Moondream detect HTTP {resp.status}: {body}")
                    return None
                data = await resp.json()
                return DetectResult(
                    request_id=data.get("request_id"),
                    objects=[BoundingBox(**o) for o in data.get("objects", [])],
                )
        except Exception:
            logger.exception("Moondream detect failed")
            return None

    async def caption_many(
        self,
        image_urls: list[str],
        length: str = "normal",
        max_concurrent: int = 10,
    ) -> list[CaptionResult | None]:
        """
        Caption multiple images concurrently (standard API).
        Good for moderate volumes (<100). For thousands, use batch_caption.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(url: str):
            async with semaphore:
                return await self.caption(url, length=length)

        return list(await asyncio.gather(*[_limited(u) for u in image_urls]))

    async def detect_many(
        self,
        image_urls: list[str],
        object_name: str,
        max_concurrent: int = 10,
    ) -> list[DetectResult | None]:
        """Detect on multiple images concurrently (standard API)."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _limited(url: str):
            async with semaphore:
                return await self.detect(url, object_name)

        return list(await asyncio.gather(*[_limited(u) for u in image_urls]))

    def _build_batch_line(
        self,
        image_path: str | Path,
        skill: str,
        line_id: str | None = None,
        **kwargs,
    ) -> str:
        """Build a single JSONL line for the batch input file."""
        path = Path(image_path)
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        line: dict = {"skill": skill, "image": b64}
        if line_id is not None:
            line["id"] = line_id
        line.update(kwargs)
        return json.dumps(line, ensure_ascii=False)

    def prepare_batch_caption(
        self,
        image_paths: list[str | Path],
        length: str = "normal",
        ids: list[str] | None = None,
    ) -> str:
        """
        Build JSONL content for a batch captioning job.

        Returns:
            The JSONL string (write to file or pass to submit_batch).
        """
        lines = []
        for i, p in enumerate(image_paths):
            line_id = ids[i] if ids else f"img_{i:06d}"
            lines.append(
                self._build_batch_line(p, skill="caption", line_id=line_id, length=length)
            )
        return "\n".join(lines)

    def prepare_batch_detect(
        self,
        image_paths: list[str | Path],
        object_name: str,
        ids: list[str] | None = None,
    ) -> str:
        """Build JSONL content for a batch detection job."""
        lines = []
        for i, p in enumerate(image_paths):
            line_id = ids[i] if ids else f"img_{i:06d}"
            lines.append(
                self._build_batch_line(
                    p, skill="detect", line_id=line_id, object=object_name
                )
            )
        return "\n".join(lines)

    async def submit_batch(self, jsonl_content: str | bytes) -> str | None:
        """
        Upload a JSONL payload via multipart upload and start batch processing.

        Args:
            jsonl_content: JSONL string or bytes

        Returns:
            batch_id on success, None on failure
        """
        if isinstance(jsonl_content, str):
            jsonl_content = jsonl_content.encode("utf-8")

        try:
            session = await self._get_session()

            async with session.post(
                f"{self.base_url}/batch?action=mpu-create",
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Batch mpu-create failed: {await resp.text()}")
                    return None
                init = await resp.json()

            file_id = init["fileId"]
            upload_id = init["uploadId"]

            parts: list[dict] = []
            chunk_size = self.batch_chunk_size
            total = len(jsonl_content)

            for part_num, offset in tqdm(enumerate(range(0, total, chunk_size), start=1)):
                chunk = jsonl_content[offset : offset + chunk_size]
                url = (
                    f"{self.base_url}/batch/{file_id}"
                    f"?action=mpu-uploadpart&uploadId={upload_id}&partNumber={part_num}"
                )
                async with session.put(
                    url,
                    data=chunk,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status != 200:
                        logger.error(
                            f"Batch upload part {part_num} failed: {await resp.text()}"
                        )
                        await self._abort_upload(file_id, upload_id)
                        return None
                    part_data = await resp.json()
                    parts.append(
                        {"partNumber": part_num, "etag": part_data["etag"]}
                    )

            complete_payload: dict = {"parts": parts}
            if self.model:
                complete_payload["model"] = self.model

            async with session.post(
                f"{self.base_url}/batch/{file_id}?action=mpu-complete&uploadId={upload_id}",
                json=complete_payload,
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Batch complete failed: {await resp.text()}")
                    return None
                result = await resp.json()
                batch_id = result["id"]
                logger.success(f"Batch submitted: {batch_id}")
                return batch_id

        except Exception:
            logger.exception("Batch submission failed")
            return None

    async def _abort_upload(self, file_id: str, upload_id: str):
        try:
            session = await self._get_session()
            await session.delete(
                f"{self.base_url}/batch/{file_id}?action=mpu-abort&uploadId={upload_id}"
            )
        except Exception:
            logger.exception("Failed to abort upload")

    async def get_batch_status(self, batch_id: str) -> BatchStatus | None:
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/batch/{batch_id}") as resp:
                if resp.status != 200:
                    logger.error(f"Batch status failed: {await resp.text()}")
                    return None
                data = await resp.json()
                return BatchStatus(**data)
        except Exception:
            logger.exception("Batch status check failed")
            return None

    async def wait_for_batch(
        self,
        batch_id: str,
        timeout: int = 3600,
    ) -> BatchStatus | None:
        """
        Poll until the batch completes or times out.

        Args:
            batch_id: ID returned from submit_batch
            timeout: max wait time in seconds (default 1 hour)
        """
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            status = await self.get_batch_status(batch_id)
            if status is None:
                return None
            if status.status == "completed":
                logger.success(f"Batch {batch_id} completed")
                return status
            if status.status == "failed":
                logger.error(f"Batch {batch_id} failed: {status.error}")
                return status

            progress = status.progress or {}
            logger.info(
                f"Batch {batch_id}: {status.status} "
                f"({progress.get('completed', '?')}/{progress.get('total', '?')})"
            )
            await asyncio.sleep(self.batch_poll_interval)

        logger.error(f"Batch {batch_id} timed out after {timeout}s")
        return None

    async def download_batch_results(
        self, status: BatchStatus
    ) -> list[dict] | None:
        """
        Download and parse all result JSONL files from a completed batch.

        Args:
            status: A BatchStatus with status == "completed"
        """
        if not status.outputs:
            logger.error("No output URLs in batch status")
            return None

        try:
            session = await self._get_session()
            all_results: list[dict] = []

            for output in status.outputs:
                url = output["url"]
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=300)
                ) as resp:
                    if resp.status != 200:
                        logger.error(f"Download failed for {url}: HTTP {resp.status}")
                        continue
                    text = await resp.text()
                    for line in text.strip().splitlines():
                        if line.strip():
                            all_results.append(json.loads(line))

            return all_results
        except Exception:
            logger.exception("Batch result download failed")
            return None

    async def batch_caption(
        self,
        image_paths: list[str | Path],
        length: str =  "normal",
        ids: list[str] | None = None,
        timeout: int = 3600,
    ) -> list[dict] | None:
        """
        End-to-end batch captioning: prepare → upload → wait → download.

        For thousands of images where latency is not critical.
        50% cheaper than standard API.
        """
        jsonl = self.prepare_batch_caption(image_paths, length=length, ids=ids)
        batch_id = await self.submit_batch(jsonl)
        if not batch_id:
            return None
        status = await self.wait_for_batch(batch_id, timeout=timeout)
        if not status or status.status != "completed":
            return None
        return await self.download_batch_results(status)

    async def batch_detect(
        self,
        image_paths: list[str | Path],
        object_name: str,
        ids: list[str] | None = None,
        timeout: int = 3600,
    ) -> list[dict] | None:
        """End-to-end batch detection: prepare → upload → wait → download."""
        jsonl = self.prepare_batch_detect(
            image_paths, object_name=object_name, ids=ids
        )
        batch_id = await self.submit_batch(jsonl)
        if not batch_id:
            return None
        status = await self.wait_for_batch(batch_id, timeout=timeout)
        if not status or status.status != "completed":
            return None
        return await self.download_batch_results(status)