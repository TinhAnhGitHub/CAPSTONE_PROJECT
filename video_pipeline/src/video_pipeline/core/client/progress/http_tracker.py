from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import httpx
from pydantic import BaseModel


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StageProgress(BaseModel):
    stage: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    complete:bool = False
    details: dict[str, Any] | None = None


class VideoProgress(BaseModel):
    run_id: str
    overall_percentage: float = 0.0
    current_stage: str | None = None
    stages: dict[str, StageProgress] | None = None
    start_time: datetime | None = None
    end_time: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING
    error: Optional[str] = None


class HTTPProgressTracker:
    """Posts pipeline progress to an external HTTP endpoint after every stage completes.

    No locking or throttling — updates are fired once per stage, so the request
    volume is low and always meaningful.
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/api/ingestion/service/status/{video_id}",
    ) -> None:
        self._progress: dict[str, VideoProgress] = {}
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint

    async def _post(self, video_id: str) -> None:
        progress = self._progress.get(video_id)
        if not progress:
            return
        payload = progress.model_dump(mode="json")
        try:
            async with httpx.AsyncClient(base_url=self.base_url) as client:
                r = await client.post(
                    self.endpoint.format(video_id=video_id), json=payload
                )
                r.raise_for_status()
        except Exception as e:
            print(f"Error on http tracker: {e=}")
            pass  

    def _recalc_overall(self, progress: VideoProgress) -> None:
        if not progress.stages or len(progress.stages) == 0:
            progress.overall_percentage = 0.0
            return
        n = len(progress.stages)
        done = sum(1 for s in progress.stages.values() if s.complete)
        progress.overall_percentage = min((done / n) * 100, 100.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start_video(self, video_id: str, stage_names: list[str]) -> None:
        """Initialise tracking for a video run with the given stage names."""
        self._progress[video_id] = VideoProgress(
            run_id=video_id,
            stages={name: StageProgress(stage=name) for name in stage_names},
            start_time=datetime.now(timezone.utc), # Use UTC
            overall_percentage=0.0,
        )
        await self._post(video_id)

    async def update_stage(
        self,
        video_id: str,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        progress = self._progress.get(video_id)
        if not progress or not progress.stages or stage not in progress.stages:
            return
        sp = progress.stages[stage]
        sp.details = details or {}
        if sp.start_time is None:
            sp.start_time = datetime.now()
        progress.current_stage = stage
        self._recalc_overall(progress)
        await self._post(video_id)

    async def complete_stage(self, video_id: str, stage: str) -> None:
        progress = self._progress.get(video_id)
        if not progress or not progress.stages or stage not in progress.stages:
            return
        sp = progress.stages[stage]
        sp.complete = True
        sp.end_time = datetime.now()
        self._recalc_overall(progress)
        await self._post(video_id)

    async def complete_run(
        self,
        video_id: str,
        status: RunStatus = RunStatus.COMPLETED,
        error: str | None = None,
    ) -> None:
        progress = self._progress.get(video_id)
        if not progress:
            return
        progress.status = status
        progress.end_time = datetime.now()
        progress.error = error
        if status == RunStatus.COMPLETED:
            progress.overall_percentage = 100.0
        await self._post(video_id)

    def get(self, video_id: str) -> VideoProgress | None:
        return self._progress.get(video_id)

    def remove(self, video_id: str) -> None:
        self._progress.pop(video_id, None)

    def clear(self) -> None:
        self._progress.clear()
