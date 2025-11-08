from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from enum import Enum
from threading import Lock
import logging
import httpx
from task import VideoIngestionTask, AutoshotProcessingTask, ASRProcessingTask, ImageProcessingTask, SegmentCaptionLLMTask, ImageCaptionLLMTask, ImageEmbeddingMilvusTask, ImageEmbeddingTask, TextSegmentCaptionMilvusTask, TextCaptionSegmentEmbeddingTask, TextImageCaptionEmbeddingTask


logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    VIDEO_INGEST = VideoIngestionTask.__name__
    AUTOSHOT_SEGMENTATION = AutoshotProcessingTask.__name__
    ASR_TRANSCRIPTION = ASRProcessingTask.__name__
    IMAGE_EXTRACTION = ImageProcessingTask.__name__
    SEGMENT_CAPTIONING = SegmentCaptionLLMTask.__name__
    IMAGE_CAPTIONING = ImageCaptionLLMTask.__name__
    IMAGE_EMBEDDING = ImageEmbeddingTask.__name__
    TEXT_CAP_SEGMENT_EMBEDDING = TextCaptionSegmentEmbeddingTask.__name__
    TEXT_CAP_IMAGE_EMBEDDING = TextImageCaptionEmbeddingTask.__name__
    IMAGE_MILVUS = ImageEmbeddingMilvusTask.__name__
    TEXT_CAP_SEGMENT_MILVUS = TextSegmentCaptionMilvusTask.__name__

class RunStatus(str,Enum):
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'


class StageProgress(BaseModel):
    stage: ProcessingStage
    total_items: int = 0
    completed_items: int = 0
    percentage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    details: dict[str, Any]|None = None


class VideoProgress(BaseModel):
    run_id: str
    overall_percentage: float = 0.0
    current_stage: Optional[ProcessingStage] = None
    stages: Dict[ProcessingStage, StageProgress] | None = None
    start_time: datetime | None= None
    end_time: Optional[datetime] = None
    status: RunStatus = RunStatus.RUNNING  # running, completed, failed
    error: Optional[str] = None

class HTTPProgressTracker:
    def __init__(
        self,
        base_url:str,
        endpoint: str = '/api/ingestion/service/status/{video_id}'
    ) -> None:
        self._progress: Dict[str, VideoProgress] = {}
        self._lock = Lock()
        self.base_url =base_url
        self.endpoint = endpoint
    
    async def _trigger_http(self, video_id:str):
        with self._lock:
            progress = self._progress.get(video_id)
            if not progress:
                return None
            payload = progress.model_dump(mode='json')
        try:
            async with httpx.AsyncClient(base_url="http://100.120.22.90:8010") as client:
                print(f"{self.endpoint.format(video_id=video_id)}")
                print(f"{self.base_url}")
                response = await client.post(
                    self.endpoint.format(video_id=video_id),
                    json=payload
                )
                response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            print(
                "Failed to send progress update for",
                video_id,
                exc,
            )
            return None

    async def start_video(self, video_id: str) -> None:
        with self._lock:
            self._progress[video_id] = VideoProgress(
                run_id=video_id,
                stages={stage: StageProgress(stage=stage) for stage in ProcessingStage},
                start_time=datetime.now(),
                overall_percentage=0.0
            )
        await self._trigger_http(video_id)
    
    async def update_stage_progress(
        self, 
        video_id: str,
        stage: ProcessingStage,
        total_items: int,
        completed_items: int,
        details: dict[str, Any] | None=None
    ):
        
        with self._lock:
            if video_id not in self._progress:
                return  

            run_progress = self._progress[video_id]
            if not run_progress.stages or ( run_progress.stages and stage not in run_progress.stages):
                return

            stage_progress = run_progress.stages[stage]
            stage_progress.total_items = total_items
            stage_progress.completed_items = completed_items

            stage_progress.percentage = (completed_items / total_items * 100) if total_items > 0 else 0.0

            stage_progress.details = details or {}
            if stage_progress.start_time is None:
                stage_progress.start_time = datetime.now()
            
            
            
            completed_stages = sum(1 for s in run_progress.stages.values() if s.percentage >= 100)


            run_progress.overall_percentage = (
                (completed_stages / len(ProcessingStage)) * 100
                + (stage_progress.percentage / len(ProcessingStage))
            )
            run_progress.overall_percentage = min(run_progress.overall_percentage, 100.0)
            

            run_progress.current_stage = stage

        await self._trigger_http(video_id)
        
    async def complete_stage(self, video_id: str, stage: ProcessingStage) -> None:
        with self._lock:
            if video_id in self._progress and stage in self._progress[video_id].stages: #type:ignore
                self._progress[video_id].stages[stage].end_time = datetime.now() #type:ignore
        
        await self._trigger_http(video_id)
        
    async def complete_run(self, video_id: str, status: RunStatus = RunStatus.COMPLETED, error: Optional[str] = None) -> None:
        with self._lock:
            if video_id in self._progress:
                self._progress[video_id].status = status
                self._progress[video_id].end_time = datetime.now()
                self._progress[video_id].error = error
                if status == "completed":
                    self._progress[video_id].overall_percentage = 100.0
        await self._trigger_http(video_id)
    
    async def remove_video_id(self, video_id: str) -> None:
        with self._lock:
            self._progress.pop(video_id, None)
    

    def clear_video_progress_cache(self):
        self._progress.clear()


            

