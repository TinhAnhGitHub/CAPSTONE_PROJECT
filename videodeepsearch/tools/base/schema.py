from __future__ import annotations
from statistics import mean
from pydantic import BaseModel
from collections import defaultdict
from typing import Callable, Any, Sequence
from abc import ABC, abstractmethod
from llama_index.core.llms import ImageBlock

from ingestion.core.artifact.schema import SegmentCaptionArtifact, ImageCaptionArtifact



class ImageBytes(BaseModel):
    image_bytes: bytes

    def convert_to_image_block(self) -> ImageBlock:
        return ImageBlock(
            image=self.image_bytes
        )
        
class BaseInterface(BaseModel, ABC):
    id: str
    related_video_id: str
    minio_path: str
    user_bucket: str

    @abstractmethod
    def accept_filter(self, filter_fn: Callable[[Any], bool]) -> bool:
        ...

    @abstractmethod
    def brief_representation(self) -> str:
        ...

    @abstractmethod
    def detailed_representation(self) -> str:
        ...

    @staticmethod
    @abstractmethod
    def quick_format(       
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        
        ...

    @staticmethod
    @abstractmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str
    ) -> str:
        ...


class ImageInterface(BaseInterface):
    frame_index: int
    timestamp: str
    image_caption: str
    score: float 

    

    def accept_filter(self, filter_fn: Callable[[ImageInterface], bool]) -> bool:
        return filter_fn(self)
        
    def brief_representation(self) -> str:
        score_str = f"{self.score:.3f}" if self.score is not None else "No score"
        return (
            f"score={score_str} | {self.related_video_id} "
            f"@ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} "
            f"| minio_path: {self.minio_path}"
        )

    def detailed_representation(self) -> str:
        representation = f"""
        score={self.score:.3f} | {self.related_video_id} @ Timestamp/FrameIndex: {self.timestamp}/{self.frame_index} | minio_pat: {self.minio_path} | image caption: {self.image_caption}
        """
        return representation


    @staticmethod
    def quick_format(       
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        items = [i for i in items if isinstance(i, ImageInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]
        tool_uses_lines = [
            f"Tool name: {tool_name}",
            f"Tool input kwargs: {tool_kwargs}",
        ]

        tool_uses_text = "\n".join(tool_uses_lines)

        lines = [
            f"Handle id: {handle_id}",
            f"Tool uses patterns:\n{tool_uses_text}",
            f"Total: {len(items)} images",
            "Top 5:",
        ]

        for i, item in enumerate(top_5):
            lines.append(
                f"   {i}. {item.detailed_representation()}"
            )
        return '\n'.join(lines)

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str
    ) -> str:
        current_items = [i for i in items if isinstance(i, ImageInterface)]
        grouped_data = defaultdict(list)

        for item in current_items:
            key = 'Unknown'
            if group_by == 'video_id':
                key = item.related_video_id
            
            elif group_by == 'score_bucket':
                bucket = int(item.score * 10) / 10.0
                key = f"Score range [{bucket:.1f} - {bucket + 0.1:.1f}]"
            
  
        
            grouped_data[key].append(item)
        
        lines = [
            f"=== Statistic Report (Images) ===",
            f"Handle ID: {handle_id}",
            f"Tool: {tool_name} | Strategy: {group_by}",
            f"Total Items: {len(current_items)}",
            "-" * 40
        ]

        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == 'score_bucket'))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)
            
            lines.append(f"Group: {key}")
            lines.append(f"  • Count: {count}")
            lines.append(f"  • Avg Score: {avg_score:.3f}")
            lines.append(f"  • Best Match: {best_item.brief_representation()}")
            lines.append("")
        return '\n'.join(lines)
        

    @classmethod
    def from_artifact(
        cls,
        artifact: ImageCaptionArtifact,
        caption: str 
    ):
        return cls(
            id  = artifact.artifact_id,
            frame_index = artifact.frame_index, 
            timestamp=artifact.time_stamp, 
            related_video_id=artifact.related_video_id, 
            image_caption=caption, 
            minio_path=artifact.minio_url_path, 
            user_bucket=artifact.user_bucket, 
            score=0.0
        )


class SegmentInterface(BaseInterface):
    start_frame: int
    end_frame: int
    start_time: str
    end_time: str
    segment_caption:str
    score: float

    def accept_filter(self, filter_fn: Callable[[SegmentInterface], bool]) -> bool:
        return filter_fn(self)
    
    def brief_representation(self) -> str:
        return (
            f"[{self.related_video_id}] "
            f"Frames {self.start_frame}-{self.end_frame} "
            f"({self.start_time} → {self.end_time}) | "
            f"score={self.score:.3f}"
            f"Minio path: {self.minio_path}"
        )
    

    def detailed_representation(self) -> str:
        return (
            f"Segment {self.id} | Video: {self.related_video_id}\n"
            f"- Frames: {self.start_frame} → {self.end_frame}\n"
            f"- Time: {self.start_time} → {self.end_time}\n"
            f"- Score: {self.score:.3f}\n"
            f"- Caption: {self.segment_caption}\n"
            f"- Caption MinIO URL: {self.minio_path}"
        )
    
    @staticmethod
    def quick_format(       
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
    ) -> str:
        items = [i for i in items if isinstance(i, SegmentInterface)]
        top_5 = sorted(items, key=lambda x: x.score, reverse=True)[:5]

        tool_uses_lines = [
            f"Tool name: {tool_name}",
            f"Tool input kwargs: {tool_kwargs}",
        ]
        tool_uses_text = "\n".join(tool_uses_lines)
        lines = [
            f"Handle id: {handle_id}",
            f"Tool uses patterns:\n{tool_uses_text}",
            f"Total: {len(items)} images",
            f"Top 5:"
        ]

        for i, item in enumerate(top_5):
            lines.append(
                f"   {i}. {item.detailed_representation()}"
            )
        return '\n'.join(lines)


    @classmethod
    def from_artifact(
        cls,
        artifact: SegmentCaptionArtifact,
        caption: str 
    ):
        return cls(
            id  = artifact.artifact_id,
            start_frame = artifact.start_frame, 
            end_frame=artifact.end_frame,
            start_time=artifact.start_timestamp,
            end_time=artifact.end_timestamp, 
            related_video_id=artifact.related_video_id, 
            segment_caption=caption, 
            minio_path=artifact.minio_url_path, 
            user_bucket=artifact.user_bucket, 
            score=0.0
        )

    @staticmethod
    def statistic_format(
        tool_name: str,
        tool_kwargs: dict,
        handle_id: str,
        items: Sequence[BaseInterface],
        group_by: str
    ) -> str:
        
        current_items = [i for i in items if isinstance(i, SegmentInterface)]
        
        grouped_data = defaultdict(list)
        
        for item in current_items:
            key = "Unknown"
            
            if group_by == 'video_id':
                key = item.related_video_id
                
            elif group_by == 'score_bucket':
                bucket = int(item.score * 10) / 10.0
                key = f"Score range [{bucket:.1f} - {bucket + 0.1:.1f}]"
                
     
            
            grouped_data[key].append(item)

        lines = [
            f"=== Statistic Report (Segments) ===",
            f"Handle ID: {handle_id}",
            f"Tool: {tool_name} | Strategy: {group_by}",
            f"Total Items: {len(current_items)}",
            "-" * 10
        ]

        sorted_keys = sorted(grouped_data.keys(), reverse=(group_by == 'score_bucket'))

        for key in sorted_keys:
            group_items = grouped_data[key]
            count = len(group_items)
            avg_score = mean([x.score for x in group_items]) if group_items else 0.0
            best_item = max(group_items, key=lambda x: x.score)
            
            lines.append(f"Group: {key}")
            lines.append(f"  • Count: {count}")
            lines.append(f"  • Avg Score: {avg_score:.3f}")
            lines.append(f"  • Best Match: {best_item.brief_representation()}")
            lines.append("")

        return '\n'.join(lines)
