# output.py
import json
from videodeepsearch.tools.base.schema import ImageInterface, SegmentInterface

from videodeepsearch.tools.base.schema import ImageBytes
from llama_index.core.llms import ImageBlock
from llama_index.core.agent.workflow import ToolCall, ToolCallResult

from .helpers import (
    extract_scores,
    build_score_stats,
    make_handle,
    build_top_summary,
)
from .helpers import build_video_ids
from .data_handle import DataHandle


### HARDCODE
VIDEO_ID2FPS = {
    '692ad412086ada3a309334ff':'25.0',
    '692ad412086ada3a30933500': '30.0',
    '692ad412086ada3a30933501': '30.0',
    '692ad412086ada3a30933502': '30.0'
}



def _format_image_item(rank: int, item: ImageInterface) -> dict:
    return {
        "rank": rank,
        "score": round(item.score, 3),
        "video_id": item.related_video_id,
        "timestamp": item.timestamp,
        "frame_index": item.frame_index,
        "minio_path": item.minio_path
    }


def _format_segment_item(rank: int, item: SegmentInterface) -> dict:
    caption = (item.segment_caption or "").strip()
    words = caption.split()
    preview = " ".join(words[:50]) + (" ..." if len(words) > 50 else "") or "(no caption)"

    fps = VIDEO_ID2FPS.get(item.related_video_id, 30.0)
    
    return {
        "rank": rank,
        "score": round(item.score, 3),
        "video_id": item.related_video_id,
        "time_range": {
            "start": item.start_time,
            "end": item.end_time
        },
        "frame_range": {
            "start": item.start_frame,
            "end": item.end_frame
        },
        "caption_preview": preview,
        "minio_path": item.minio_path,
        'fps': fps
    }

def output_image_results(
    images: list[ImageInterface],
) -> DataHandle:
    """
    This is the image output middleware. It will return the DataHandle with handle id, and some metadata (summaries...). After that, please use the viewing toolset to view the results in brief, detail...
    """

    
    scores = extract_scores(images, lambda x: x.score)
        
    video_ids = build_video_ids(images, lambda x: x.related_video_id)
    
    stats = build_score_stats(scores)
    top_matches = [_format_image_item(i + 1, img) for i, img in enumerate(images[:10])]

    summary_dict = {
        "result_type": "image_search",
        "total_results": len(images),
        "video_ids": video_ids[:5],
        "statistics": {
            "avg_score": round(stats['avg'], 3),
            "score_range": stats['range'],
            "percentiles": {
                "p25": round(stats['p25'], 3),
                "p50": round(stats['p50'], 3),
                "p75": round(stats['p75'], 3)
            }
        },
        "top_matches": top_matches
    }
    
    # summary = json.dumps(summary_dict, indent=2, ensure_ascii=False)
    
    handle = make_handle(summary_dict, video_ids)
    handle.set_data(data=images)        
    return handle


def output_segment_results(
    segments: list[SegmentInterface],
) -> DataHandle:
    """
    This is the segment output middleware. It will return the DataHandle with handle id, and some metadata (summaries...). After that, please use the viewing toolset to view the results in brief, detail...
    """

    scores = extract_scores(segments, lambda x: x.score)
    video_ids = build_video_ids(segments, lambda x: x.related_video_id)
    stats = build_score_stats(scores)
    top_matches = [_format_segment_item(i + 1, seg) for i, seg in enumerate(segments[:10])]

    summary_dict = {
        "result_type": "segment_caption_search",
        "total_results": len(segments),
        "video_ids": video_ids[:5],
        "statistics": {
            "avg_score": round(stats['avg'], 3),
            "score_range": stats['range'],
            "percentiles": {
                "p25": round(stats['p25'], 3),
                "p50": round(stats['p50'], 3),
                "p75": round(stats['p75'], 3)
            }
        },
        "top_matches": top_matches
    }

    # summary = json.dumps(summary_dict, indent=2, ensure_ascii=False)

    handle = make_handle(summary_dict, video_ids)
    handle.set_data(data=segments)
    return handle


def output_image_bytes(image_bytes: list[ImageBytes]) -> list[ImageBlock]:
    """
    This is the image bytes output middleware. The function will return a list of ImageBlock that you will read the images directly.
    """

    return [img_byt.convert_to_image_block() for img_byt in image_bytes]