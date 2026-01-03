from __future__ import annotations

import json
import numpy as np
from typing import Sequence, TypeVar, Callable, Any

from pydantic import BaseModel
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import ToolCall, ToolCallResult

from videodeepsearch.agent.context.worker_context import SmallWorkerContext
from .data_handle import DataHandle

T = TypeVar("T")


def extract_scores(items: Sequence[T], score_getter: Callable[[T], float | None]) -> list[float]:
    return [s for x in items if (s := score_getter(x)) is not None]




def build_score_stats(scores: Sequence[float]) -> dict[str, Any]:
    if not scores:
        return {"avg": 0.0, "range": "N/A", "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "avg": sum(scores) / len(scores),
        "range": f"{min(scores):.3f} - {max(scores):.3f}",
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
    }


def build_top_summary(
    items: Sequence[T],
    score_getter: Callable[[T], float | None],
    formatter: Callable[[int, T], str],
    limit: int = 10,
) -> str:
    sorted_items = sorted(items, key=lambda x: score_getter(x) or 0.0, reverse=True)[:limit]
    lines = [formatter(i + 1, item) for i, item in enumerate(sorted_items)]
    return "\n".join(lines) or "No high-scoring results."


def make_handle(
    summary: dict,
    video_ids: list[str],
) -> DataHandle:
    return DataHandle(
        summary=summary,
        related_video_ids=video_ids,
        tool_used=None
    )

def build_video_ids(items: Sequence[T], video_id_getter: Callable[[T], str]) -> list[str]:
    return list({video_id_getter(item) for item in items})