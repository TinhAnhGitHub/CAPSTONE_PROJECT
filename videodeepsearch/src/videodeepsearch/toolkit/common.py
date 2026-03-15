"""Common utilities and classes shared across toolkits.

This module provides:
- CacheManager: For accessing agno function cache results
- SearchResultContainer: For holding and transforming search results
- SparseEncoderInterface: Placeholder for sparse encoder integration
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

from videodeepsearch.schemas import ImageInterface, SegmentInterface


# =============================================================================
# Sparse Encoder Interface (Placeholder)
# =============================================================================

class SparseEncoderInterface:
    """Placeholder interface for sparse encoder (SPLADE/BM25).

    Implement this interface when integrating the actual sparse encoder service.
    """

    async def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode texts to sparse vectors.

        Args:
            texts: List of texts to encode

        Returns:
            List of sparse vectors as {token_index: value}

        Raises:
            NotImplementedError: Until actual implementation is provided
        """
        raise NotImplementedError(
            "Sparse encoder not implemented. "
            "Implement SparseEncoderInterface.encode() to enable hybrid search."
        )


# =============================================================================
# Cache Manager for agno Function Cache Access
# =============================================================================

class CacheManager:
    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path(gettempdir()) / "agno_cache" / "functions"

    def _generate_cache_key(self, function_name: str, args: dict[str, Any]) -> str:
        args_str = json.dumps(args, sort_keys=True, default=str)
        return hashlib.md5(f"{function_name}:{args_str}".encode()).hexdigest()

    def _get_cache_file_path(self, function_name: str, cache_key: str) -> Path:
        return self.cache_dir / function_name / f"{cache_key}.json"

    def get_cached_result(self, function_name: str, args: dict[str, Any]) -> tuple[Any | None, bool]:
        try:
            cache_key = self._generate_cache_key(function_name, args)
            cache_file = self._get_cache_file_path(function_name, cache_key)

            if not cache_file.exists():
                return None, False

            with open(cache_file, "r") as f:
                cache_data = json.load(f)

            return cache_data.get("result"), True

        except Exception as e:
            logger.warning(f"Failed to read cache for {function_name}: {e}")
            return None, False


# =============================================================================
# Search Result Container
# =============================================================================

class SearchResultContainer(BaseModel):
    """Container for search results with view transformation support.

    This container holds the raw search results and provides methods
    to transform them into different view formats for the agent.
    """

    tool_name: str
    tool_kwargs: dict[str, Any]
    results: list[ImageInterface | SegmentInterface]
    result_type: Literal["image", "segment"]

    def get_brief(self, top_n: int = 5) -> str:
        """Get brief representation of top N results.

        Args:
            top_n: Number of top results to show

        Returns:
            Brief string representation
        """
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            f"Total: {len(self.results)} {self.result_type}(s)",
            f"Top {min(top_n, len(sorted_results))}:",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"  {i}. {item.brief_representation()}")

        return "\n".join(lines)

    def get_detailed(self, top_n: int = 5) -> str:
        """Get detailed representation of top N results.

        Args:
            top_n: Number of top results to show

        Returns:
            Detailed string representation
        """
        sorted_results = sorted(self.results, key=lambda x: x.score, reverse=True)[:top_n]

        lines = [
            "=== Detailed Results ===",
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            f"Total: {len(self.results)} {self.result_type}(s)",
            f"Top {min(top_n, len(sorted_results))}:",
            "",
        ]

        for i, item in enumerate(sorted_results):
            lines.append(f"[{i}] {item.detailed_representation()}")
            lines.append("")

        return "\n".join(lines)

    def get_statistics(self, group_by: str = "video_id") -> str:
        """Get statistics of results grouped by a field.

        Args:
            group_by: Field to group by ('video_id' or 'score_bucket')

        Returns:
            Statistics string representation
        """
        if self.result_type == "image":
            return ImageInterface.statistic_format(
                tool_name=self.tool_name,
                tool_kwargs=self.tool_kwargs,
                handle_id="local",
                items=self.results,
                group_by=group_by,
            )
        else:
            return SegmentInterface.statistic_format(
                tool_name=self.tool_name,
                tool_kwargs=self.tool_kwargs,
                handle_id="local",
                items=self.results,
                group_by=group_by,
            )

    def get_full(self) -> str:
        """Get full representation of all results.

        Returns:
            Full string representation with all items
        """
        lines = [
            f"=== Full Results ({len(self.results)} items) ===",
            f"Tool: {self.tool_name}",
            f"Args: {self.tool_kwargs}",
            "",
        ]

        for i, item in enumerate(self.results):
            lines.append(f"[{i}] {item.detailed_representation()}")
            lines.append("")

        return "\n".join(lines)


__all__ = [
    "CacheManager",
    "SearchResultContainer",
    "SparseEncoderInterface",
]