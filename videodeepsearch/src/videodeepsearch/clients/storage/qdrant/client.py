"""Base Qdrant client with common search functionality."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Prefetch,
    Rrf,
    RrfQuery,
    ScoredPoint,
    SparseVector,
)

T = TypeVar("T")


class BaseQdrantClient(Generic[T], ABC):
    """Base class for Qdrant search clients.

    Provides common functionality for connecting to Qdrant and searching vectors.
    Subclasses implement the specific mapping from Qdrant results to domain objects.
    """

    def __init__(
        self,
        host: str,
        port: int,
        collection_name: str,
        grpc_port: int = 6334,
        prefer_grpc: bool = True,
    ):
        """Initialize the Qdrant client.

        Args:
            host: Qdrant server host
            port: Qdrant HTTP port (default 6333)
            collection_name: Name of the Qdrant collection
            grpc_port: Qdrant gRPC port (default 6334)
            prefer_grpc: Whether to prefer gRPC over HTTP (default True)
        """
        self.client = AsyncQdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )
        self.base_collection_name = collection_name

    async def close(self) -> None:
        """Close the Qdrant client connection."""
        await self.client.close()

    def build_filter(
        self,
        video_ids: list[str] | None = None,
        user_id: str | None = None,
    ) -> Filter | None:
        """Build a Qdrant filter from common parameters.

        Args:
            video_ids: List of video IDs to filter by
            user_id: User ID to filter by

        Returns:
            Qdrant Filter object or None if no filters
        """
        conditions = []

        if video_ids:
            conditions.append(
                FieldCondition(
                    key="related_video_id",
                    match=MatchAny(any=video_ids),
                )
            )

        if user_id:
            conditions.append(
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    async def search_dense(
        self,
        query_vector: list[float],
        vector_name: str,
        collection_name: str,
        limit: int = 10,
        query_filter: Filter | None = None,
        
    ) -> list[T]:
        """Search using a dense vector.

        Args:
            query_vector: The query vector
            vector_name: Name of the vector field to search
            limit: Maximum number of results
            query_filter: Optional filter to apply

        Returns:
            List of domain objects
        """
        response = await self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [self._hit_to_item(hit) for hit in response.points]

    async def search_sparse(
        self,
        sparse_vector: dict[int, float],
        vector_name: str,
        collection_name: str,
        limit: int = 10,
        query_filter: Filter | None = None,
    ) -> list[T]:
        """Search using a sparse vector.

        Args:
            sparse_vector: The sparse vector as {index: value}
            vector_name: Name of the sparse vector field
            limit: Maximum number of results
            query_filter: Optional filter to apply

        Returns:
            List of domain objects
        """
        response = await self.client.query_points(
            collection_name=collection_name,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            ),
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [self._hit_to_item(hit) for hit in response.points]

    async def search_hybrid(
        self,
        dense_vector: list[float],
        dense_vector_name: str,
        sparse_vector: dict[int, float],
        sparse_vector_name: str,
        collection_name: str,
        limit: int = 10,
        query_filter: Filter | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[T]:
        """Hybrid search combining dense and sparse vectors using RRF fusion.

        Args:
            dense_vector: The dense query vector
            dense_vector_name: Name of the dense vector field
            sparse_vector: The sparse vector as {index: value}
            sparse_vector_name: Name of the sparse vector field
            limit: Maximum number of results
            query_filter: Optional filter to apply
            dense_weight: Weight for dense search results
            sparse_weight: Weight for sparse search results

        Returns:
            List of domain objects
        """
        prefetch_queries = [
            Prefetch(
                query=dense_vector,
                using=dense_vector_name,
                filter=query_filter,
            ),
            Prefetch(
                query=SparseVector(
                    indices=list(sparse_vector.keys()),
                    values=list(sparse_vector.values()),
                ),
                using=sparse_vector_name,
                filter=query_filter,
            ),
        ]

        # Use RRF fusion to combine results
        response = await self.client.query_points(
            collection_name=collection_name,
            query=RrfQuery(rrf=Rrf(weights=[dense_weight, sparse_weight])),
            prefetch=prefetch_queries,
            limit=limit,
            with_payload=True,
        )

        return [self._hit_to_item(hit) for hit in response.points]

    @staticmethod
    @abstractmethod
    def _hit_to_item(hit: ScoredPoint) -> T:
        """Convert a Qdrant search hit to a domain object.

        Args:
            hit: The Qdrant search result

        Returns:
            Domain object
        """
        ...