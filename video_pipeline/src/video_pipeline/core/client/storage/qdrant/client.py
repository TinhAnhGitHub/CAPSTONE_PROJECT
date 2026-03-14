from typing import Any
from grpc import StatusCode
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    VectorParams,
    PointStruct,
    CollectionInfo,
    SparseVectorParams,
    SparseIndexParams,
)
from loguru import logger

from .config import QdrantConfig, QdrantIndexConfig
from .exception import QdrantClientError

class QdrantStorageClient:
    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self.port = config.port
        self.host = config.host
        self.timeout = config.timeout
        self.use_grpc = config.use_grpc
        self.prefer_grpc = config.prefer_grpc
        self._client: AsyncQdrantClient | None = None
        self.collection_name = config.collection_name

        self.embedding_field_name: list[str] | None = None
        self.index_configs: list[QdrantIndexConfig] | None = None

    async def connect(self) -> None:
        try:
            try:
                import fastembed  # noqa: F401
            except ImportError:
                pass  

            self._client = AsyncQdrantClient(
                host=self.host,
                port=self.port,
                timeout=self.timeout,
                grpc_port=6334,
                prefer_grpc=self.prefer_grpc,
            )

            self._client.set_sparse_model(embedding_model_name="prithivida/Splade_PP_en_v1")
            logger.info("qdrant_client_connected", host=self.host, port=self.port)
        except Exception as e:
            logger.exception("qdrant_connection_failed", error=str(e))
            raise QdrantClientError(f"Failed to connect to Qdrant: {e}") from e

    async def close(self):
        if self._client:
            await self._client.close()
            logger.info("qdrant_client_closed")

    async def __aenter__(self) -> "QdrantStorageClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise QdrantClientError("Client not connected. Call connect() first.")
        return self._client

    def build_vector_config(
        self,
        index_configs_list: list[QdrantIndexConfig],
        embedding_field_name_list: list[str],
    ) -> tuple[dict[str, VectorParams], dict[str, SparseVectorParams]]:
        
        assert len(index_configs_list) == len(
            embedding_field_name_list
        ), "Index configs and field name length must be the same."

        vectors_config = {}
        sparse_vectors_config = {}

        for field_name, embed_config in zip(embedding_field_name_list, index_configs_list):
            if not embed_config.is_sparse:
                vectors_config[field_name] = VectorParams(
                    size=embed_config.vector_size,
                    distance=embed_config.distance,
                    on_disk=embed_config.on_disk,
                    hnsw_config=embed_config.hnsw_config,
                    quantization_config=embed_config.quantization_config,
                )
            else:
                sparse_vectors_config[field_name] = SparseVectorParams(
                    index=SparseIndexParams(on_disk=embed_config.on_disk)
                )
                
        return vectors_config, sparse_vectors_config

    async def create_collection_if_not_exists(
        self,
        index_configs_list: list[QdrantIndexConfig],
        embedding_field_name_list: list[str],
    ):
        self.embedding_field_name = embedding_field_name_list
        self.index_configs = index_configs_list

        try:
            exists = await self.has_user_collection()
            if exists:
                logger.info(f"Qdrant collection exist: {self.collection_name=}")
                return

            vectors_config, sparse_vectors_config = self.build_vector_config(
                index_configs_list=index_configs_list,
                embedding_field_name_list=embedding_field_name_list,
            )

            try:
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=vectors_config or None,
                    sparse_vectors_config=sparse_vectors_config or None
                )
                logger.info(
                    f"Qdrant collection name created: {self.collection_name=}",
                )
            except Exception as e:
                # Handle race condition: another task may have created it first
                if self._is_already_exists_error(e):
                    logger.info(f"Qdrant collection already exists (race): {self.collection_name=}")
                else:
                    raise
        except Exception as e:
            logger.exception(f"Failed to create collection: {e}")
            raise QdrantClientError(f"Failed to create collection: {e}") from e

    def _is_already_exists_error(self, error: Exception) -> bool:
        """Check if the error is 'already exists' from gRPC or REST."""
        error_str = str(error).lower()
        if "already_exists" in error_str or "already exists" in error_str:
            return True
        # Check gRPC status code
        if hasattr(error, "_state") and hasattr(error._state, "code"):
            return error._state.code == StatusCode.ALREADY_EXISTS
        return False

    async def insert_vectors(self, data: list[dict[str, Any]]) -> list[str]:
        """
        Data example Structure
        [
            {
                "id": "1", 
                "text_dense": [0.1, 0.2, 0.3, ...], 
                "text_sparse": Document("This is my raw text document. Qdrant will encode it."), 
                "meta_title": "Hello World"
            },
            {
                "id": "2", 
                "text_dense":[0.1, 0.2, 0.3, ...],
                "text_sparse": Document("Another document about AI and databases."), 
                "meta_title": "Another Document"
            }
        ]
        """
        assert (
            self.embedding_field_name
        ), "Embedding name list is empty. Please call create_collection_if_not_exists first!"
        try:
            points = []
            for item in data:
                point_id = item.get("id")
                if not point_id:
                    raise QdrantClientError("Each data item must have an 'id' field")

                vectors = {}
                payload = {}
                for key, value in item.items():
                    if key == "id":
                        continue
                    elif key in self.embedding_field_name:
                        vectors[key] = value
                    else:
                        payload[key] = value

                points.append(PointStruct(id=point_id, vector=vectors, payload=payload))
            
            await self.client.upsert(
                collection_name=self.collection_name, points=points
            )
            logger.info(f"Qdrant vector inserted: {len(points)=}")
            return [str(p.id) for p in points]

        except Exception as e:
            logger.exception(
                f"Qdrant insertion failed: {self.collection_name=} with {e=}"
            )
            raise QdrantClientError(f"Failed to insert vectors: {e}") from e

    async def get_collection_stats(self) -> dict[str, Any]:
        try:
            info: CollectionInfo = await self.client.get_collection(
                self.collection_name
            )
            return {
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status,
            }
        except Exception as e:
            logger.exception("qdrant_stats_failed")
            raise QdrantClientError(f"Failed to get stats: {e}") from e

    async def has_user_collection(self) -> bool:
        """Check if the collection exists, swallowing errors to bool."""
        try:
            return await self.client.collection_exists(self.collection_name)
        except Exception:
            return False