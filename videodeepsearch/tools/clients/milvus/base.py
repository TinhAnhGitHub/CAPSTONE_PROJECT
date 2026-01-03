from abc import abstractmethod, ABC
from pymilvus import AsyncMilvusClient
from typing import Iterable, TypeVar, Generic
from pydantic import BaseModel
from loguru import logger

OutputT = TypeVar('OutputT', bound=BaseModel)
class BaseMilvusClient(ABC, Generic[OutputT]):
    """
    The abstract base client
    """
    def __init__(self, uri: str, collection: str):
        self.uri = uri
        self.collection = collection
        self.client: AsyncMilvusClient | None = None

    async def connect(self):
        logger.info(f"Connection to {self.collection=}")
        self.client = AsyncMilvusClient(uri=self.uri)
        await self.client.load_collection(self.collection)
        logger.info("Connected")
        return self

    async def close(self):
        if self.client:
            await self.client.close()
        self.client = None

    @staticmethod
    def _flatten_hits(search_result) -> Iterable:
        if isinstance(search_result, list) and search_result and isinstance(search_result[0], list):
            for hits in search_result:
                for hit in hits:
                    yield hit
        else:
            for hit in search_result:
                yield hit
    
    @staticmethod
    @abstractmethod
    def _hit_to_item(hit) -> OutputT:
        raise NotImplementedError("Must implement the _hit_to_item functionality")

    def _from_hit_to_response(self, hits: list[list[dict]]) -> list[OutputT]:
        flattened = self._flatten_hits(hits)
        return [self._hit_to_item(hit) for hit in flattened]