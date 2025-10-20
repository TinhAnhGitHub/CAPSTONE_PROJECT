from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar, Type, Literal, ClassVar
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from core.pipeline.service_registry import ConsulServiceRegistry
from prefect_agent.shared.schema import ModelInfo, LoadModelRequest, UnloadModelRequest
from pymilvus import (
    AsyncMilvusClient,
    DataType,
    FieldSchema,
    CollectionSchema,
)


class ClientError(Exception):
    """Base exception for client errors"""
    pass
class ServiceUnavailableError(ClientError):
    """raise when the service is not health"""
    pass

class ClientConfig(BaseModel):
    timeout_seconds: float 
    max_retries: int
    retry_min_wait: float 
    retry_max_wait: float
    consul_host: str 
    consul_port: int
    

class MilvusClientError(Exception):
    pass

class MilvusCollectionConfig(BaseModel):
    collection_name: str
    dimension: int
    metric_type: Literal['L2', 'COSINE', 'IP'] = 'COSINE'
    index_type: Literal['FLAT', 'IVF_FLAT', 'HNSW', 'AUTOINDEX'] = 'AUTOINDEX'
    description: str = ""

    nlist: int = 128  
    m: int = 16 
    ef_construction: int = 200

    @property
    def index_params(self)->dict:
        index_params: dict[str, Any] = {
            "field_name": "embedding",
            "index_type": self.index_type,
            "metric_type": self.metric_type,
        }   
        if self.index_type  == 'IVF_FLAT':
            index_params["params"] = {"nlist": self.nlist}
        elif self.index_type == 'HNSW':
            index_params['params'] = {
                "M": self.m,
                "efConstruction": self.ef_construction
            }
        return index_params
    



TRequest = TypeVar('TRequest', bound=BaseModel)
TResponse = TypeVar('TResponse', bound=BaseModel)

class BaseServiceClient(ABC, Generic[TRequest, TResponse]):
    """
    Base client for microservices with Consul discovery, retries, and timeouts.

    Features:
    - Service discovery via Consul
    - Automatic  retry with exp backoff
    - configurable timeouts
    """



    def __init__(
        self,
        config: ClientConfig
    ):
        self.config = config
        self.http_client: httpx.AsyncClient | None = None
        self.consul: ConsulServiceRegistry | None = None

    @property
    @abstractmethod
    def service_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def inference_endpoint(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def load_endpoint(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def unload_endpoint(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def models_endpoint(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def status_endpoint(self) -> str:
        raise NotImplementedError


    async def __aenter__(self) -> BaseServiceClient[TRequest, TResponse]:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    
    async def connect(self) -> None:
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_seconds),
            follow_redirects=True
        )
        
        self.consul = ConsulServiceRegistry(
            host=self.config.consul_host,
            port=self.config.consul_port,
        )
        logger.info(
            f"{self.service_name} client connected",
            consul_host=self.config.consul_host,
            consul_port=self.config.consul_port,
        )
    

    async def close(self) -> None:
        if self.http_client:
            await self.http_client.aclose()
        logger.info(f"{self.service_name} client closed")
    
    async def get_service_url(self) -> str | None:
        if self.consul:
            try:
                service_info = await self.consul.get_healthy_service(self.service_name)
                if service_info:
                    url = f"http://{service_info.address}:{service_info.port}"
                    logger.debug(
                        f"{self.service_name}_service_discovered",
                        url=url,
                        service_id=service_info.service_id,
                    )
                    return url

                logger.warning(f"No health service or not found: {self.service_name}")
            
            except Exception as e:
                logger.error(
                    f"{self.service_name} consul discovery failed",
                    error=str(e)
                )
            
                raise ServiceUnavailableError(
                    f"No healthy {self.service_name} service available and no fallback configured"
                )

    async def make_request(
        self,
        method: str,
        endpoint:str,
        request_data: BaseModel | None = None,
        **kwargs
    ):
        """
        Make the request with retry logic and response validation
        """

        if self.http_client is None:
            raise ClientError("Client not connected. Connect client bro")

        async def _attempt_request():
            base_url = await self.get_service_url()

            url = urljoin(base_url, endpoint) #type:ignore

            print(f"URL service: {url=}")
            print(f"{self.service_name=}")
            
            request_kwargs = {**kwargs}
            if request_data:
                request_kwargs['json'] = request_data.model_dump(mode='json')

            logger.debug(
                f"{self.service_name} request attempt",
                method=method,
                url=url
            )
            
            print(f"{url=}")
            response = await self.http_client.request(method, url, **request_kwargs) #type:ignore
            response.raise_for_status()
            print(f"Response from make request: {response=}")

            response_data = response.json()
            return response_data
    
        try:
            async for attemp in AsyncRetrying(
                stop=stop_after_attempt(self.config.max_retries),
                wait=wait_exponential(
                    min=self.config.retry_min_wait,
                    max=self.config.retry_max_wait
                ),
                retry = retry_if_exception_type(Exception),
                reraise=True
            ):
                with attemp:
                    return await _attempt_request()
        
        except RetryError as e:
            logger.exception(
                f"{self.service_name} request failed retries",
                error=str(e),
                endpoint=endpoint
            )

            raise ClientError(
                f"Request to {self.service_name=} failed after {self.config.max_retries=}"
            )
        except Exception as e:
            logger.exception(
                f"{self.service_name=} has some unexpected error",
                error=str(e)
            )
            raise ClientError(f"Unexpected Error: {e}") from e
        

    async def invoke(self, request: TRequest) -> TResponse:
        response = await self.make_request(
            method="POST",
            endpoint=self.inference_endpoint,
            request_data=request,
        )
        
        if response is None:
            raise ClientError(f"Received empty response from {self.service_name}")
        
        return response
    

    async def load_model(
        self,
        model_name: str,
        device: Literal['cuda','cpu'] = "cuda"
    ):
        request = LoadModelRequest(model_name=model_name, device=device)
        
        response = await self.make_request(
            method="POST",
            endpoint=self.load_endpoint,
            request_data=request,
        )
        
        if response is None:
            raise ClientError(f"Failed to load model {model_name}")
        
        logger.info(
            f"{self.service_name}_model_loaded",
            model_name=model_name,
            device=device
        )
        
        return response
    
    async def unload_model(self, cleanup_memory: bool = True) -> dict[str, str]:
        request = UnloadModelRequest(cleanup_memory=cleanup_memory)
        
        if self.http_client is None:
            raise ClientError("Client not connected")
        
        base_url = await self.get_service_url()
        url = urljoin(base_url, self.unload_endpoint) #type:ignore
        
        response = await self.http_client.post(
            url,
            json=request.model_dump(mode='json')
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"{self.service_name}_model_unloaded")
        
        return result
    
    async def list_models(self):
        response = await self.make_request(
            method="GET",
            endpoint=self.models_endpoint,
            request_data=None,
        )
        
        if response is None:
            raise ClientError("Failed to list models")
        
        return response
    
    async def get_status(self) -> dict[str, Any]:
        if self.http_client is None:
            raise ClientError("Client not connected")
        
        base_url = await self.get_service_url()
        url = urljoin(base_url, self.status_endpoint)#type:ignore

        print(f"{url=}")
        
        
        response = await self.http_client.get(url)
        response.raise_for_status()
        print(f"{response=}")
    
        return response.json()
    

class BaseMilvusClient(ABC):

    def __init__(
            self, 
            config_collection: MilvusCollectionConfig, 
            host: str, 
            port: int, 
            user: str = "", 
            password: str = "", 
            db_name: str = 'default', 
            timeout: float=30.0
        ):

        self.host = host
        self.port = port
        self.user = user
        self.config = config_collection
        self.password = password
        self.db_name = db_name
        self.timeout = timeout
        self._client: AsyncMilvusClient | None = None
    
    async def connect(self) -> None:
        try:
            
            uri = f"http://{self.host}:{self.port}"
            print(uri)
            self._client = AsyncMilvusClient(
                uri=uri,
                user=self.user,
                password=self.password,
                db_name=self.db_name,
                timeout=self.timeout
            )
            logger.info(
                "milvus_client_connected",
                host=self.host,
                port=self.port,
                db=self.db_name
            )
        except Exception as e:
            logger.exception("milvus_connection_failed", error=str(e))
            raise MilvusClientError(f"Failed to connect to Milvus: {e}") from e
    
    async def close(self) -> None:
        if self._client:
            await self._client.close()
            logger.info("milvus_client_closed")
    
    async def __aenter__(self) -> BaseMilvusClient:
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    
    @property
    def client(self) -> AsyncMilvusClient:
        if self._client is None:
            raise MilvusClientError("Client not connected. Call connect() first.")
        return self._client
    
    @abstractmethod
    def get_schema(self) -> CollectionSchema:
        """Define the collection schema"""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def embedding_field(self) -> str:
        raise NotImplementedError


    async def create_collection_if_not_exists(self) -> None:
        """
        Create both collections and index name
        """

        has_collection = await self.client.has_collection(self.config.collection_name)
        if has_collection:
            logger.info(
                'Milvus collection exists'
            )
            return
            
        schema = self.get_schema()
        await self.client.create_collection(
            collection_name=self.config.collection_name,
            schema=schema,
            timeout=10
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=self.embedding_field,
            index_type=self.config.index_type,
            metric_type=self.config.metric_type,
            params=self.config.index_params
        )
        await self.client.create_index(index_params=index_params, collection_name=self.config.collection_name)
       
    async def insert_vectors(self, data: list[dict[str, Any]]) -> list[str]:
        config = self.config
        try:
            
            result = await self.client.insert(
                collection_name=self.config.collection_name,
                data=data
            )
            logger.info("Milvus vector inserted")
            return result.get("ids", [])
        except Exception as e:
            logger.exception(
                "Milvus insertion failed",
                collection=self.config.collection_name,
                error=str(e)
            )
            raise MilvusClientError(f"Failed to insert vectors: {e}") from e
    

    async def get_collection_stats(self) -> dict[str, Any]:
        try:
            stats = await self.client.get_collection_stats(self.config.collection_name)
            return stats
        except Exception as e:
            logger.exception(
                "milvus_stats_failed",
                collection=self.config.collection_name,
                error=str(e)
            )
            raise MilvusClientError(f"Failed to get stats: {e}") from e
        
    
    async def record_exists(self, filter_expr: str) -> bool:
        try:
            await self.ensure_collection_loaded()
            result = await self.client.query(
                collection_name=self.config.collection_name,
                filter=filter_expr, 
                output_fields=['id'],
                limit=1
            )

            return len(result) > 0
        except Exception as e:
            logger.exception(
                "Milvus existence check failed",
                collection=self.config.collection_name,
                error=str(e)
            )
            raise MilvusClientError(f"Failed to check existence: {e}") from e
        

    async def ensure_collection_loaded(self):
        try:
            await self.client.load_collection(self.config.collection_name)
        except Exception as e:
            raise MilvusClientError(f"Failed to load collection: {e}") from e

    
    async def delete_by_filter(self, filter_expr: str) -> int:
        try:
            await self.ensure_collection_loaded()
            result = await self.client.delete(
                collection_name=self.config.collection_name,
                filter=filter_expr
            )
            return result.get("delete_count", 0)
        except Exception as e:
            logger.exception(
                "Milvus deletion failed",
                collection=self.config.collection_name,
                error=str(e)
            )
            raise MilvusClientError(f"Failed to delete records: {e}") from e

    async def has_user_collection(self) -> bool:
        """Check if the user-scoped collection exists, swallowing errors to bool."""
        try:
            return await self.client.has_collection(self.config.collection_name)
        except Exception:
            return False
