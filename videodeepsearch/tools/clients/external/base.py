from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, Literal, cast 
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


class ClientError(Exception):
    """Base exception for client errors"""
    pass

class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_name: str
    device: Literal["cpu", "cuda"] = "cuda"

class UnloadModelRequest(BaseModel):
    cleanup_memory: bool = True


class TextEmbeddingSettings(BaseModel):
    model_name: str
    device: Literal['cuda', 'cpu'] 
    batch_size: int


class ImageEmbeddingSettings(BaseModel):
    model_name: str
    device: Literal['cuda', 'cpu'] 
    batch_size: int


class BaseExternalClient(ABC):
    def __init__(
        self,
        base_url: str,
    ):
        self.http_client: httpx.AsyncClient | None = None
        self.base_url = base_url
    
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
    

    async def connect(self) -> None:
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(1800),
            follow_redirects=True
        )
    
    async def close(self) -> None:
        if self.http_client:
            await self.http_client.aclose()
    
    
    async def make_request(
        self,
        method: str,
        endpoint:str,
        request_data: BaseModel | None = None,
        **kwargs
    ):
        url = urljoin(self.base_url, endpoint)
        request_kwargs = {**kwargs}
        if request_data:
            request_kwargs['json'] = request_data.model_dump(mode='json')
        
        response = await self.http_client.request(method, url, **request_kwargs) #type:ignore
        response.raise_for_status()
        response_data = response.json()
        return response_data

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
        
        url = urljoin(self.base_url, self.unload_endpoint) #type:ignore
        
        response = await self.http_client.post(
            url,
            json=request.model_dump(mode='json')
        )
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"{self.service_name}_model_unloaded")
        
        return result
    



class ImageEmbeddingClient(BaseExternalClient):
    @property
    def service_name(self) -> str:
        return "service-image-embedding"
    
    @property
    def inference_endpoint(self) -> str:
        return '/image-embedding/infer'

    @property
    def load_endpoint(self) -> str:
        return '/image-embedding/load'

    @property
    def unload_endpoint(self) -> str:
        return '/image-embedding/unload'

    @property
    def models_endpoint(self) -> str:
        return '/image-embedding/models'

    @property
    def status_endpoint(self) -> str:
        return  '/image-embedding/status'
    

class TextEmbeddingClient(BaseExternalClient):
    @property
    def service_name(self) -> str:
        return "service-text-embedding"
    
    @property
    def inference_endpoint(self) -> str:
        return '/text-embedding/infer'

    @property
    def load_endpoint(self) -> str:
        return '/text-embedding/load'

    @property
    def unload_endpoint(self) -> str:
        return '/text-embedding/unload'

    @property
    def models_endpoint(self) -> str:
        return '/text-embedding/models'
    
    @property
    def status_endpoint(self) -> str:
        return  '/text-embedding/status'