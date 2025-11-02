from typing import Optional
from core.clients.base import BaseServiceClient, ClientConfig, ClientError
from prefect_agent.service_autoshot.schema import AutoShotRequest, AutoShotResponse

class AutoshotCLientError(ClientError):
    pass


class AutoshotClient(BaseServiceClient[AutoShotRequest, AutoShotResponse]):
    @property
    def service_name(self) -> str:
        # Must match the service name registered in Consul
        # See: prefect_agent/service_autoshot/.env -> SERVICE_NAME=service-autoshot
        return "service-autoshot"
    
    @property
    def inference_endpoint(self) -> str:
        return '/autoshot/infer'
    
    @property
    def load_endpoint(self) -> str:
        return '/autoshot/load'
    
    @property
    def unload_endpoint(self) -> str:
        return '/autoshot/unload'

    @property
    def models_endpoint(self) -> str:
        return '/autoshot/models'

    
    @property
    def status_endpoint(self) -> str:
        return  '/autoshot/status'
    
  
    

        
