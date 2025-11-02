import asyncio
from typing import Optional, Any
from dataclasses import dataclass


from consul.aio import Consul
from loguru import logger

@dataclass
class ServiceInfo:
    service_id: str
    service_name: str
    address: str
    port: int
    tags: list[str]
    meta: dict[str, str]
    health_status: str = "unknown"

class ConsulServiceRegistry:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8500,
        datacenter: str = "dc1",
        check_interval: str = "10s",
        check_timeout: str = "5s",
    ):
        self.consul = Consul(host=host, port=port)
        self.datacenter = datacenter
        self.check_interval = check_interval
        self.check_timeout = check_timeout
    
    async def register_service(
        self,
        service_name: str,
        address: str,
        port: int,
        service_id: str | None = None,
        tags: list[str] | None = None,
        health_check_url: str | None = None,
        health_check_tcp: str | None= None
    ):
        service_id = service_id or f"{service_name}-{address}-{port}"
        check: dict[str, Any] = {}

        if health_check_url:
            check = {
                "http": health_check_url,
                "interval": self.check_interval,
                "timeout": self.check_timeout,
                "DeregisterCriticalServiceAfter": "30s",
            }
        elif health_check_tcp:
            check = {
                "tcp": health_check_tcp,
                "interval": self.check_interval,
                "timeout": self.check_timeout,
                "DeregisterCriticalServiceAfter": "30s",
            }
        
        await self.consul.agent.service.register(
            name=service_name,
            service_id=service_id,
            address=address,
            port=port,
            tags=tags or [],
            check=check or None
        )


    async def deregister_service(self, service_id: str) -> None:
        await self.consul.agent.service.deregister(service_id)
        logger.info(f"Deregistered service {service_id}")
    
    async def discover_service(self, service_name: str) -> list[ServiceInfo]:
        index, nodes = await self.consul.catalog.service(service_name, dc=self.datacenter)
        services = []
        for node in nodes:
            health_status = await self._get_health_status(node["ServiceID"])
            services.append(
                ServiceInfo(
                    service_id=node["ServiceID"],
                    service_name=node["ServiceName"],
                    address=node["ServiceAddress"] or node["Address"],
                    port=node["ServicePort"],
                    tags=node.get("ServiceTags", []),
                    meta=node.get("ServiceMeta", {}),
                    health_status=health_status,
                )
            )
        return services

    async def _get_health_status(self, service_id: str) -> str:
        _, checks = await self.consul.health.checks(service_id)
        statuses = [c["Status"] for c in checks]
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif "passing" in statuses:
            return "passing"
        return "unknown"

    async def get_healthy_service(self, service_name: str) -> Optional[ServiceInfo]:
        services = await self.discover_service(service_name)
        healthy = [s for s in services]
        return healthy[0] if healthy else None


    async def set_key_value(self, key: str, value: str) -> None:
        await self.consul.kv.put(key, value.encode())
        logger.info(f"Set KV: {key}")

    async def get_key_value(self, key: str) -> Optional[str]:
        index, data = await self.consul.kv.get(key)
        if not data or not data.get("Value"):
            return None
        return data["Value"].decode("utf-8")
    
    async def close(self):
        pass