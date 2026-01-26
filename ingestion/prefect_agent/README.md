# Multimodal AI Microservices Platform

A production-ready microservices architecture for multimodal AI inference, including Automatic Speech Recognition (ASR), Shot Boundary Detection, Image/Text Embedding Generation, and Large Language Model (LLM) inference.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Shared Architecture](#shared-architecture)
- [Project Structure](#project-structure)
- [Services](#services)
- [Shared Components](#shared-components)
- [Technology Stack](#technology-stack)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Data Flow](#data-flow)
- [Deployment](#deployment)
- [Monitoring](#monitoring)

---

## Architecture Overview

This platform implements a **microservices architecture** where each AI capability runs as an independent, containerized service. Services communicate via REST APIs and use Consul for service discovery and MinIO for object storage.

### Design Principles

- **Modularity**: Each service is self-contained with its own dependencies, configuration, and model handlers
- **Scalability**: Stateless services can be scaled horizontally based on demand
- **Resource Efficiency**: Models are loaded/unloaded dynamically to manage GPU memory
- **Observability**: Integrated Prometheus metrics and structured logging
- **Service Discovery**: Consul integration for dynamic service registration and discovery

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestration Layer                         │
│                    (Prefect Agent ready)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service Discovery                           │
│                         (Consul)                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  AutoShot     │   │     ASR       │   │  LLM Service  │
│  (Port 8001)  │   │   (Port 8002) │   │   (Port 8004) │
└───────────────┘   └───────────────┘   └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Object Storage                               │
│                      (MinIO/S3)                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GPU Inference Layer                           │
│  TransNetV2  │  Chunkformer  │  Gemini/OpenRouter  │  ...       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
prefect_agent/
├── README.md                          # This file
├── script_code.sh                     # Utility shell script
├── output.txt                         # Output artifact
│
├── shared/                            # SHARED CORE MODULES
│   ├── config.py                      # ServiceConfig, LogConfig
│   ├── logger.py                      # Loguru logging setup
│   ├── metrics.py                     # Prometheus metrics
│   ├── monitor.py                     # System monitoring
│   ├── registry.py                    # BaseModelHandler & registry
│   ├── service.py                     # BaseService abstract class
│   ├── service_registry.py            # Consul integration
│   ├── storage.py                     # MinIO/S3 client
│   ├── util.py                        # S3 URL parsing, async utilities
│   ├── retry.py                       # Retry decorators
│   └── schema.py                      # Shared Pydantic schemas
│
├── service_autoshot/                  # Shot Boundary Detection
├── service_asr/                       # Automatic Speech Recognition
├── service_image_embedding/           # Image Embedding Generation
├── service_text_embedding/            # Text Embedding Generation
├── service_llm/                       # LLM Inference
│
├── weight/                            # Pre-trained model weights
│   ├── transnetv2-pytorch-weights.pth
│   ├── beit3/
│   └── chunkformer-large-vie/
│
└── .spec-workflow/                    # Documentation templates
```

---

## Services

### 1. AutoShot Service (Port 8001)

**Purpose**: Video shot boundary detection using TransNetV2

**Endpoint**: `/autoshot`

**Model**: TransNetV2 PyTorch implementation

**Capabilities**:
- Detect scene changes in video content
- Return timestamped shot boundaries
- Process video frames for scene transitions

**Key Files**:
- `service_autoshot/model/autoshot.py` - AutoShot wrapper
- `service_autoshot/model/transnet_v2.py` - TransNetV2 implementation
- `service_autoshot/model/registry.py` - Model handler

### 2. ASR Service (Port 8002)

**Purpose**: Automatic Speech Recognition with timestamps

**Endpoint**: `/asr`

**Model**: Chunkformer RNN-T ASR

**Capabilities**:
- Transcribe audio to text
- Generate word-level timestamps
- Support long audio via chunked processing

**Key Files**:
- `service_asr/model/asr_core.py` - ASRProcessor with Chunkformer
- `service_asr/model/audio_extraction.py` - Audio preprocessing
- `service_asr/model/decode_batch.py` - Batch decoding

### 3. LLM Service (Port 8004)

**Purpose**: Large Language Model inference

**Endpoint**: `/llm`

**Models**:
- Google Gemini (via API)
- OpenRouter models (Llama, etc.)

**Capabilities**:
- Text generation
- Token counting
- Multi-model support

**Key Files**:
- `service_llm/model/gemini.py` - Gemini API handler
- `service_llm/model/openrouter.py` - OpenRouter handler
- `service_llm/model/qwen.py` - Qwen model handler

### 4. Image Embedding Service (Port 8000)

**Purpose**: Generate visual feature vectors

**Endpoint**: `/image-embedding`

**Models**:
- OpenCLIP - CLIP-based embeddings
- BEiT3 - Vision-language model

**Capabilities**:
- Generate image embeddings
- Support for multiple embedding models

**Key Files**:
- `service_image_embedding/model/open_clip/` - OpenCLIP implementation
- `service_image_embedding/model/beit3/` - BEiT3 implementation

### 5. Text Embedding Service (Port 8003)

**Purpose**: Generate semantic text vectors

**Endpoint**: `/text-embedding`

**Models**:
- Sentence-BERT (sentence-transformers)
- mmBERT (multimodal BERT)

**Capabilities**:
- Generate text embeddings
- Semantic similarity calculations

**Key Files**:
- `service_text_embedding/model/sentence_transformers.py`
- `service_text_embedding/model/mmbert.py`

---

## Shared Architecture

This section details the shared architecture pattern that all services inherit from, ensuring consistency, maintainability, and code reuse across the platform.

### Shared Architecture Philosophy

The platform uses a **layered inheritance pattern** where:

1. **Shared Core Layer** (`shared/`): Contains abstract base classes and utilities
2. **Service Layer** (`service_*/core/`): Inherits from shared base classes
3. **Model Layer** (`service_*/model/`): Implements model-specific handlers

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Service Layer                                 │
│   AutoshotService / ASRService / LLMService / etc.                  │
│   (service_xxx/core/service.py)                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ▲ inherits
                                │
┌─────────────────────────────────────────────────────────────────────┐
│                       Shared Core Layer                              │
│   BaseService<InputT, OutputT>                                       │
│   (shared/service.py)                                                │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │              Shared Utilities & Infrastructure               │   │
│   │  BaseModelHandler  │  ConsulServiceRegistry  │  StorageClient│   │
│   │  ServiceMetrics    │  setup_service_logger  │  config       │   │
│   └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

### BaseService Architecture

`BaseService` is the core abstract class that all services inherit from. It provides:

#### Class Definition

```python
# shared/service.py

class BaseService(Generic[InputT, OutputT], ABC):
    """Abstract base class providing common service functionality."""

    def __init__(self, service_config: ServiceConfig, log_config: LogConfig) -> None:
        # Initialization from shared components
        setup_service_logger(...)      # Loguru logging setup
        self.metrics = ServiceMetrics(...)  # Prometheus metrics
        self.loaded_model: Optional[BaseModelHandler[InputT, OutputT]] = None
        self.loaded_model_info: Optional[ModelInfo] = None
```

#### Key Responsibilities

| Method | Responsibility | Returns |
|--------|----------------|---------|
| `load_model()` | Model lifecycle management | `ModelInfo` |
| `unload_model()` | Resource cleanup | `None` |
| `infer()` | Execute model inference | `OutputT` |
| `get_system_status()` | Resource monitoring | `dict` |
| `update_system_metrics()` | Update metrics | `None` |
| `get_available_models()` | List registered models | `list[str]` |

#### Generic Type Parameters

```python
BaseService[InputT, OutputT]
```

- **`InputT`**: Pydantic model for request validation (bound to `BaseModel`)
- **`OutputT`**: Pydantic model for response formatting (bound to `BaseModel`)

---

### Service Inheritance Pattern

Each service implements a thin wrapper around `BaseService`:

#### Example: AutoshotService

```python
# service_autoshot/core/service.py

from shared.service import BaseService
from service_autoshot.core.config import AutoshotConfig
from service_autoshot.schema import AutoShotRequest, AutoShotResponse

class AutoshotService(BaseService[AutoShotRequest, AutoShotResponse]):
    """Thin wrapper around shared BaseService for autoshot models."""

    def __init__(self, service_config: AutoshotConfig, log_config: LogConfig) -> None:
        super().__init__(service_config=service_config, log_config=log_config)

    def get_available_models(self) -> list[str]:
        # Filter available models for this service
        registered = super().get_available_models()
        return [model for model in registered if model == "autoshot"]
```

#### Example: ASRService

```python
# service_asr/core/service.py

from shared.service import BaseService
from service_asr.core.config import ASRServiceConfig
from service_asr.core.schema import ASRInferenceRequest, ASRInferenceResponse

class ASRService(BaseService[ASRInferenceRequest, ASRInferenceResponse]):
    """Thin wrapper around shared BaseService for ASR models."""

    def __init__(self, service_config: ASRServiceConfig, log_config: LogConfig) -> None:
        super().__init__(service_config=service_config, log_config=log_config)

    def get_available_models(self) -> list[str]:
        registered = super().get_available_models()
        return [model for model in registered if model == "chunkformer"]
```

#### Example: LLMService

```python
# service_llm/core/service.py

from shared.service import BaseService
from service_llm.core.config import LLMServiceConfig
from service_llm.schema import LLMRequest, LLMResponse

class LLMService(BaseService[LLMRequest, LLMResponse]):
    """Service wrapper exposing multimodal LLM handlers."""

    def __init__(self, service_config: LLMServiceConfig, log_config: LogConfig) -> None:
        super().__init__(service_config=service_config, log_config=log_config)
        self._model_cache: Dict[str, BaseModelHandler[...]] = {}

    def get_available_models(self) -> list[str]:
        registered = super().get_available_models()
        enabled = []
        if self._service_config.gemini_api_key and "gemini_api" in registered:
            enabled.append("gemini_api")
        if self._service_config.openrouter_api_key and "openrouter_api" in registered:
            enabled.append("openrouter_api")
        return enabled
```

---

### BaseModelHandler Architecture

`BaseModelHandler` defines the interface that all model implementations must follow:

#### Class Definition

```python
# shared/registry.py

class BaseModelHandler(Generic[InputT, OutputT], ABC):
    """Interface each concrete model handler must implement."""

    def __init__(self, model_name: str, config: ServiceConfig) -> None:
        self.model_name = model_name
        self.config = config

    @abstractmethod
    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        """Load model weights into memory."""

    @abstractmethod
    async def unload_model_impl(self) -> None:
        """Release model resources."""

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Expose metadata for the loaded model."""

    @abstractmethod
    async def preprocess_input(self, input_data: InputT) -> Any:
        """Transform request payload into model-ready tensors."""

    @abstractmethod
    async def run_inference(self, preprocessed_data: Any) -> Any:
        """Perform forward pass and return raw outputs."""

    @abstractmethod
    async def postprocess_output(
        self, output_data: Any, original_input_data: InputT
    ) -> OutputT:
        """Convert raw outputs into the service response schema."""
```

#### Processing Pipeline

```
InputT (Request)
     │
     ▼
┌────────────────────────────────────────────┐
│           preprocess_input()                │
│  • Validate input data                     │
│  • Fetch from S3 if needed                 │
│  • Convert to tensors                      │
└────────────────────────────────────────────┘
     │
     ▼
Preprocessed Data (Tensors)
     │
     ▼
┌────────────────────────────────────────────┐
│           run_inference()                  │
│  • Model forward pass                      │
│  • GPU/CPU computation                     │
│  • Return raw predictions                  │
└────────────────────────────────────────────┘
     │
     ▼
Output Data (Raw Predictions)
     │
     ▼
┌────────────────────────────────────────────┐
│          postprocess_output()              │
│  • Format timestamps                       │
│  • Calculate confidence scores             │
│  • Return typed OutputT                    │
└────────────────────────────────────────────┘
     │
     ▼
OutputT (Response)
```

---

### Model Handler Registration Pattern

Model handlers are self-registering via the `@register_model()` decorator:

#### Example: AutoshotModelHandler

```python
# service_autoshot/model/registry.py

from shared.registry import BaseModelHandler, register_model
from shared.schema import ModelInfo
from service_autoshot.core.config import AutoshotConfig
from service_autoshot.model.autoshot import AutoShot
from service_autoshot.schema import AutoShotRequest, AutoShotResponse

@register_model("autoshot")
class AutoshotModelHandler(BaseModelHandler[AutoShotRequest, AutoShotResponse]):
    """Model handler for TransNetV2 shot detection."""

    def __init__(self, model_name: str, config: AutoshotConfig) -> None:
        super().__init__(model_name, config)
        self._model: AutoShot | None = None
        self._device: str | None = None

    async def load_model_impl(self, device: Literal["cpu", "cuda"]) -> None:
        model_path = self._service_config.autoshot_model_path
        self._model = AutoShot(pretrained_path=model_path, device=device)
        self._device = device

    async def unload_model_impl(self) -> None:
        if self._model is not None:
            del self._model
        self._model = None

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self.model_name, model_type="autoshot")

    async def preprocess_input(self, input_data: AutoShotRequest) -> str:
        # Fetch video from S3
        video_path = await fetch_object_from_s3(
            input_data.s3_minio_url, storage=self._client, suffix='.mp4'
        )
        return video_path

    async def run_inference(self, preprocessed_data: str) -> list[list[int]]:
        return self._model.process_video(preprocessed_data)

    async def postprocess_output(
        self, output_data: list[list[int]], original_input_data: AutoShotRequest
    ) -> AutoShotResponse:
        scenes = [tuple(s) for s in output_data]
        return AutoShotResponse(
            scenes=scenes,
            total_scenes=len(scenes),
            status="success",
        )
```

#### Registry Mechanics

```python
# shared/registry.py

MODEL_REGISTRY: Dict[str, Type[BaseModelHandler[Any, Any]]] = {}

def register_model(name: str):
    """Decorator that self-registers model handlers."""
    def decorator(cls: Type[BaseModelHandler[Any, Any]]) -> Type[BaseModelHandler[Any, Any]]:
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY.keys())

def get_model_handler(name: str, config: ServiceConfig) -> BaseModelHandler[Any, Any]:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not registered")
    return MODEL_REGISTRY[name](name, config)
```

---

### Shared Component Usage Flow

#### 1. Service Initialization

```python
# Each service_xxx/main.py follows this pattern

from shared.config import LogConfig
from service_xxx.core.config import XxxConfig
from service_xxx.core.service import XxxService

config = XxxConfig()  # Loads from .env
log_config = LogConfig()  # Loads from .env

service = XxxService(service_config=config, log_config=log_config)
# BaseService.__init__() is called:
#   - setup_service_logger()  # Loguru configuration
#   - ServiceMetrics()        # Prometheus metrics
#   - Registers with Consul   # Service discovery
```

#### 2. Model Loading Flow

```python
# User calls POST /xxx/load with {"model": "model_name", "device": "cuda"}

async def load_model(self, model_name: str, device: Literal["cuda", "cpu"]) -> ModelInfo:
    # 1. Validate model is available
    if model_name not in self.get_available_models():
        raise HTTPException(...)

    # 2. Check GPU availability
    if device == "cuda" and not is_gpu_available():
        if self.service_config.cpu_fallback:
            device = "cpu"
        else:
            raise HTTPException(...)

    # 3. Get handler from registry
    handler = get_model_handler(model_name, self.service_config)

    # 4. Load model via handler
    await handler.load_model_impl(device)

    # 5. Store reference for inference
    self.loaded_model = handler
    self.loaded_model_info = handler.get_model_info()

    # 6. Update system metrics
    self.update_system_metrics()

    return self.loaded_model_info
```

#### 3. Inference Flow

```python
# User calls POST /xxx/infer with request body

async def infer(self, input_data: InputT) -> OutputT:
    if self.loaded_model is None:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        start_time = time.time()

        # Preprocess: Convert request to model input
        preprocessed = await self.loaded_model.preprocess_input(input_data)

        # Inference: Run model
        result = await self.loaded_model.run_inference(preprocessed)

        # Postprocess: Convert output to response
        output = await self.loaded_model.postprocess_output(result, input_data)

        # Metrics: Track request
        duration = time.time() - start_time
        self.metrics.observe_request_duration("infer", duration)
        self.metrics.track_request("infer", "success")

        return output
    except Exception as exc:
        self.metrics.track_error("infer", type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
```

---

### Shared Component Details

#### Configuration Management

```python
# shared/config.py

class ServiceConfig(BaseSettings):
    service_name: str
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int
    cpu_fallback: bool = False

    class Config:
        env_file = ".env"

class LogConfig(BaseSettings):
    log_level: str = "INFO"
    log_format: str = "console"
    log_file: Optional[str] = None
    log_rotation: str = "100 MB"
    log_retention: str = "10 days"
```

#### Logging Setup

```python
# shared/logger.py

def setup_service_logger(
    service_name: str,
    service_version: str,
    log_level: str,
    log_format: str = "console",
    log_file: Optional[str] = None,
    log_rotation: str = "100 MB",
    log_retention: str = "10 days",
) -> None:
    # Configure Loguru with service context
    logger.configure(
        handlers=[
            {
                "sink": sys.stdout,
                "format": log_format,
                "level": log_level,
            },
        ]
    )
    # Add service context to all log records
    logger = logger.bind(service=service_name, version=service_version)
```

#### Metrics Collection

```python
# shared/metrics.py

class ServiceMetrics:
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.requests_total = Counter(...)
        self.request_duration = Histogram(...)
        self.errors_total = Counter(...)
        self.cpu_usage = Gauge(...)
        self.memory_bytes = Gauge(...)
        self.gpu_memory_used = Gauge(...)
        self.gpu_utilization = Gauge(...)

    def track_request(self, endpoint: str, status: str) -> None:
        self.requests_total.labels(endpoint=endpoint, status=status).inc()

    def observe_request_duration(self, endpoint: str, duration: float) -> None:
        self.request_duration.labels(endpoint=endpoint).observe(duration)

    def update_cpu_usage(self, percent: float) -> None:
        self.cpu_usage.set(percent)

    def update_gpu_metrics(self, gpu_id: int, memory_used: int, memory_total: int, utilization: float) -> None:
        self.gpu_memory_used.labels(gpu_id=gpu_id).set(memory_used)
        self.gpu_utilization.labels(gpu_id=gpu_id).set(utilization)
```

#### Storage Client

```python
# shared/storage.py

class StorageClient:
    """MinIO/S3-compatible storage client."""

    def __init__(self, settings: MinioSettings):
        self.client = Minio(
            endpoint=settings.endpoint,
            access_key=settings.access_key,
            secret_key=settings.secret_key,
            secure=settings.secure,
        )
        self.bucket_videos = "videos"
        self.bucket_artifacts = "video-artifacts"

    async def download_file(self, s3_url: str, suffix: str) -> str:
        # Parse S3 URL and download to local temp file
        pass

    async def upload_file(self, local_path: str, s3_key: str, bucket: str) -> str:
        # Upload local file to S3
        pass
```

#### Service Discovery

```python
# shared/service_registry.py

class ConsulServiceRegistry:
    """Service discovery integration with Consul."""

    def __init__(self, host: str, port: int):
        self.consul = Consul(host=host, port=port)

    async def register_service(
        self,
        service_name: str,
        address: str,
        port: int,
        tags: list[str],
        health_check_url: str,
    ) -> None:
        """Register service with Consul."""
        self.consul.agent.service.register(
            name=service_name,
            address=address,
            port=port,
            tags=tags,
            check=AgentCheckCheck(
                http=health_check_url,
                interval="10s",
                timeout="5s",
            ),
        )
```

---

### Service Structure Template

All services follow this consistent structure:

```
service_xxx/
├── main.py                    # FastAPI app entry point
├── core/
│   ├── api.py                # API routes (FastAPI endpoints)
│   ├── config.py             # ServiceConfig subclass
│   ├── service.py            # Service class inheriting BaseService
│   ├── dependencies.py       # Dependency injection
│   ├── lifespan.py           # Startup/shutdown lifecycle
│   └── schema.py             # Request/Response Pydantic models
├── model/
│   ├── registry.py           # Model handlers with @register_model()
│   ├── handler.py            # Concrete model implementation
│   └── utils.py              # Model-specific utilities
├── Dockerfile
├── pyproject.toml
└── .env.example
```

### Summary: How Services Inherit from Shared Components

| Shared Component | Inheritance | Usage by Services |
|------------------|-------------|-------------------|
| `BaseService` | Abstract Base Class | All services inherit this to get model lifecycle, metrics, logging |
| `BaseModelHandler` | Interface | All model handlers implement this interface |
| `ServiceConfig` | Base Settings | All service configs inherit for .env loading |
| `LogConfig` | Base Settings | All services use for logging configuration |
| `ServiceMetrics` | Composition | Injected into BaseService for metrics collection |
| `ConsulServiceRegistry` | Composition | Used in main.py for service registration |
| `StorageClient` | Composition | Used in model handlers for S3 operations |
| `setup_service_logger()` | Function Call | Called in BaseService.__init__() |
| `is_gpu_available()` | Function Call | Called before model loading |
| `get_system_stats()` | Function Call | Called in get_system_status() |

---

## Technology Stack

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.11+ | Primary development |
| **Web Framework** | FastAPI 0.118+ | REST API framework |
| **ASGI Server** | Uvicorn | ASGI implementation |
| **ML Framework** | PyTorch 2.8.0 | Deep learning |
| **GPU Support** | CUDA 12.8, cuDNN 9 | GPU acceleration |
| **Package Manager** | uv | Fast Python packages |

### ML/AI Libraries

| Library | Service | Purpose |
|---------|---------|---------|
| **chunkformer** | ASR | Speech recognition |
| **TransNetV2** | AutoShot | Shot detection |
| **open-clip-torch** | Image Embedding | CLIP embeddings |
| **BEiT3** | Image Embedding | Vision encoder |
| **sentence-transformers** | Text Embedding | Text vectors |
| **google-generativeai** | LLM | Gemini API |
| **llama-index** | LLM | LLM utilities |

### Infrastructure

| Library | Purpose |
|---------|---------|
| **loguru** | Structured logging |
| **prometheus-client** | Metrics |
| **python-consul2** | Service discovery |
| **minio** | S3 storage |
| **tenacity** | Retry logic |
| **pydantic** | Data validation |

---

## Configuration

### Environment Variables Pattern

Each service uses `pydantic-settings` with `.env` files:

```python
class ServiceConfig(BaseSettings):
    service_name: str
    service_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int
    cpu_fallback: bool = False
    
    class Config:
        env_file = ".env"
        env_prefix = ""
```

### Service-Specific Configuration

**LLM Service (.env.example)**:
```
SERVICE_NAME=service-llm
SERVICE_VERSION=0.1.0
HOST=llm
PORT=8004
CPU_FALLBACK=TRUE
GEMINI_API_KEY=<api-key>
GEMINI_MODEL_NAME=gemini-flash-lite-latest
OPENROUTER_API_KEY=<api-key>
OPENROUTER_MODEL_NAME=google/gemini-2.5-flash-lite-preview-09-2025
LOG_LEVEL=DEBUG
```

**ASR Service (.env.example)**:
```
SERVICE_NAME=service-asr
SERVICE_VERSION=0.1.0
HOST=asr
PORT=8002
CPU_FALLBACK=FALSE
model_name=vinai/ASR-chunkformer-large-vie
temp_dir=/tmp/asr
default_chunk_size=300000
default_sample_rate=16000
```

**AutoShot Service (.env.example)**:
```
SERVICE_NAME=service-autoshot
SERVICE_VERSION=0.1.0
HOST=autoshot
PORT=8001
CPU_FALLBACK=FALSE
AUTOSHOT_MODEL_PATH=/app/weight/transnetv2-pytorch-weights.pth
```

**Image Embedding Service (.env.example)**:
```
SERVICE_NAME=service-image-embedding
SERVICE_VERSION=0.1.0
HOST=image-embedding
PORT=8000
CPU_FALLBACK=FALSE
BEIT3_MODEL_CHECKPOINT=/app/weight/beit3/beit3_large_patch16_384_f30k_retrieval.pth
BEIT3_TOKENIZER_CHECKPOINT=/app/weight/beit3/beit3.spm
OPEN_CLIP_MODEL_NAME=ViT-L-14
OPEN_CLIP_PRETRAINED=laion2b_s32b_b82k
```

**Text Embedding Service (.env.example)**:
```
SERVICE_NAME=service-text-embedding
SERVICE_VERSION=0.1.0
HOST=text-embedding
PORT=8003
CPU_FALLBACK=FALSE
SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2
MMBERT_MODEL_NAME=bert-base-multilingual-cased
```

---

## API Documentation

### Common Endpoints

All services implement consistent REST API patterns:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/{service}/load` | Load model |
| POST | `/{service}/unload` | Unload model |
| POST | `/{service}/infer` | Run inference |
| GET | `/{service}/models` | List models |
| GET | `/{service}/status` | System status |

### Service Ports Mapping

| Service | Port | Base Path |
|---------|------|-----------|
| Image Embedding | 8000 | `/image-embedding` |
| AutoShot | 8001 | `/autoshot` |
| ASR | 8002 | `/asr` |
| Text Embedding | 8003 | `/text-embedding` |
| LLM | 8004 | `/llm` |

### Request/Response Schemas

#### ASR Request/Response
```python
class ASRRequest(BaseModel):
    video_path: str  # S3 URL or local path
    language: Optional[str] = None
    return_timestamps: bool = True

class ASRResponse(BaseModel):
    text: str
    timestamps: List[Timestamp]
    confidence: float
```

#### AutoShot Request/Response
```python
class AutoShotRequest(BaseModel):
    video_path: str  # S3 URL or local path
    threshold: Optional[float] = 0.5

class AutoShotResponse(BaseModel):
    shots: List[Shot]
    total_shots: int
    processing_time: float
```

#### LLM Request/Response
```python
class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7

class LLMResponse(BaseModel):
    text: str
    model: str
    usage: TokenUsage
```

---

## Data Flow

### ASR Processing Flow

```
Video (S3 URL)
       │
       ▼
/asr/infer ──► fetch_object_from_s3() ──► extract_audio()
                                          │
                                          ▼
                              ASRProcessor.process_audio()
                                          │
                                          ▼
                              Chunkformer.endless_decode()
                                          │
                                          ▼
                              time_transform() ──► ASRResult
                                                    │
                                              timestamped tokens
```

### AutoShot Processing Flow

```
Video (S3 URL)
       │
       ▼
/autoshot/infer ──► fetch_object_from_s3()
                              │
                              ▼
                   AutoShot.process_video()
                              │
                              ▼
                   get_frames() ──► TransNetV2.predict()
                                          │
                                          ▼
                              predictions_to_scenes()
                                          │
                                          ▼
                              [(start_frame, end_frame), ...]
```

### LLM Inference Flow

```
/llm/infer ──► GeminiAPIHandler._run_inference()
                                 │
                                 ▼
                    genai.GenerativeModel.generate_content()
                                 │
                                 ▼
                            LLMResponse
                                 │
                    text + token counts
```

---

## Deployment

### Docker Build Pattern

All services use multi-stage Docker builds:

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base
FROM ghcr.io/astral-sh/uv:latest AS uvbin
FROM base AS runtime
COPY --from=uvbin /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock
RUN uv venv --system-site-packages .venv
RUN uv sync --frozen --no-dev
CMD ["uv", "run", "service_xxx/main.py"]
```

### Service Dependencies

**External Services**:
- **Consul** (port 8500) - Service discovery
- **MinIO** (port 9000) - Object storage
- **GPU** - CUDA 12.8 capable GPU

**API Keys**:
- Gemini API key (LLM service)
- OpenRouter API key (LLM service)

### Building a Service

```bash
cd service_llm
docker build -t multimodal-llm:latest .
docker run -d --gpus all \
  --network host \
  -e GEMINI_API_KEY=${GEMINI_API_KEY} \
  multimodal-llm:latest
```

---

## Monitoring

### Prometheus Metrics

All services expose metrics at `/metrics`:

| Metric | Type | Description |
|--------|------|-------------|
| `service_requests_total` | Counter | Total requests by endpoint/status |
| `service_request_duration_seconds` | Histogram | Request latency |
| `service_request_errors_total` | Counter | Error count by type |
| `service_cpu_usage_percent` | Gauge | CPU utilization |
| `service_memory_bytes` | Gauge | RAM usage |
| `service_gpu_memory_used_bytes` | Gauge | GPU VRAM |
| `service_gpu_utilization_percent` | Gauge | GPU utilization |

### Health Endpoints

- `/health` - Simple health check (returns 200 if service is running)
- `/{service}/status` - Detailed status including:
  - CPU usage
  - Memory usage
  - GPU memory
  - GPU utilization
  - Loaded model info

### Logging

Structured logging with Loguru:

```python
from shared.logger import setup_service_logger

logger = setup_service_logger(
    service_name="service-llm",
    log_level="DEBUG",
    log_format="console"  # or "json" for production
)
```

---

## Development

### Adding a New Service

1. Create service directory: `service_new/`
2. Copy structure from existing service
3. Implement `core/service.py` inheriting from `BaseService`
4. Implement `model/registry.py` with `BaseModelHandler`
5. Update `main.py` entry point
6. Add Dockerfile
7. Update Consul service registration

### Adding a New Model Handler

```python
from shared.registry import BaseModelHandler, register_model

@register_model("my-model")
class MyModelHandler(BaseModelHandler[InputT, OutputT]):
    async def load_model_impl(self, device: str) -> None:
        # Load model weights
        pass
    
    async def run_inference(self, preprocessed: Any) -> Any:
        # Run model inference
        pass
```

---

## License

This project is part of the multimodal AI platform for video understanding and processing.
