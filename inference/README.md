# Inference Services

This directory contains the inference infrastructure for the Moment Retrieval Agent. It deploys and orchestrates multiple ML models for video understanding, including shot boundary detection, automatic speech recognition (ASR), sparse text embedding, and multimodal (vision-language) embedding. The services are containerized with Docker Compose and expose inference APIs via NVIDIA Triton Inference Server and llama.cpp servers.

---

## Deployed Models

| Model | Container | Purpose | Input | Output |
|-------|-----------|---------|-------|--------|
| **autoshot** | Triton (`onnxruntime_onnx`) | Shot boundary detection / video segmentation | Video frames `[1,100,27,48,3]` | Shot boundary probabilities `[1,100,1]` |
| **splade** | Triton (`python` backend) | Sparse text embedding for retrieval | Text string | Sparse vector (indices + values) |
| **parakeet_asr** | Triton (`python` backend) | Automatic Speech Recognition (backup) | Audio bytes | Transcription text |
| **mmbert** | llama.cpp server | Dense text embedding | Text | Embedding vector |
| **qwen_vl_embedding** | llama.cpp server-cuda | Multimodal (image+text) embedding | Text / Images | Embedding vector |
| **qwen3-asr** | qwenllm/qwen3-asr | Automatic Speech Recognition (primary) | Audio | Transcription text |
| **ocr_lighton** | *(commented out)* llama.cpp server-cuda | OCR text extraction | Images | Extracted text |

---

## Prerequisites

- **Docker** & **Docker Compose**
- **NVIDIA Container Toolkit** (`nvidia-docker2`) for GPU support
- **NVIDIA GPU** with sufficient VRAM (varies by model)
- An external Docker network named `video_shared_net`:
  ```bash
  docker network create video_shared_net
  ```

---

## Environment Variables

Create a `.env` file in the `inference/` directory:

```env
# Triton Inference Server ports
TRITON_HTTP_PORT=8000
TRITON_GRPC_PORT=8001
TRITON_METRICS_PORT=8002

# Triton model repository path (absolute path on host)
TRITON_MODEL_REPOSITORY=/path/to/triton_infra/model_repository

# Hugging Face token (for downloading models inside Triton)
HF_TOKEN=hf_your_token_here

# mmbert (llama.cpp text embedding)
MMBERT_PORT=8003
MMBERT_MODEL_DIR=/path/to/mmbert/models
MMBERT_MODEL=mmbert.gguf

# Qwen-VL embedding (llama.cpp multimodal embedding)
QWEN_VL_EMBEDDING_PORT=8004
QWEN_VL_EMBEDDING_MODEL_DIR=/path/to/qwen_vl/models
QWEN_VL_EMBEDDING_MODEL=qwen_vl.gguf
QWEN_VL_EMBEDDING_MODEL_MMPROJ=mmproj.gguf

# Qwen3 ASR
QWEN_ASR_PORT=8005
HUGGINGFACE_CACHE_DIR=~/.cache/huggingface

# OCR Lighton (currently commented out in docker-compose)
# OCR_LIGHTON_PORT=8006
# OCR_LIGHTON_MODEL_DIR=/path/to/ocr/models
# OCR_LIGHTON_MODEL=ocr.gguf
# OCR_LIGHTON_MMPROJ=mmproj.gguf
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TRITON_HTTP_PORT` | Yes | Triton HTTP API port |
| `TRITON_GRPC_PORT` | Yes | Triton gRPC API port |
| `TRITON_METRICS_PORT` | Yes | Triton Prometheus metrics port |
| `TRITON_MODEL_REPOSITORY` | Yes | Host path to Triton model repository |
| `HF_TOKEN` | Yes | HuggingFace token for model downloads |
| `MMBERT_PORT` | Yes | Port for mmbert llama.cpp server |
| `MMBERT_MODEL_DIR` | Yes | Host path to mmbert GGUF model |
| `MMBERT_MODEL` | Yes | Filename of the mmbert GGUF model |
| `QWEN_VL_EMBEDDING_PORT` | Yes | Port for Qwen-VL llama.cpp server |
| `QWEN_VL_EMBEDDING_MODEL_DIR` | Yes | Host path to Qwen-VL models |
| `QWEN_VL_EMBEDDING_MODEL` | Yes | Qwen-VL GGUF model filename |
| `QWEN_VL_EMBEDDING_MODEL_MMPROJ` | Yes | Qwen-VL multimodal projection GGUF |
| `QWEN_ASR_PORT` | No | Port for Qwen3 ASR (default: 8005) |
| `HUGGINGFACE_CACHE_DIR` | No | HF cache directory (default: `~/.cache/huggingface`) |

---

## Running the Services

```bash
# 1. Navigate to inference directory
cd inference

# 2. Create the shared Docker network (one-time setup)
docker network create video_shared_net

# 3. Create .env file with your paths and tokens
cp .env.example .env  # then edit .env

# 4. Start all services
docker compose up -d

# 5. Check status
docker compose ps

# 6. View logs
docker compose logs -f triton-server
docker compose logs -f qwen_vl_embedding

# 7. Stop all services
docker compose down
```

### Individual Services

```bash
# Start only Triton
docker compose up -d triton-server

# Start only Qwen-VL embedding
docker compose up -d qwen_vl_embedding

# Start only Qwen3 ASR
docker compose up -d qwen3-asr

# Start only mmbert
docker compose up -d mmbert
```

---

## Triton Model Repository

The Triton server loads models from `triton_infra/model_repository/`:

```
triton_infra/model_repository/
├── autoshot/
│   ├── config.pbtxt          # Model configuration (ONNX Runtime, GPU)
│   └── 1/
│       └── model.onnx        # TransNetV2 shot detection model
├── splade/
│   ├── config.pbtxt          # Model configuration (Python backend)
│   └── 1/
│       └── model.py          # SPLADE sparse embedding implementation
│       └── splade_env.tar.gz # Python execution environment
└── parakeet_asr/             # (in backup/)
    ├── config.pbtxt
    └── 1/
        └── model.py
```

### autoshot — Shot Boundary Detection

- **Backend:** ONNX Runtime
- **Input:** `input_frames` — uint8 tensor `[1, 100, 27, 48, 3]`
- **Output:** `shot_boundary_prob` — float32 tensor `[1, 100, 1]`
- **GPU:** Required (`KIND_GPU`)
- **Purpose:** Detects shot boundaries in video to segment content for retrieval

### splade — Sparse Text Embedding

- **Backend:** Python (custom execution environment)
- **Input:** `TEXT` — string `[1]`
- **Output:** `INDICES` (int32) + `VALUES` (float32) — sparse vector representation
- **Purpose:** Generates sparse lexical embeddings for efficient text retrieval
- **Client:** See `notebooks/test.ipynb` for example usage

### parakeet_asr — Speech Recognition (Backup)

- **Backend:** Python
- **Input:** `AUDIO` — audio bytes
- **Output:** `OUTPUT` — transcription text
- **Purpose:** Converts video audio to text for indexing and retrieval

---

## Custom Dockerfile (triton_infra/)

`triton_infra/Dockerfile` extends `nvcr.io/nvidia/tritonserver:24.08-py3` to:

- Install `ffmpeg`, `libgl1`, `libglib2.0-0` for video/image processing
- Install `uv` (fast Python package manager)
- Install dependencies from `pyproject.toml`
- Copy the model repository into the container
- Set HuggingFace cache directories
- Expose Triton ports: `8000` (HTTP), `8001` (gRPC), `8002` (metrics)

Build the custom image:
```bash
cd triton_infra
docker build -t custom-triton:latest .
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Inference Stack                          │
├─────────────────────────────────────────────────────────────┤
│  Triton Inference Server (nvcr.io/nvidia/tritonserver)      │
│  ├── autoshot        → ONNX Runtime → GPU → Shot detection  │
│  ├── splade          → Python backend → Sparse embedding    │
│  └── parakeet_asr    → Python backend → ASR (backup)        │
├─────────────────────────────────────────────────────────────┤
│  llama.cpp Servers                                          │
│  ├── mmbert          → CPU/GGUF → Text embedding            │
│  ├── qwen_vl_embedding → CUDA/GGUF → Multimodal embedding   │
│  └── ocr_lighton     → *(disabled)* → OCR                   │
├─────────────────────────────────────────────────────────────┤
│  Standalone Containers                                      │
│  └── qwen3-asr       → vLLM-based → Primary ASR             │
└─────────────────────────────────────────────────────────────┘
                              │
                    Shared Docker Network
                    (video_shared_net)
```

### Service Communication

All containers connect via the external Docker network `video_shared_net`, allowing other services (e.g., the video pipeline) to reach them by container name:

| Service | Internal Address | Protocol |
|---------|-----------------|----------|
| Triton HTTP | `triton-server:8000` | HTTP/REST |
| Triton gRPC | `triton-server:8001` | gRPC |
| mmbert | `te_mmbert:8000` | HTTP (OpenAI-compatible) |
| Qwen-VL | `qwen_vl_embedding:8080` | HTTP (OpenAI-compatible) |
| Qwen3-ASR | `qwen3-asr-06b:80` | HTTP (OpenAI-compatible) |

---

## Testing & Client Usage

### SPLADE Client Example

See `notebooks/test.ipynb` for a complete working example:

```python
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

client = InferenceServerClient(url="localhost:8001")

texts = ["Hello, world!", "Video retrieval system"]
text_input = InferInput("TEXT", [len(texts), 1], "BYTES")
text_input.set_data_from_numpy(np.array([[t] for t in texts], dtype=object))

outputs = [
    InferRequestedOutput("INDICES"),
    InferRequestedOutput("VALUES"),
]

result = client.infer("splade", inputs=[text_input], outputs=outputs)
indices = result.as_numpy("INDICES")
values = result.as_numpy("VALUES")
```

### Health Checks

All services expose health check endpoints:

```bash
# Triton health
curl http://localhost:8000/v2/health/ready

# mmbert health
curl http://localhost:8003/v1/models

# Qwen-VL health
curl http://localhost:8004/v1/models

# Qwen3-ASR health
curl http://localhost:8005/v1/models
```

---

## Hardware Requirements

| Service | Minimum GPU | VRAM | Notes |
|---------|-------------|------|-------|
| Triton (autoshot) | NVIDIA GPU | 2 GB | ONNX Runtime GPU |
| Triton (splade) | CPU only | — | Python backend |
| mmbert | CPU preferred | — | `--mlock`, 12 threads |
| qwen_vl_embedding | NVIDIA GPU | 8+ GB | `--n-gpu-layers 2`, CUDA container |
| qwen3-asr | NVIDIA GPU | 4+ GB | `--gpu-memory-utilization 0.25` |

> **Tip:** If GPU resources are limited, models with CPU backends (`splade`, `mmbert`) can run on CPU while GPU-intensive models (`autoshot`, `qwen_vl`, `qwen3-asr`) share the GPU.

---

## Notes

- **Model files** (`.onnx`, `.gguf`, `.tar.gz`) are excluded from Git via `.gitignore` due to large file sizes. Download or mount them separately.
- **OCR Lighton** is commented out in `docker-compose.yaml` — uncomment and configure ports/model paths if needed.
- **Parakeet ASR** is preserved in `backup/`; the primary ASR is now `qwen3-asr`.
- **Shared network:** The `video_shared_net` Docker network must exist before running `docker compose up`. Other project services (e.g., the video pipeline) may also connect to this network.
- **Auto-restart:** All services use `restart: unless-stopped` for high availability.
- **HF Token:** Required for Triton to download models from HuggingFace Hub at runtime. Store securely and do not commit to version control.
