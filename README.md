# Moment Retrieval Agent

> **Capstone Project — Intelligent Multi-Modal Video Search & Understanding**

A full-stack system for natural-language video retrieval. Users upload videos, which are automatically processed into a rich multi-modal index (frames, transcripts, captions, embeddings, knowledge graphs). They then chat with an AI agent via an interactive web interface to find specific moments across their video library with precise timestamps.

---

## Demo Video

<!-- TODO: Replace the link below with your YouTube demo URL -->

[![Watch the Demo](https://img.youtube.com/vi/u34fC7j0WmA/0.jpg)](https://www.youtube.com/watch?v=u34fC7j0WmA)

> **Click the thumbnail above to watch the full demo on YouTube.**
>
> *The demo showcases the end-to-end workflow: video upload → ingestion progress → natural language querying → AI agent streaming → video moment retrieval with timestamps.*

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                    ┌─────────────────────────┐                              │
│                    │   React + Vite Frontend │  ← Upload, Chat, Video Player │
│                    │    [frontend/]          │                              │
│                    └──────────┬──────────────┘                              │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │ HTTP + Socket.IO
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                         API GATEWAY                                          │
│                    ┌─────────────────────────┐                              │
│                    │   FastAPI Backend       │  ← Auth, Chat, Video CRUD    │
│                    │    [backend/]           │                              │
│                    └──────────┬──────────────┘                              │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │ WebSocket
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                      AGENT & PIPELINE SERVICES                               │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────────────────┐   │
│  │  VideoDeepSearch         │    │  Video Pipeline                      │   │
│  │  [videodeepsearch/]      │    │  [video_pipeline/]                   │   │
│  │  Multi-agent system      │◄──►│  Video ingestion & indexing          │   │
│  │  Greeter → Orchestrator  │    │  ASR · Caption · Embedding · KG      │   │
│  │  → Worker Agents         │    │  Qdrant · ArangoDB · Elasticsearch   │   │
│  └──────────────────────────┘    └──────────────────────────────────────┘   │
│                                                                              │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │ HTTP / gRPC
┌───────────────────────────────┼─────────────────────────────────────────────┐
│                         INFERENCE CLUSTER                                    │
│                    ┌─────────────────────────┐                              │
│                    │   Triton + llama.cpp    │  ← Model Serving             │
│                    │    [inference/]         │                              │
│                    │  · Autoshot (segment)   │                              │
│                    │  · Qwen-VL (embed)      │                              │
│                    │  · mmBERT (embed)       │                              │
│                    │  · SPLADE (sparse)      │                              │
│                    │  · Qwen3-ASR            │                              │
│                    └─────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System Components

| Component | Technology | Purpose | Details |
|-----------|------------|---------|---------|
| **Frontend** | React 19, Vite, TailwindCSS, Zustand | Web UI for video upload, chat, and player | [frontend/README.md](frontend/README.md) |
| **Backend** | FastAPI, Python-SocketIO, Beanie/MongoDB | Auth, chat sessions, video metadata, MinIO | [backend/README.md](backend/README.md) |
| **Video Pipeline** | Prefect, FastAPI, Qdrant, ArangoDB, Elasticsearch | Ingest & index videos: ASR, caption, embed, KG | [video_pipeline/README.md](video_pipeline/README.md) |
| **VideoDeepSearch** | Agno, FastAPI, OpenRouter | Multi-agent NL video search & retrieval | [videodeepsearch/README.md](videodeepsearch/README.md) |
| **Inference** | Triton, llama.cpp, vLLM | Model serving: autoshot, QwenVL, mmBERT, SPLADE, ASR | [inference/README.md](inference/README.md) |

---

## High-Level Data Flow

```
1. User uploads video(s) via React frontend
        │
        ▼
2. Backend receives upload, stores in MinIO, creates DB records
        │
        ▼
3. Backend triggers Video Pipeline via API (POST /uploads)
        │
        ▼
4. Video Pipeline orchestrates ~20 parallel stages:
   ├─ Audio Branch: ASR → Segmentation → Caption → Embedding → Qdrant
   └─ Image Branch: Frame Extraction → Caption/OCR → Embedding → Qdrant
        │
        ▼
5. Knowledge Graph pipeline builds entity/event graph → ArangoDB
        │
        ▼
6. User asks a natural language question in chat
        │
        ▼
7. Backend forwards query via WebSocket to VideoDeepSearch agent
        │
        ▼
8. Multi-agent system (Greeter → Orchestrator → Workers) searches:
   ├─ Qdrant: vector similarity on images, segments, audio
   ├─ Elasticsearch: BM25 over OCR text
   └─ ArangoDB: graph traversal over entities & events
        │
        ▼
9. Agent streams results back to frontend in real-time
        │
        ▼
10. Frontend renders rich blocks: text, image galleries, video segments with timestamps
```

---

## Key Features

- **Natural Language Video Search:** Ask questions like *"Find the scene where the speaker discusses climate change"* and get exact timestamps
- **Real-Time Streaming:** Agent responses stream live with thinking steps, tool calls, and media results
- **Multi-Modal Indexing:** Videos are indexed by audio (ASR), visual frames, captions, OCR text, and knowledge graph entities
- **Knowledge Graph:** Automatically extracts entities, events, and relationships; detects thematic communities
- **Group & Session Management:** Organize videos into groups; chat in persistent sessions with full history
- **Google OAuth:** Secure authentication with JWT sessions
- **Block-Based Chat UI:** Rich message rendering with Markdown, syntax highlighting, image galleries, and video players

---

## Project Structure

```
CAPSTONE_PROJECT/
├── frontend/               # React web application
│   └── README.md          # ← Detailed frontend docs
├── backend/               # FastAPI application server
│   └── README.md          # ← Detailed backend docs
├── video_pipeline/        # Prefect video ingestion pipeline
│   └── README.md          # ← Detailed pipeline docs
├── videodeepsearch/       # Multi-agent video search engine
│   └── README.md          # ← Detailed agent docs
├── inference/             # ML model inference services (Triton + llama.cpp)
│   └── README.md          # ← Detailed inference docs
├── lib/                   # Shared libraries (if any)
├── capstone_diagrams.excalidraw
├── LICENSE                # MIT License
└── README.md              # ← This file
```

---

## Getting Started

Each component has its own setup instructions. See the linked READMEs above for:

- Environment variables
- Installation steps
- Architecture details
- API references

### Quick Start (Full Stack)

```bash
# 1. Start inference services (Docker Compose)
cd inference
docker network create video_shared_net
docker compose up -d

# 2. Start video pipeline (Docker Compose)
cd ../video_pipeline/docker
docker compose up -d

# 3. Start videodeepsearch (Docker Compose)
cd ../../videodeepsearch
docker compose up -d

# 4. Start backend
# cd ../backend
# cp .env.example .env
# uv sync && uv run main.py

# 5. Start frontend
# cd ../frontend
# npm install && npm run dev
```

---

## Technology Summary

| Layer | Technologies |
|-------|-------------|
| **Frontend** | React 19, Vite, TailwindCSS, Zustand, React Query, Socket.IO Client, Video.js, React Markdown |
| **Backend** | FastAPI, Python-SocketIO, Beanie (MongoDB ODM), Motor, MinIO, PyJWT, LlamaIndex |
| **Pipeline** | Prefect, FastAPI, OpenCV, MoviePy, PostgreSQL, Qdrant, ArangoDB, Elasticsearch |
| **Agent** | Agno, FastAPI, OpenRouter, WebSocket, MLflow |
| **Inference** | NVIDIA Triton, llama.cpp, vLLM, ONNX Runtime, Docker Compose |
| **Models** | Qwen-VL, mmBERT, SPLADE, Autoshot (TransNetV2), Qwen3-ASR |

---

## Authors

- **Tinh Anh** — [TinhAnhGitHub](https://github.com/TinhAnhGitHub)
- **Gia Phuc** — [phucnguyenlamp](https://github.com/phucnguyenlamp)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- Built with [FastAPI](https://fastapi.tiangolo.com/), [React](https://react.dev/), and [Prefect](https://www.prefect.io/)
- Multi-agent framework powered by [Agno](https://github.com/agno-agi/agno)
- LLM inference via [OpenRouter](https://openrouter.ai/)
- Evaluation based on [MAGNET / AVHaystacksQA](https://arxiv.org/abs/2506.07016)

---

> **Capstone Project — 2026**
