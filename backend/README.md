# Moment Retrieval Agent — Backend

A FastAPI-based backend service that powers the Moment Retrieval Agent application. It provides RESTful APIs and real-time Socket.IO communication for user authentication, video management, chat sessions, and AI agent streaming. The backend integrates with MongoDB (via Beanie ODM), MinIO for object storage, Google OAuth for authentication, and an external AI workflow service for video understanding and moment retrieval.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI 0.116+ |
| Runtime | Python 3.12+ |
| Async Driver | Uvicorn |
| Database | MongoDB (via Motor + Beanie ODM) |
| Object Storage | MinIO |
| Real-time | Python-SocketIO 5 (async mode) |
| Auth | Google OAuth 2.0 + PyJWT |
| LLM Framework | LlamaIndex 0.14+ (with OpenAI/MockLLM) |
| Validation | Pydantic 2 + Pydantic-Settings |
| **Task Queue** | **Celery 5 + Redis** |
| Package Manager | `uv` |

---

## Prerequisites

- **Python** >= 3.12
- **uv** — Modern Python package manager (`pip install uv`)
- **MongoDB** — Running instance (local or remote)
- **MinIO** — Running S3-compatible object storage
- **Google OAuth Credentials** — From Google Cloud Console
- **AI Workflow Service** (optional) — WebSocket service at `ws://localhost:8080/ws/start_workflow`

---

## Installation & Running

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create virtual environment and install dependencies
uv sync

# 3. Create environment file (see Environment Variables below)
cp .env .env.local   # then edit .env.local with your values

# 4. Start Redis (required for Celery task queue)
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 5. Start Celery worker (in a separate terminal)
cd backend
uv run celery -A app.worker.celery_app worker --loglevel=info --concurrency=4

# 6. Run the FastAPI application
uv run main.py
```

The server will start at `http://0.0.0.0:8011/` by default.

> **Thesis demo tip:** Run `uv run celery -A app.worker.celery_app flower` in an extra terminal to get a beautiful web UI at `http://localhost:5555` showing task status, retries, and worker health.

### Available Endpoints

- `GET /` — API info
- `GET /health` — Health check
- `GET /docs` — Interactive Swagger UI (OpenAPI)
- `GET /redoc` — ReDoc documentation
- Socket.IO endpoint: `/socket.io/`

---

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Google OAuth 2.0 (required for authentication)
GOOGLE_OAUTH_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
GOOGLE_OAUTH_CLIENT_SECRET=your-google-client-secret

# MongoDB connection
MONGO_URI=mongodb://localhost:27017

# MinIO S3-compatible storage
MINIO_PUBLIC_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Server binding
HOST=0.0.0.0
PORT=8011

# Redis (Celery broker + result backend)
REDIS_URL=redis://localhost:6379/0

# Ingestion pipeline service endpoints
INGESTION_SERVICE_URL=http://<pipeline-host>:8050
INGESTION_CANCEL_URL=http://<pipeline-host>:8000
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GOOGLE_OAUTH_CLIENT_ID` | Yes | — | Google OAuth Client ID |
| `GOOGLE_OAUTH_CLIENT_SECRET` | Yes | — | Google OAuth Client Secret |
| `MONGO_URI` | Yes | `mongodb://localhost:27017` | MongoDB connection string |
| `MINIO_PUBLIC_ENDPOINT` | Yes | `localhost:9000` | MinIO server host:port |
| `MINIO_ACCESS_KEY` | Yes | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | Yes | `minioadmin` | MinIO secret key |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8011` | Server bind port |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis broker + result backend for Celery |
| `INGESTION_SERVICE_URL` | No | `http://100.113.186.28:8050` | Video pipeline ingestion endpoint |
| `INGESTION_CANCEL_URL` | No | `http://100.113.186.28:8000` | Video pipeline cancel endpoint |

> **Note:** Additional config (collection names, upload directory) is defined in `app/core/config.py` and can be overridden via env vars if needed.

---

## Application Configuration

### Project Config (`pyproject.toml`)
- **Build system:** Standard Python project with `uv` lockfile
- **Dependencies:** FastAPI, Beanie, Motor, MinIO, Socket.IO, LlamaIndex, OpenCV, MoviePy, Pydantic Settings, etc.
- **Python version:** `>=3.12`

### App Settings (`app/core/config.py`)
Pydantic-Settings class that loads from `.env`:
- **MongoDB:** `MONGO_URI`, `MONGO_DB` (default: `mydatabase`)
- **Collections:** Configurable names for `chat_history`, `users`, `groups`, `videos`, `session_videos`, `chat_messages`
- **Uploads:** `UPLOAD_DIR` (default: `uploads`)
- **Server:** `HOST`, `PORT`
- **MinIO:** `MINIO_PUBLIC_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`
- **Google OAuth:** `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`

---

## Overall Code Architecture

### Directory Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── chat.py                    # REST chat endpoints (send, history, sessions)
│   │   ├── user.py                    # User auth, uploads, groups, videos, sessions
│   │   ├── ingestion.py               # Ingestion progress webhook from pipeline
│   │   ├── socket.py                  # Main Socket.IO event handlers (stream_chat, join, cancel)
│   │   └── socket_handlers/
│   │       ├── agent_stream.py        # Handle AgentStream events (thinking + text)
│   │       ├── agent_progress_event.py# Handle AgentProgressEvent (loading indicator)
│   │       ├── save_text_block.py     # Flush accumulated text to message blocks
│   │       ├── save_thinking_block.py # Flush accumulated thinking steps
│   │       ├── save_tool_block.py     # Flush accumulated tool calls
│   │       └── utils/
│   │           └── session_room.py    # Helper: session room naming
│   ├── core/
│   │   ├── config.py                  # Pydantic settings + env loading
│   │   ├── dependencies.py            # FastAPI dependency injectors (Agent, ChatService, UserService)
│   │   ├── lifespan.py                # Startup/shutdown lifecycle (MongoDB, MinIO, services)
│   │   └── auth.py                    # Auth utilities (if extended)
│   ├── model/
│   │   ├── user.py                    # User document (Beanie)
│   │   ├── chat_history.py            # Chat session document
│   │   ├── session_message.py         # Message document with polymorphic blocks
│   │   ├── video.py                   # Video metadata document
│   │   ├── group.py                   # Video group document
│   │   └── session_video.py           # Session-to-video mapping with selection state
│   ├── repository/
│   │   ├── user.py                    # (reserved) User repository layer
│   │   └── chat.py                    # (reserved) Chat repository layer
│   ├── schema/
│   │   ├── user.py                    # Pydantic schemas: Token, TokenData, JWT config
│   │   ├── chat.py                    # Pydantic schemas: ChatRequest, ChatResponse, SessionInfo
│   │   ├── group.py                   # Group schemas
│   │   └── data_handle.py             # Data handling schemas
│   └── service/
│       ├── agent.py                   # LLM agent wrapper (LlamaIndex)
│       ├── chat.py                    # Chat persistence service
│       ├── user.py                    # User business logic (auth, videos, groups, thumbnails)
│       └── minio.py                   # MinIO client wrapper (uploads, thumbnails, URLs)
├── utils/
│   ├── random_name.py                 # Random chat/group name generator
│   └── blocks.py                      # Block utilities
├── main.py                            # FastAPI app factory + uvicorn entrypoint
├── pyproject.toml                     # Project metadata + dependencies
├── uv.lock                            # Locked dependency versions
└── .env                               # Environment variables
```

### Layered Architecture

The backend follows a clean layered architecture:

```
┌─────────────────────────────────────┐
│  API Layer (FastAPI Routers)        │  ← HTTP + Socket.IO handlers
│  chat.py / user.py / socket.py      │
├─────────────────────────────────────┤
│  Service Layer                      │  ← Business logic
│  ChatService / UserService / Agent  │
├─────────────────────────────────────┤
│  Repository / Model Layer           │  ← Data access
│  Beanie Documents (MongoDB)         │
├─────────────────────────────────────┤
│  External Services                  │  ← MinIO, Google OAuth, AI WS
│  MinioService / Google Auth / WS    │
└─────────────────────────────────────┘
```

#### 1. API Layer (`app/api/`)
- **`chat.py`:** REST endpoints for sending messages, retrieving chat history, listing sessions, and deleting sessions.
- **`user.py`:** REST endpoints for Google OAuth login, file uploads, group/video/session CRUD, chat history search, and thumbnail generation.
- **`ingestion.py`:** Webhook endpoint for the ingestion pipeline to report progress (`overall_percentage`, `run_id`).
- **`socket.py`:** Core Socket.IO handler for `stream_chat`, `join_session`, and `cancel_stream`. Manages global session tasks and accumulates streaming blocks.
- **`socket_handlers/`:** Modular handlers for specific event types from the AI workflow service:
  - `agent_stream.py` — Parses thinking deltas and text responses
  - `agent_progress_event.py` — Emits loading indicators
  - `save_*_block.py` — Flushes accumulated content into typed blocks

#### 2. Service Layer (`app/service/`)
- **`ChatService`** — Persists messages to MongoDB, manages `ChatHistory` sessions, and converts content blocks to `SessionMessage` documents.
- **`UserService`** — Handles Google OAuth verification, JWT creation, user CRUD, video uploads (via MinIO), group management, session management, thumbnail generation, and chat history text search.
- **`Agent`** — Thin wrapper around LlamaIndex LLM (`MockLLM` in dev; can be swapped for `OpenAI`).
- **`MinioService`** — Manages MinIO buckets (`avatars`, `videos`, `thumbnails`), uploads videos, generates thumbnails using MoviePy + Pillow, creates presigned URLs, and deletes objects.

#### 3. Model Layer (`app/model/`)
All models extend Beanie `Document` for MongoDB ODM:
- **`User`** — User profile with Google OAuth linkage.
- **`ChatHistory`** — Chat session metadata (`user_id`, `name`, `last_updated`).
- **`SessionMessage`** — Polymorphic message with a list of `ContentBlock` (text, image, video, thinking, tools, tool_call, tool_call_result).
- **`Video`** — Video metadata (`user_id`, `group_id`, `name`, `length`, `fps`, `thumbnail`, `url`, `ingested_status`).
- **`Group`** — Video collection/group per user.
- **`SessionVideo`** — Many-to-many mapping between sessions and videos with a `selected` boolean.

#### 4. Configuration & Lifespan (`app/core/`)
- **`config.py`** — `AppSettings` (Pydantic-Settings) loads from `.env` and provides typed configuration.
- **`lifespan.py`** — `AppState` singleton holds initialized services (MongoDB client, MinIO, LLM, services, Socket.IO server). The `lifespan` context manager runs on startup/shutdown.
- **`dependencies.py`** — FastAPI `Depends` injectors expose services to route handlers.

---

### Real-Time Streaming Architecture

The backend acts as a **bridge** between the frontend and the external AI workflow service:

```
Frontend (Socket.IO)  ←→  Backend (Socket.IO + WebSocket)  ←→  AI Service (WebSocket)
        |                           |                               |
    stream_chat                 stream_chat                   start_workflow
    response/thinking           accum + emit                  AgentStream / ToolCall
    media/tool_result           format + emit                 ToolCallResult
    stream_end                  save_message                  AgentOutput
```

**Key Design Patterns:**
- **Session Rooms:** Each chat session has a Socket.IO room (`session:<session_id>`) so events are scoped.
- **Global Task Registry:** `global_session_tasks` tracks active streaming tasks per session to support cancellation and reconnection (`continue_stream`).
- **AccumulatedData:** An accumulator pattern collects streaming deltas (text, thinking, tools) and flushes them into typed blocks when the event type changes.
- **Block-Based Persistence:** Messages are stored as arrays of polymorphic blocks in MongoDB, matching the frontend's block renderer.

---

### Authentication Flow

1. **Google OAuth Login**
   - Frontend sends authorization `code` to `POST /api/user/login/google`.
   - Backend exchanges code for tokens with Google.
   - Backend verifies the `id_token` using Google's certs.
   - Backend creates/updates `User` in MongoDB.
   - Backend generates a JWT (`my_app_token`) with `user_id`, `email`, `exp`.
   - Frontend stores JWT and uses it in the `Authorization: Bearer <token>` header.

2. **Token Verification**
   - `verify_token` dependency decodes the JWT.
   - On failure, it currently falls back to a hardcoded tester payload (for development convenience).

---

### External Service Integrations

| Service | Protocol | Endpoint | Purpose |
|---------|----------|----------|---------|
| **MongoDB** | Motor (async) | `MONGO_URI` | Primary database for all documents |
| **MinIO** | S3-compatible HTTP | `MINIO_PUBLIC_ENDPOINT` | Video, thumbnail, avatar storage |
| **Google OAuth** | HTTPS | `oauth2.googleapis.com` | User authentication |
| **AI Workflow** | WebSocket | `ws://localhost:8080/ws/start_workflow` | Video understanding & moment retrieval |
| **Ingestion Pipeline** | HTTP webhook | `POST /api/ingestion/service/status/{video_id}` | Video ingestion progress |

---

## Key Features

- **Google OAuth Authentication:** Secure login with JWT sessions
- **Video Upload & Storage:** Multi-file upload to MinIO with thumbnail generation
- **Ingestion Tracking:** Real-time ingestion progress via webhook + Socket.IO push
- **Block-Based Chat:** Polymorphic message blocks (text, image, video, thinking, tools)
- **Real-Time AI Streaming:** Bidirectional streaming through Socket.IO + WebSocket bridge
- **Session Management:** Create, rename, delete chat sessions; persist messages in MongoDB
- **Group & Video Management:** CRUD for video groups; per-session video selection
- **Thumbnail Generation:** On-demand timeline thumbnails (5 frames around a timestamp) using MoviePy
- **Chat History Search:** Full-text search across message blocks
- **CORS:** Configured to allow all origins for development (`allow_origins=["*"]`)

---

## Notes

- **MockLLM:** The current LLM is `MockLLM(max_tokens=2)` for development. Replace with `OpenAI` or another LlamaIndex LLM in production.
- **Hardcoded Fallback:** `verify_token` in `app/api/user.py` falls back to a tester payload on JWT errors. **Remove this before production.**
- **AI Service URL:** The WebSocket URL for the AI workflow is hardcoded to `ws://localhost:8080/ws/start_workflow`. Update this in `app/api/socket.py` for your deployment.
- **MinIO URLs:** Thumbnail and video URLs are currently generated with a hardcoded base (`http://100.113.186.28:9000/`). Consider making this configurable.
- **Development CORS:** `allow_origins=["*"]` is set for development. Restrict this in production.
- **Static Files:** Uploaded files are served from `/uploads` via `StaticFiles`.
