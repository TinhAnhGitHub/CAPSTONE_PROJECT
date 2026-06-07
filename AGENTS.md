# Repository Guidelines

## Project Structure & Module Organization

This repository is split into service directories. `frontend/` is a Vite React app; UI code lives in `frontend/src/components`, pages in `frontend/src/pages`, API clients in `frontend/src/api`, stores in `frontend/src/stores`, and tests in `frontend/src/tests`. `backend/` is the FastAPI app for auth, videos, chat, Socket.IO streaming, and storage; code is under `backend/app`, with tests in `backend/tests`. `videodeepsearch/` contains the FastAPI/WebSocket retrieval service in `videodeepsearch/src/videodeepsearch`. `video_pipeline/` contains the Prefect ingestion pipeline in `video_pipeline/src/video_pipeline`, with Docker assets in `video_pipeline/docker`. `inference/` contains inference deployment assets and notebooks.

## Build, Test, and Development Commands

- `cd frontend && npm run dev`: start Vite.
- `cd frontend && npm run build`: create a production build.
- `cd frontend && npm run lint`: run ESLint.
- `cd frontend && npm test`: run Vitest once; use `npm run test:coverage` for coverage.
- `cd backend && uv sync && uv run main.py`: install backend dependencies and run the API.
- `cd backend && uv run pytest`: run backend pytest tests.
- `cd videodeepsearch && uv sync && uv run uvicorn main:app --reload --port 8080`: run the retrieval service locally.
- `cd video_pipeline/docker && docker compose up -d`: start pipeline infrastructure.

## Coding Style & Naming Conventions

Frontend code uses ES modules, React 19, and ESLint. Use 2-space indentation, PascalCase for React components, camelCase for functions and variables, and colocate component styles when practical. Python services target Python 3.11 or 3.12; use typed Pydantic models for API boundaries and keep business logic out of route handlers. `video_pipeline` defines Ruff with a 100-character line length.

## Testing Guidelines

Name Python tests `test_*.py`, classes `Test*`, and functions `test_*`; backend pytest auto-discovers `backend/tests`. Frontend tests use Vitest and Testing Library under `frontend/src/tests`. Add tests when changing API schemas, auth, Socket.IO events, stores, or chat parsing utilities.

## Commit & Pull Request Guidelines

Recent history uses short commits with prefixes such as `feat:` plus occasional `Update:` messages. Prefer `feat:`, `fix:`, `test:`, `docs:`, or `refactor:` with a concise summary. Pull requests should name the changed service, include test results, link issues, and add screenshots or recordings for visible frontend changes.

## Security & Configuration Tips

Do not commit real `.env` values. Service configuration lives in files such as `backend/.env`, `frontend/.env`, `videodeepsearch/config/settings.yaml`, and `video_pipeline/docker/.env`; document new required variables in the relevant README or example config.
