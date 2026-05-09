# Moment Retrieval Agent — Frontend

A modern React-based web application for intelligent video search and moment retrieval. Users can upload videos, organize them into groups, and interact with an AI agent via natural language to find specific moments within their video library. The app features real-time streaming responses, block-based message rendering, and a responsive chat interface.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | React 19 |
| Build Tool | Vite 7 |
| Styling | TailwindCSS 4 + `@tailwindcss/vite` |
| Routing | React Router DOM 7 |
| State Management | Zustand 5 (with persistence) |
| Server State | React Query 3 (TanStack Query) |
| Real-time | Socket.IO Client 4 |
| Auth | Google OAuth 2.0 (`@react-oauth/google`) |
| UI Components | Headless UI, Heroicons |
| Video Player | Video.js 8 |
| Markdown | React Markdown + React Syntax Highlighter |
| Forms | React Hook Form |
| Notifications | React Hot Toast |
| Utilities | clsx, JSZip, use-debounce, jwt-decode |

---

## Prerequisites

- **Node.js** >= 18 (recommended: 20 LTS)
- **npm** or **yarn**

---

## Installation & Running

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Create environment file (see Environment Variables below)
cp .env .env.local   # then edit .env.local with your values

# 4. Start development server
npm run dev

# 5. Start with network access (for mobile testing)
npm run dev:server

# 6. Build for production
npm run build

# 7. Preview production build
npm run preview
```

The dev server typically runs at `http://localhost:5173/`.

---

## Environment Variables

Create a `.env` file in the `frontend/` directory:

```env
# Google OAuth 2.0 credentials (from Google Cloud Console)
VITE_GOOGLE_OAUTH_CLIENT_ID=your-google-client-id.apps.googleusercontent.com
VITE_GOOGLE_OAUTH_CLIENT_SECRET=your-google-client-secret

# Backend API base URL (must match the backend server)
VITE_PRIMARY_URL=http://localhost:8011/
```

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_GOOGLE_OAUTH_CLIENT_ID` | Yes | Google OAuth Client ID for sign-in |
| `VITE_GOOGLE_OAUTH_CLIENT_SECRET` | Yes | Google OAuth Client Secret |
| `VITE_PRIMARY_URL` | Yes | Base URL of the backend FastAPI server |

> **Note:** All env vars prefixed with `VITE_` are exposed to the client bundle at build time.

---

## Application Configuration

### Vite Config (`vite.config.js`)
- **Plugins:** `@vitejs/plugin-react` (Fast Refresh), `@tailwindcss/vite`
- **Path Alias:** `@/` resolves to `src/`
- **Allowed Hosts:** `capstone.departmentofcodingknight.site` (for deployment)

### JS Config (`jsconfig.json`)
Enables IDE path resolution for `@/*` → `src/*`.

### ESLint (`eslint.config.js`)
- Extends: `@eslint/js` recommended, `react-hooks/recommended-latest`, `react-refresh/vite`
- Globals: browser environment
- Custom rule: ignores unused vars matching `^[A-Z_]` (for React components)

### TailwindCSS
- Uses TailwindCSS v4 with the Vite plugin (`@tailwindcss/vite`)
- Typography plugin (`@tailwindcss/typography`) for prose styling
- Scrollbar plugin (`tailwind-scrollbar`) for custom scrollbars
- Custom CSS variables for theming in `src/index.css`

---

## Overall Code Architecture

### Directory Structure

```
frontend/
├── public/                 # Static assets (favicons, test images/videos)
├── src/
│   ├── api/
│   │   ├── api.jsx         # Axios instance with JWT interceptor
│   │   ├── socket.jsx      # Socket.IO client initialization
│   │   └── services/
│   │       ├── query.jsx   # Basic upload helper
│   │       └── hooks/
│   │           ├── query.jsx    # React Query hooks (videos, groups, chat)
│   │           └── edit.jsx     # Generic edit state hook
│   ├── components/
│   │   ├── Appbar/         # Top application bar
│   │   ├── common/         # Shared UI (Upload, Chip, ImageCard, VideoPlayer, etc.)
│   │   └── Home/
│   │       ├── Chat.jsx            # Main chat interface
│   │       ├── BlockRenderer.jsx   # Renders message blocks by type
│   │       ├── SendButton.jsx      # Send/stop streaming button
│   │       ├── Chat/
│   │       │   ├── Blocks/         # TextBlock, ImageGallery, VideoBlock
│   │       │   ├── Thinking.jsx    # Thinking step UI
│   │       │   ├── Tools.jsx       # Tool call UI
│   │       │   └── VideoPlayer.jsx # Video player with markers
│   │       └── Sidebar/
│   │           ├── Sidebar.jsx     # Left sidebar layout
│   │           └── components/     # History, Library, Video lists, UserBar
│   ├── constants/
│   │   ├── url.js          # PRIMARY_URL from env
│   │   ├── auth.js         # Google client IDs from env
│   │   └── image.js        # Image constants
│   ├── pages/
│   │   ├── Home.jsx        # Main layout (Sidebar + Chat)
│   │   ├── Login.jsx       # Google OAuth login page
│   │   └── PrivateHome.jsx # Private route wrapper
│   ├── routes/
│   │   ├── index.jsx       # Router configuration
│   │   ├── publicRoutes.jsx
│   │   └── privateRoutes.jsx
│   ├── stores/
│   │   ├── user.jsx        # User auth state (Zustand + persist)
│   │   ├── chat.jsx        # Chat session state (Zustand + persist)
│   │   ├── modal.js        # Generic modal state
│   │   └── videoModal.js   # Video preview modal state
│   ├── utils/
│   │   ├── chat/
│   │   │   ├── parseChunkToBlock.js   # Parse socket chunks to blocks
│   │   │   ├── addBlockToMessages.js  # Merge blocks into messages
│   │   │   └── mergeBlock.js          # Block-level merge logic
│   │   ├── ensure/         # Session/Group ID guards
│   │   ├── format.jsx      # Formatting utilities
│   │   └── imagePath.jsx   # Image path helpers
│   ├── App.jsx             # Root app with Outlet + Toaster + VideoModal
│   ├── main.jsx            # Entry point (React 18 createRoot)
│   └── index.css           # Global styles + Tailwind directives
├── .env                    # Environment variables
├── vite.config.js
├── jsconfig.json
├── eslint.config.js
└── package.json
```

### Key Architectural Concepts

#### 1. Block-Based Chat System
Messages are not plain strings. Each message contains an array of **blocks** with discriminated types:
- `text` — Markdown-rendered text with syntax highlighting
- `image` — Gallery of search result images
- `video` — Video segments with thumbnails and timestamps
- `thinking` — Agent reasoning steps (collapsible)
- `tools` — Tool call steps with pending/finished status

The `BlockRenderer` component dynamically renders each block type with memoized equality checks for performance.

#### 2. Real-Time Streaming via Socket.IO
- The frontend connects to the backend via Socket.IO at `PRIMARY_URL`.
- Key socket events handled:
  - `message_received` — User message acknowledged; shows loading state
  - `response` — Text delta streaming
  - `thinking` — Agent thinking steps
  - `media` — Image or video search results
  - `tool_call` / `tool_result` — Tool invocation status
  - `stream_end` — Streaming finished
  - `continue_stream` — Resume in-progress stream on reconnect
- The `addBlockToMessages` utility merges consecutive blocks of the same type to minimize re-renders.

#### 3. State Management (Zustand)
- **`user` store:** Holds `user` object and JWT `token`. Persisted to `localStorage`.
- **`chat` store:** Holds `session_id`, `chatMessages`, `chatHistory`, `workspaceVideos`, `currentGroup`, `sidebarOpen`. Partially persisted (only `session_id` and `currentGroup`).
- **`videoModal` store:** Simple modal state for video previews.
- **`modal` store:** Generic keyed modal system for extensibility.

#### 4. Server State (React Query)
Data fetching is centralized in `src/api/services/hooks/query.jsx`:
- `useVideos(groupId, sessionId)` — Fetch videos in a group with selection state
- `useGroups()` — Fetch user groups
- `useCreateNewChat()` — Create a new chat session
- `useDeleteSession()` — Delete a session with fallback logic
- `useCreateGroup()` / `useDeleteGroup()` / `useRenameGroup()` — Group CRUD
- `useRenameVideo()` — Rename a video
- `useSearchChatHistory(searchTerm)` — Search across chat history

All mutations invalidate relevant query keys to keep the UI in sync.

#### 5. Authentication Flow
1. User clicks "Continue with Google" on `Login.jsx`.
2. `@react-oauth/google` initiates OAuth flow with `flow: 'auth-code'`.
3. Authorization code is sent to backend `POST /api/user/login/google`.
4. Backend exchanges code for tokens, verifies Google ID token, creates/updates user, and returns a JWT.
5. Frontend stores JWT in `user` store (persisted to `localStorage`).
6. Axios interceptor (`api.jsx`) attaches `Authorization: Bearer <token>` to every request.

#### 6. Routing
- `createBrowserRouter` from React Router v7.
- Public routes: `/` (Home), `/login`, `*`
- Private routes: `/private`
- `App.jsx` wraps everything with `Outlet`, global `Toaster`, and `VideoModal`.

---

## Key Features

- **Google OAuth Sign-In:** Secure login with JWT session management
- **Video Upload & Management:** Upload videos to groups, track ingestion progress
- **AI Chat with Streaming:** Real-time streaming responses with text, thinking, and tool blocks
- **Video Moment Retrieval:** AI finds specific moments in videos and returns playable segments
- **Image Search Results:** Visual search results with gallery layout
- **Responsive Sidebar:** Collapsible on mobile, fixed on desktop
- **Persistent Sessions:** Chat history and session state survive page reloads
- **Keyboard Shortcuts:** Typing any alphanumeric key focuses the chat input
- **Auto-Scroll:** Smart scroll-to-bottom only when user is already near the bottom
- **Syntax Highlighting:** Code blocks with copy-to-clipboard and language labels
- **Video Player:** Video.js integration with segment markers and thumbnail previews

---

## Notes

- The app uses **React 19** with StrictMode disabled in `main.jsx` (commented out) to avoid double-effect issues with Socket.IO.
- The backend AI service WebSocket is expected at `ws://localhost:8080/ws/start_workflow` (configurable in `backend/app/api/socket.py`).
- Uploads are handled by the backend and stored in MinIO; the frontend only deals with video URLs and metadata.
