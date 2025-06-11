# 📁 Project Structure

## Complete Directory Tree

```
Multimodal-Video-Analysis/
│
├── 📚 Documentation
│   ├── README.md                    # Full documentation
│   ├── QUICK_START.md              # Quick setup guide
│   ├── SETUP.md                    # Detailed setup instructions
│   ├── MIGRATION_SUMMARY.md        # Migration details
│   └── PROJECT_STRUCTURE.md        # This file
│
├── 🔧 Scripts
│   ├── start.sh                    # Start both servers
│   └── verify_setup.sh             # Verify installation
│
├── 🐍 Backend (Flask API)
│   ├── app/
│   │   ├── __init__.py             # Flask app factory with SocketIO
│   │   │
│   │   ├── routes/                 # API Endpoints
│   │   │   ├── video.py           # POST /api/video/process, GET /api/video/{id}
│   │   │   ├── chat.py            # POST /api/chat/
│   │   │   └── search.py          # POST /api/search/
│   │   │
│   │   ├── services/               # Business Logic
│   │   │   ├── video_processor.py # AI video processing (GPT-4, CLIP)
│   │   │   ├── chat_service.py    # RAG-based chat with citations
│   │   │   └── search_service.py  # Visual + text search
│   │   │
│   │   ├── models/                 # Data Models
│   │   │   └── __init__.py        # VideoSection, ProcessingResult
│   │   │
│   │   └── utils/                  # Utilities
│   │       ├── cache.py           # Redis + in-memory caching
│   │       └── youtube.py         # YouTube download & transcript
│   │
│   ├── config.py                   # Configuration settings
│   ├── app.py                      # Entry point
│   ├── requirements.txt            # Python dependencies
│   ├── .env.example               # Environment template
│   ├── uploads/                    # Downloaded videos (gitignored)
│   └── temp/                       # Embeddings storage (gitignored)
│
├── ⚛️ Frontend (Next.js)
│   ├── app/                        # Next.js App Router
│   │   ├── layout.tsx             # Root layout
│   │   ├── page.tsx               # Main application page
│   │   └── globals.css            # Global styles
│   │
│   ├── components/                 # React Components
│   │   ├── VideoPlayer.tsx        # YouTube player with controls
│   │   ├── VideoSections.tsx      # Section navigation
│   │   ├── ChatInterface.tsx      # Chat UI with citations
│   │   └── SearchInterface.tsx    # Search UI (visual/text)
│   │
│   ├── lib/                        # Utilities
│   │   ├── api.ts                 # API client (axios)
│   │   └── utils.ts               # Helper functions
│   │
│   ├── types/                      # TypeScript Definitions
│   │   └── index.ts               # All type definitions
│   │
│   ├── package.json                # Node dependencies
│   ├── tsconfig.json              # TypeScript config
│   ├── tailwind.config.ts         # Tailwind CSS config
│   ├── next.config.mjs            # Next.js config
│   ├── .env.local.example         # Environment template
│   └── node_modules/              # Dependencies (gitignored)
│
├── 🗄️ Legacy Files (backup)
│   └── legacy_backup/              # Old FastAPI implementation
│       ├── main.py
│       ├── video_processor.py
│       ├── static/
│       ├── templates/
│       └── ...
│
└── ⚙️ Configuration
    ├── .env                        # Environment variables (gitignored)
    ├── .env.template              # Environment template
    ├── .gitignore                 # Git ignore rules
    └── .git/                       # Git repository
```

## File Counts

### Backend (Python)
- **Python Files**: 11 files
  - Routes: 3 files (video, chat, search)
  - Services: 3 files (processor, chat, search)
  - Utils: 2 files (cache, youtube)
  - Models: 1 file
  - Config: 1 file
  - Entry: 1 file

### Frontend (TypeScript/React)
- **TypeScript Files**: 12 files
  - Pages: 2 files (layout, page)
  - Components: 4 files (player, sections, chat, search)
  - Utils: 2 files (api, utils)
  - Types: 1 file
  - Config: 3 files (tsconfig, tailwind, next)

### Total Lines of Code
- Backend: ~2,500 lines
- Frontend: ~1,800 lines
- **Total**: ~4,300 lines

## Key Files Explained

### Backend

#### `backend/app.py` (Entry Point)
```python
# Creates Flask app with SocketIO
# Runs on port 5000
```

#### `backend/app/__init__.py` (App Factory)
```python
# Flask application factory pattern
# Registers blueprints (routes)
# Initializes CORS and SocketIO
```

#### `backend/app/services/video_processor.py` (Core Logic)
```python
# VideoProcessor class
# - Downloads YouTube videos
# - Extracts frames with OpenCV
# - Generates sections with GPT-4
# - Creates embeddings with CLIP
# - Stores in Redis cache
```

#### `backend/app/services/chat_service.py` (RAG Chat)
```python
# ChatService class
# - Builds context from transcript
# - Calls GPT-4 for responses
# - Extracts timestamp citations
# - Returns formatted response
```

#### `backend/app/services/search_service.py` (Search)
```python
# SearchService class
# - Text search via keywords
# - Visual search via CLIP embeddings
# - Combines results by relevance
```

### Frontend

#### `frontend/app/page.tsx` (Main Page)
```typescript
// Main application UI
// - URL input form
// - Video player integration
// - Sections, chat, search panels
// - WebSocket connection
// - State management
```

#### `frontend/components/VideoPlayer.tsx`
```typescript
// React Player wrapper
// - Custom controls
// - Seek functionality
// - Time tracking
// - Volume control
```

#### `frontend/components/ChatInterface.tsx`
```typescript
// Chat UI
// - Message display
// - Input handling
// - Citation rendering
// - Auto-scroll
```

#### `frontend/lib/api.ts` (API Client)
```typescript
// Axios-based API client
// - videoApi.processVideo()
// - chatApi.sendMessage()
// - searchApi.search()
```

## Data Flow

### Video Processing Flow
```
1. User enters YouTube URL
   ↓
2. Frontend → POST /api/video/process
   ↓
3. Backend starts background processing
   ↓
4. WebSocket updates sent to frontend
   ↓
5. Process steps:
   - Download video (yt-dlp)
   - Extract transcript (YouTube API)
   - Generate sections (GPT-4)
   - Extract frames (OpenCV)
   - Create embeddings (CLIP + Sentence Transformers)
   - Store in Redis
   ↓
6. WebSocket: "completed" with data
   ↓
7. Frontend displays video interface
```

### Chat Flow
```
1. User types question
   ↓
2. Frontend → POST /api/chat/
   ↓
3. Backend:
   - Retrieves video data from cache
   - Searches transcript for relevant segments
   - Builds context for GPT-4
   - Gets response with citations
   ↓
4. Response returned to frontend
   ↓
5. Frontend displays message + clickable citations
```

### Search Flow
```
1. User enters search query
   ↓
2. Frontend → POST /api/search/
   ↓
3. Backend:
   - Text search: keyword matching in transcript
   - Visual search: CLIP embedding similarity
   - Combine and rank results
   ↓
4. Results returned to frontend
   ↓
5. Frontend displays clickable results
```

## Technology Stack Breakdown

### Backend Dependencies (requirements.txt)
```
Core Framework:
- Flask 3.0.0              # Web framework
- flask-cors 4.0.0         # CORS support
- flask-socketio 5.3.5     # WebSocket support

AI/ML:
- openai 1.10.0           # GPT-4 + CLIP
- sentence-transformers    # Text embeddings
- torch 2.1.2             # PyTorch
- transformers 4.36.2     # CLIP models

Video Processing:
- yt-dlp 2024.1.0         # YouTube download
- opencv-python 4.9.0     # Frame extraction
- youtube-transcript-api   # Transcript extraction

Caching:
- redis 5.0.1             # Cache layer
- chromadb 0.4.22         # Vector store

Utilities:
- pydantic 2.5.3          # Data validation
- numpy 1.26.3            # Numerical operations
```

### Frontend Dependencies (package.json)
```
Framework:
- next 14.2.0             # React framework
- react 18.3.0            # UI library
- typescript 5.x          # Type safety

UI:
- tailwindcss 3.4.0       # Styling
- lucide-react 0.344.0    # Icons
- clsx 2.1.0              # Class utilities

API:
- axios 1.6.0             # HTTP client
- socket.io-client 4.7.0  # WebSocket

Video:
- react-player 2.16.0     # YouTube player
```

## Environment Variables

### Backend (.env)
```env
# Required
OPENAI_API_KEY=sk-...              # OpenAI API key
SECRET_KEY=random-string           # Flask secret

# Optional
REDIS_URL=redis://localhost:6379/0  # Redis connection
GOOGLE_API_KEY=...                 # Future use
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000    # Backend URL
NEXT_PUBLIC_WS_URL=http://localhost:5000     # WebSocket URL
```

## Git Ignore Patterns

```gitignore
# Python
__pycache__/
*.pyc
venv/
.env

# Node
node_modules/
.next/
.env.local

# Generated
uploads/      # Downloaded videos
temp/         # Embeddings
*.mp4
*.npz

# Legacy
legacy_backup/
```

## Port Configuration

| Service | Port | URL |
|---------|------|-----|
| Backend API | 5000 | http://localhost:5000 |
| Frontend | 3000 | http://localhost:3000 |
| Redis (optional) | 6379 | redis://localhost:6379 |

## API Endpoints Summary

```
Video Processing:
POST   /api/video/process          Process YouTube URL
GET    /api/video/{id}             Get video data
GET    /api/video/{id}/status      Check processing status
DELETE /api/video/{id}             Delete video data

Chat:
POST   /api/chat/                  Send chat message

Search:
POST   /api/search/                Search video content

Utility:
GET    /health                     Health check

WebSocket:
Event  processing_status           Real-time updates
```

## Component Hierarchy

```
Frontend Component Tree:
└── RootLayout (layout.tsx)
    └── Home (page.tsx)
        ├── URLInput (form)
        ├── VideoPlayer
        ├── VideoSections
        │   └── SectionCard (mapped)
        ├── ChatInterface
        │   ├── MessageList
        │   └── MessageInput
        └── SearchInterface
            ├── SearchForm
            └── ResultsList
                └── ResultCard (mapped)
```

## Database Schema (Cache)

```
Redis Keys:
video:{video_id}                   # Video data (JSON)
video:{video_id}:embeddings        # Not used (stored as .npz)
video:{video_id}:error             # Error messages

File Storage:
uploads/{video_id}.mp4             # Downloaded video
temp/{video_id}_embeddings.npz     # Embeddings (numpy)
```

---

**Last Updated**: 2025-11-19
**Version**: 2.0 (Complete Reconstruction)
