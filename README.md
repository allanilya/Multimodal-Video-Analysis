# Multimodal Video Analysis

A comprehensive AI-powered system that allows users to upload YouTube videos and interact with them through natural language chat, timestamped navigation, and visual content search.

## Features

- **Video Processing**: Analyze YouTube videos with AI-powered insights
- **Section Breakdown**: Automatically generate timestamped sections with summaries using GPT-4
- **Interactive Chat**: Ask questions about video content with cited responses and RAG
- **Visual Search**: Search for specific visual content within the video using CLIP embeddings
- **Timestamp Navigation**: Click on sections or citations to jump to specific moments
- **Modern UI**: Responsive web interface with real-time updates via WebSocket

## Tech Stack

### Backend
- **Flask**: REST API and WebSocket server
- **OpenAI GPT-4**: Chat responses and section generation
- **OpenAI CLIP**: Visual embeddings for image search
- **Sentence Transformers**: Text embeddings for semantic search
- **Redis**: Caching layer (optional, falls back to in-memory)
- **yt-dlp**: YouTube video downloading
- **OpenCV**: Video frame extraction

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Socket.IO**: Real-time processing updates
- **React Player**: YouTube video playback

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── __init__.py          # Flask app factory
│   │   ├── routes/              # API endpoints
│   │   │   ├── video.py         # Video processing routes
│   │   │   ├── chat.py          # Chat routes
│   │   │   └── search.py        # Search routes
│   │   ├── services/            # Business logic
│   │   │   ├── video_processor.py   # Video processing with AI
│   │   │   ├── chat_service.py      # Chat with RAG
│   │   │   └── search_service.py    # Visual & text search
│   │   ├── models/              # Data models
│   │   └── utils/               # Helper functions
│   ├── config.py                # Configuration
│   ├── app.py                   # Entry point
│   └── requirements.txt         # Python dependencies
│
├── frontend/
│   ├── app/                     # Next.js app router
│   │   ├── page.tsx            # Main page
│   │   ├── layout.tsx          # Root layout
│   │   └── globals.css         # Global styles
│   ├── components/             # React components
│   │   ├── VideoPlayer.tsx     # Video player with controls
│   │   ├── VideoSections.tsx   # Section navigation
│   │   ├── ChatInterface.tsx   # Chat UI
│   │   └── SearchInterface.tsx # Search UI
│   ├── lib/                    # Utilities
│   │   ├── api.ts             # API client
│   │   └── utils.ts           # Helper functions
│   ├── types/                  # TypeScript types
│   └── package.json           # Node dependencies
│
└── README.md
```

## Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- Redis (optional but recommended)
- OpenAI API key

### Quick Setup

1. **Environment Variables** (in root directory):
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

2. **Backend**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

3. **Frontend**:
```bash
cd frontend
npm install
cd ..
```

4. **Run the Application**:
```bash
./start.sh
# Or manually: backend in one terminal, frontend in another
```

See [QUICK_START.md](QUICK_START.md) for detailed setup instructions.

## Usage

1. **Start both servers** (backend and frontend)

2. **Open the app** in your browser at `http://localhost:3000`

3. **Enter a YouTube URL** in the input field

4. **Wait for processing** - The system will:
   - Download the video
   - Extract frames
   - Generate embeddings
   - Create AI-powered sections
   - Extract transcript

5. **Interact with the video**:
   - Click sections to navigate
   - Ask questions in the chat
   - Search for visual or text content
   - Jump to specific timestamps via citations

## API Endpoints

### Video Processing
- `POST /api/video/process` - Process a YouTube video
- `GET /api/video/{video_id}` - Get video information
- `GET /api/video/{video_id}/status` - Check processing status
- `DELETE /api/video/{video_id}` - Delete video data

### Chat
- `POST /api/chat/` - Send a chat message

### Search
- `POST /api/search/` - Search video content

### WebSocket Events
- `processing_status` - Real-time processing updates

## Features in Detail

### AI-Powered Section Generation
- Uses GPT-4 to analyze video transcripts
- Automatically creates 3-6 meaningful sections
- Each section includes:
  - Timestamp range
  - Descriptive title
  - Summary
  - Keywords

### Interactive Chat with RAG
- Retrieval-Augmented Generation using video transcript
- GPT-4 powered responses
- Automatic timestamp citations
- Context-aware answers

### Visual Search with CLIP
- CLIP embeddings for frame similarity
- Text-to-image search capabilities
- Combined visual + text search
- Ranked results by relevance

### Real-time Updates
- WebSocket connection for processing status
- Live progress updates
- Instant notifications on completion

## Performance Optimizations

- **Caching**: Redis for fast video data retrieval
- **Frame Sampling**: Intelligent frame extraction (up to 100 frames)
- **Batch Processing**: Efficient embedding generation
- **Async Operations**: Non-blocking video processing
- **GPU Support**: Automatic GPU detection for faster inference

## Development

### Backend Development
```bash
cd backend
source venv/bin/activate
python app.py
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Building for Production

Backend:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Frontend:
```bash
npm run build
npm start
```

## Troubleshooting

### Redis Connection Issues
The app works without Redis (falls back to in-memory cache). To use Redis:
```bash
# Install Redis
brew install redis  # macOS
sudo apt-get install redis  # Ubuntu

# Start Redis
redis-server
```

### GPU Support
For GPU acceleration:
```bash
# Install CUDA version of dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Video Download Issues
If yt-dlp fails:
```bash
# Update yt-dlp
pip install --upgrade yt-dlp
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- OpenAI for GPT-4 and CLIP
- Hugging Face for Transformers
- yt-dlp for YouTube downloading
- Next.js and Flask teams
