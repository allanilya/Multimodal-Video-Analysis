# 🚀 Quick Start Guide

## One-Time Setup (10 minutes)

### Step 1: Environment Setup
```bash
# Copy the environment template
cp .env.example .env
```

**Edit `.env` in the root directory and add your OpenAI API key:**
```env
OPENAI_API_KEY=sk-your-key-here
SECRET_KEY=any-random-string
```

### Step 2: Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### Step 3: Frontend Setup
```bash
cd frontend
npm install
cd ..
```

## Running the Application

### Easy Method
```bash
./start.sh
```

### Manual Method
**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Access Points
- 🎨 **Frontend**: http://localhost:3000
- 🔧 **Backend API**: http://localhost:5000
- 📊 **Health Check**: http://localhost:5000/health

## First Test

1. Open http://localhost:3000
2. Paste a YouTube URL (try: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
3. Click "Analyze Video"
4. Wait 30-60 seconds for processing
5. Explore sections, chat, and search!

## Features to Try

### 1. Video Sections
- Auto-generated timestamped sections
- Click any section to jump to that moment
- AI-powered summaries and keywords

### 2. Chat with Video
- Ask questions like "What is the main topic?"
- Get responses with timestamp citations
- Click citations to jump to relevant moments

### 3. Visual Search
- Search for "person wearing red"
- Find specific visual content
- Combined text + visual search

## Common Commands

```bash
# Verify setup
./verify_setup.sh

# Start everything
./start.sh

# Stop everything
Ctrl+C (in terminal)

# Backend only
cd backend && source venv/bin/activate && python app.py

# Frontend only
cd frontend && npm run dev

# Install/Update dependencies
cd backend && pip install -r requirements.txt
cd frontend && npm install
```

## Troubleshooting

### "Port already in use"
```bash
# Change backend port in backend/app.py
# Change frontend port: npm run dev -- -p 3001
```

### "API key error"
Make sure `OPENAI_API_KEY` is set in `backend/.env`

### "Module not found"
```bash
cd backend && source venv/bin/activate && pip install -r requirements.txt
cd frontend && npm install
```

### "Video won't process"
- Check backend terminal for errors
- Verify API key is correct
- Try a different YouTube URL
- Check internet connection

## What's Happening Behind the Scenes

1. **Video Download**: yt-dlp downloads the video
2. **Frame Extraction**: OpenCV extracts key frames
3. **Transcript**: YouTube API gets the transcript
4. **AI Analysis**: GPT-4 creates sections
5. **Embeddings**: CLIP creates visual embeddings
6. **Caching**: Redis stores results for fast retrieval
7. **WebSocket**: Real-time progress updates

## File Structure Quick Reference

```
backend/
  app/
    routes/     → API endpoints
    services/   → Business logic
    utils/      → Helper functions
  app.py        → Start here

frontend/
  app/          → Pages
  components/   → UI components
  lib/          → API client
```

## Need More Help?

- 📖 Full docs: [README.md](README.md)
- 🔧 Setup guide: [SETUP.md](SETUP.md)
- 📝 Migration info: [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md)
- ✅ Verify setup: `./verify_setup.sh`

## Production Deployment

```bash
# Backend
cd backend
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Frontend
cd frontend
npm run build
npm start
```

## Environment Variables

### Backend (.env)
```env
OPENAI_API_KEY=sk-...         # Required
SECRET_KEY=random-string       # Required
REDIS_URL=redis://...         # Optional
```

### Frontend (.env.local)
```env
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=http://localhost:5000
```

---

**That's it! You're ready to analyze videos with AI.** 🎉
