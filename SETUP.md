# Quick Setup Guide

## Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.9 or higher
- ✅ Node.js 18 or higher
- ✅ OpenAI API key
- ✅ (Optional) Redis installed

## Quick Start

### 1. Backend Setup (5 minutes)

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Frontend Setup (3 minutes)

```bash
# Navigate to frontend (in a new terminal)
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local
# No changes needed for local development
```

### 3. Run the Application

**Option A: Using the startup script**
```bash
./start.sh
```

**Option B: Manually (recommended for development)**

Terminal 1 (Backend):
```bash
cd backend
source venv/bin/activate
python app.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

### 4. Access the Application

Open your browser and go to: **http://localhost:3000**

## Testing the System

1. Enter a YouTube URL (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
2. Click "Analyze Video"
3. Wait for processing (usually 30-60 seconds)
4. Explore:
   - Video sections
   - Chat with the video
   - Search for content

## Common Issues

### "ModuleNotFoundError" in Backend
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### "Cannot find module" in Frontend
```bash
cd frontend
npm install
```

### "OPENAI_API_KEY not found"
Edit `backend/.env` and add:
```
OPENAI_API_KEY=sk-...your-key-here
```

### Port already in use
If port 5000 or 3000 is busy:
- Backend: Change port in `backend/app.py`
- Frontend: Run `npm run dev -- -p 3001`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [API documentation](#) for endpoint details
- See [architecture diagrams](#) for system design

## Need Help?

- Check backend logs in the terminal running Flask
- Check browser console for frontend errors
- Ensure all environment variables are set correctly
