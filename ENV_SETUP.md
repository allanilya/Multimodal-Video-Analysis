# Environment Variables Guide

## Single .env File (Recommended)

This project uses a **single `.env` file in the root directory** that both backend and frontend read from. This is cleaner and easier to manage.

## Location

```
Multimodal-Video-Analysis/
├── .env                    # ← Single env file (gitignored)
├── .env.example            # ← Template to copy
├── backend/
└── frontend/
```

## Setup

```bash
# Copy the example file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

## Required Variables

### OpenAI API Key (Required)
```env
OPENAI_API_KEY=sk-proj-...
```
- Get your key from: https://platform.openai.com/api-keys
- Used for: GPT-4 chat, section generation, embeddings

### Flask Secret Key (Required)
```env
SECRET_KEY=your-random-secret-key-here
```
- Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`
- Used for: Flask session security

## Optional Variables

### Redis Configuration
```env
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379
```
- App works without Redis (uses in-memory cache)
- Redis provides persistent caching between restarts

### Video Processing
```env
MAX_VIDEO_DURATION=3600      # Max 1 hour videos
FRAME_SAMPLE_RATE=1          # 1 frame per second
MAX_FRAMES=100               # Max 100 frames extracted
BATCH_SIZE=32                # Embedding batch size
```

### Development
```env
FLASK_ENV=development
BACKEND_PORT=5000
FRONTEND_PORT=3000
CORS_ORIGINS=http://localhost:3000
```

## Complete .env Example

```env
# API Keys (Required)
OPENAI_API_KEY=sk-proj-your-key-here
SECRET_KEY=a-very-long-random-secret-key

# Flask
FLASK_ENV=development

# Redis (Optional)
REDIS_URL=redis://localhost:6379/0
REDIS_HOST=localhost
REDIS_PORT=6379

# Video Processing (Optional)
MAX_VIDEO_DURATION=3600
FRAME_SAMPLE_RATE=1
BATCH_SIZE=32
MAX_FRAMES=100

# CORS (Optional)
CORS_ORIGINS=http://localhost:3000

# Ports (Optional)
BACKEND_PORT=5000
FRONTEND_PORT=3000
```

## How It Works

### Backend
The backend reads `.env` from the project root:
```python
# backend/config.py
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
```

### Frontend
Next.js automatically reads `.env` from the root directory. No configuration needed!

Variables prefixed with `NEXT_PUBLIC_` are exposed to the browser:
```env
# These would be in .env if frontend needed separate config
NEXT_PUBLIC_API_URL=http://localhost:5000
NEXT_PUBLIC_WS_URL=http://localhost:5000
```

**Note**: Currently the frontend uses hardcoded URLs. If you need to customize them, add these variables to `.env`.

## Security Notes

⚠️ **Never commit `.env` to git!**
- Already in `.gitignore`
- Contains sensitive API keys
- Use `.env.example` for sharing template

✅ **Best Practices:**
- Use strong random strings for `SECRET_KEY`
- Rotate API keys periodically
- Different `.env` for development/production
- Never share your `.env` file

## Troubleshooting

### "OPENAI_API_KEY not found"
```bash
# Check if .env exists in root
ls -la .env

# Check if it contains the key
grep OPENAI_API_KEY .env

# Make sure there are no spaces around =
# ✓ OPENAI_API_KEY=sk-...
# ✗ OPENAI_API_KEY = sk-...
```

### "Redis connection failed"
Redis is optional. The app will automatically fall back to in-memory cache.

To install Redis (optional):
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu
sudo apt-get install redis
sudo service redis-server start
```

### Backend can't find .env
The backend looks for `.env` in the project root. Make sure:
```bash
# You're in the project root when running
cd /path/to/Multimodal-Video-Analysis

# .env is in root, not in backend/
ls .env  # Should exist
ls backend/.env  # Should NOT exist
```

## Migration from Old Setup

If you had separate `.env` files:

```bash
# Old structure (DON'T USE)
backend/.env
frontend/.env.local

# New structure (USE THIS)
.env  # Single file in root
```

Delete old env files:
```bash
rm backend/.env 2>/dev/null
rm frontend/.env.local 2>/dev/null
```

## Quick Check

Verify your setup:
```bash
# Run the verification script
./verify_setup.sh

# It will check:
# ✓ .env exists in root
# ✓ OPENAI_API_KEY is set
# ✓ No old env files exist
```

---

**Need help?** Check the main [README.md](README.md) or [QUICK_START.md](QUICK_START.md)
