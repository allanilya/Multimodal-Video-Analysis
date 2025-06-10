from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import json
import numpy as np
from typing import List, Dict, Optional
from video_processor import VideoProcessor
import logging
from dotenv import load_dotenv
import uvicorn
import traceback

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store processing results in memory (in production, use a database)
video_data = {}

# Initialize video processor (will be done in lifespan)
video_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global video_processor
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("temp", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    
    # Initialize video processor
    video_processor = VideoProcessor()
    
    logger.info("Application started")
    yield
    # Shutdown
    logger.info("Application shutting down")

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Request models
class ProcessVideoRequest(BaseModel):
    url: str

class ChatRequest(BaseModel):
    video_id: str
    message: str

class SearchRequest(BaseModel):
    video_id: str
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process-video")
async def process_video(request: ProcessVideoRequest):
    """Process a YouTube video."""
    try:
        logger.info(f"Processing video: {request.url}")
        
        # Extract video ID first to validate URL
        video_id = video_processor.extract_video_id(request.url)
        
        # Check if already processed
        if video_id in video_data:
            return {"video_id": video_id, "message": "Video already processed"}
        
        # Process the video
        result = video_processor.process_video(request.url)
        
        # Store results (excluding non-serializable objects)
        video_data[video_id] = {
            "video_id": result["video_id"],
            "transcript": result["transcript"],
            "frame_descriptions": result["frame_descriptions"],
            "sections": result["sections"],
            "metadata": result["metadata"]
        }
        
        # Store embeddings and index separately (not serializable)
        if result["embeddings"] is not None:
            video_data[video_id]["_embeddings"] = result["embeddings"]
            video_data[video_id]["_search_index"] = result["search_index"]
        
        return {
            "video_id": video_id,
            "sections": result["sections"],
            "transcript_available": len(result["transcript"]) > 0,
            "frames_analyzed": len(result["frame_descriptions"])
        }
        
    except ValueError as e:
        logger.error(f"Invalid URL: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        logger.error(traceback.format_exc())
        
        # Provide more specific error messages
        if "HTTP Error 400" in str(e):
            raise HTTPException(
                status_code=500, 
                detail="Failed to download video. This might be due to YouTube restrictions or an outdated downloader. Try updating pytube or using a different video."
            )
        elif "unavailable" in str(e).lower():
            raise HTTPException(
                status_code=400,
                detail="This video is unavailable or private. Please try a different video."
            )
        else:
            raise HTTPException(status_code=500, detail=f"Failed to process video: {str(e)}")

@app.get("/api/video/{video_id}")
async def get_video(video_id: str):
    """Get video data."""
    if video_id not in video_data:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Return serializable data only
    data = {k: v for k, v in video_data[video_id].items() if not k.startswith("_")}
    return data

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Chat about video content."""
    try:
        if request.video_id not in video_data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = video_data[request.video_id]
        
        # Prepare context from transcript and visual descriptions
        context_parts = []
        
        # Add relevant transcript segments
        for segment in video_info["transcript"][:50]:  # Limit context
            context_parts.append(f"[{segment['start']}s] {segment['text']}")
        
        # Add frame descriptions
        for frame in video_info["frame_descriptions"]:
            context_parts.append(f"[{frame['timestamp']}s - Visual] {frame['description']}")
        
        context = "\n".join(context_parts)
        
        # Generate response using OpenAI
        response = video_processor.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant analyzing video content. Always cite timestamps when referencing specific moments."},
                {"role": "user", "content": f"Video context:\n{context}\n\nUser question: {request.message}"}
            ]
        )
        
        return {
            "response": response.choices[0].message.content,
            "citations": []  # You can enhance this to extract timestamps from the response
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search")
async def search(request: SearchRequest):
    """Search for content in video."""
    try:
        if request.video_id not in video_data:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_info = video_data[request.video_id]
        
        # Check if embeddings exist
        if "_embeddings" not in video_info:
            raise HTTPException(status_code=400, detail="Search index not available for this video")
        
        # Create query embedding
        query_embedding = video_processor.sentence_model.encode([request.query])
        
        # Search using FAISS
        k = min(5, len(video_info["metadata"]))  # Return top 5 results
        distances, indices = video_info["_search_index"].search(query_embedding, k)
        
        # Get results
        results = []
        for idx in indices[0]:
            if idx < len(video_info["metadata"]):
                metadata = video_info["metadata"][idx]
                results.append({
                    "timestamp": metadata["timestamp"],
                    "type": metadata["type"],
                    "content": metadata["content"]
                })
        
        return {"results": results}
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run without reload for now to avoid the import string issue
    uvicorn.run(app, host="127.0.0.1", port=8000)