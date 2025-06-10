import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from video_processor import VideoProcessor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Multimodal Video Analysis System")

# Setup directories
static_dir = Path("static")
templates_dir = Path("templates")
static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

# Global storage for processed videos
processed_videos: Dict[str, Dict[str, Any]] = {}

# Initialize video processor
video_processor = VideoProcessor()

class VideoURL(BaseModel):
    url: str

class ChatMessage(BaseModel):
    video_id: str
    message: str

class SearchQuery(BaseModel):
    video_id: str
    query: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process-video")
async def process_video(video_data: VideoURL):
    try:
        # Extract video ID
        video_id = video_processor.extract_video_id(video_data.url)
        print(f"Processing video: {video_id}")
        
        # Check if already processed
        if video_id in processed_videos:
            return JSONResponse({
                "status": "success",
                "message": "Video already processed",
                "video_id": video_id,
                "video_data": processed_videos[video_id]["metadata"]
            })
        
        # Download and process video
        print(f"Downloading video: {video_id}")
        try:
            video_metadata = video_processor.download_youtube_video(video_data.url)
            print(f"Video downloaded successfully: {video_metadata['title']}")
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Failed to download video: {str(e)}"
            }, status_code=500)
        
        # Extract frames
        print("Extracting frames...")
        try:
            frames_data = video_processor.extract_frames(video_metadata["video_path"])
            print(f"Extracted {len(frames_data)} frames")
        except Exception as e:
            print(f"Error extracting frames: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Failed to extract frames: {str(e)}"
            }, status_code=500)
        
        # Generate section breakdown
        print("Generating section breakdown...")
        try:
            sections = video_processor.generate_section_breakdown(
                video_metadata["transcript"], 
                video_metadata
            )
            print(f"Generated {len(sections)} sections")
        except Exception as e:
            print(f"Error generating sections: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Failed to generate sections: {str(e)}"
            }, status_code=500)
        
        # Create embeddings
        print("Creating embeddings...")
        try:
            embeddings_data = video_processor.create_embeddings(sections, frames_data)
            print("Embeddings created successfully")
        except Exception as e:
            print(f"Error creating embeddings: {str(e)}")
            return JSONResponse({
                "status": "error",
                "message": f"Failed to create embeddings: {str(e)}"
            }, status_code=500)
        
        # Store processed data
        processed_videos[video_id] = {
            "metadata": video_metadata,
            "sections": sections,
            "frames_data": frames_data,
            "embeddings_data": embeddings_data
        }
        
        print(f"Video processing completed successfully: {video_id}")
        return JSONResponse({
            "status": "success",
            "message": "Video processed successfully",
            "video_id": video_id,
            "video_data": video_metadata,
            "sections": sections
        })
        
    except Exception as e:
        print(f"Unexpected error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }, status_code=500)

@app.post("/api/chat")
async def chat_with_video(chat_data: ChatMessage):
    try:
        if chat_data.video_id not in processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_data = processed_videos[chat_data.video_id]
        
        # Get chat response
        response = video_processor.chat_with_video(
            chat_data.message,
            video_data["metadata"],
            video_data["embeddings_data"]
        )
        
        return JSONResponse({
            "status": "success",
            "response": response
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.post("/api/search")
async def search_video_content(search_data: SearchQuery):
    try:
        if search_data.video_id not in processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_data = processed_videos[search_data.video_id]
        
        # Search content
        results = video_processor.search_content(
            search_data.query,
            video_data["embeddings_data"],
            k=10
        )
        
        return JSONResponse({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

@app.get("/api/video/{video_id}")
async def get_video_data(video_id: str):
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_data = processed_videos[video_id]
    return JSONResponse({
        "status": "success",
        "video_data": video_data["metadata"],
        "sections": video_data["sections"]
    })

@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    transcript = processed_videos[video_id]["metadata"].get("transcript", [])
    return JSONResponse({
        "status": "success",
        "transcript": transcript
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",  # Import string format
        host=os.getenv("HOST", "127.0.0.1"), 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )