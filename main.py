import os
import json
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, Request, HTTPException
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

# Initialize  video processor
try:
    video_processor = VideoProcessor()
    print("✅ Video Processor initialized successfully")
except Exception as e:
    print(f"❌ Error initializing video processor: {str(e)}")
    video_processor = None

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
    if not video_processor:
        return JSONResponse({
            "status": "error",
            "message": "Video processor not initialized. Check your API keys."
        }, status_code=500)
    
    try:
        # Extract video ID
        video_id = video_processor.extract_video_id(video_data.url)
        print(f"🎬 Processing video: {video_id}")
        
        # Check if already processed
        if video_id in processed_videos:
            print("✅ Video already processed, returning cached data")
            return JSONResponse({
                "status": "success",
                "message": "Video already processed",
                "video_id": video_id,
                "video_data": processed_videos[video_id]["metadata"],
                "sections": processed_videos[video_id]["analysis"]["sections"]
            })
        
        # Step 1: Upload/analyze video with Gemini
        print("📤 Uploading video to Gemini for analysis...")
        video_metadata = video_processor.upload_video_to_gemini(video_data.url)
        print(f"✅ Video metadata obtained: {video_metadata['title']}")
        
        # Step 2: Generate comprehensive analysis
        print("🔍 Generating comprehensive video analysis...")
        analysis = video_processor.generate_comprehensive_analysis(video_metadata)
        print(f"✅ Analysis complete - Found {len(analysis.get('sections', []))} sections")
        
        # Store processed data
        processed_videos[video_id] = {
            "metadata": video_metadata,
            "analysis": analysis
        }
        
        # Format sections for frontend
        sections = []
        for section in analysis.get("sections", []):
            sections.append({
                "title": section.get("title", ""),
                "summary": section.get("description", ""),
                "start_time": section.get("start_time", 0),
                "end_time": section.get("start_time", 0) + 60  # Estimate end time
            })
        
        print(f"🎉 Video processing completed successfully: {video_id}")
        return JSONResponse({
            "status": "success",
            "message": "Video processed successfully",
            "video_id": video_id,
            "video_data": video_metadata,
            "sections": sections,
            "transcript_available": bool(analysis.get("transcript")),
            "visual_analysis_available": bool(analysis.get("visual_content"))
        })
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "message": f"Failed to process video: {str(e)}"
        }, status_code=500)

@app.post("/api/chat")
async def chat_with_video(chat_data: ChatMessage):
    try:
        if chat_data.video_id not in processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        
        video_data = processed_videos[chat_data.video_id]
        
        print(f"💬 Chat question: {chat_data.message}")
        
        # Get chat response using processor
        response = video_processor.chat_with_video(
            chat_data.message,
            video_data["metadata"],
            video_data["analysis"]
        )
        
        print(f"✅ Chat response generated (used {response.get('context_used', 0)} context sources)")
        
        return JSONResponse({
            "status": "success",
            "response": {
                "answer": response["answer"],
                "citations": [],  # Will add citations based on analysis
                "analysis_available": response.get("analysis_available", False)
            }
        })
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
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
        
        print(f"🔍 Search query: {search_data.query}")
        
        # Search using processor
        results = video_processor.search_content(
            search_data.query,
            video_data["analysis"]
        )
        
        print(f"✅ Found {len(results)} search results")
        
        return JSONResponse({
            "status": "success",
            "results": results
        })
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
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
        "analysis": video_data["analysis"]
    })

@app.get("/api/transcript/{video_id}")
async def get_transcript(video_id: str):
    if video_id not in processed_videos:
        raise HTTPException(status_code=404, detail="Video not found")
    
    analysis = processed_videos[video_id]["analysis"]
    transcript = analysis.get("transcript", [])
    
    return JSONResponse({
        "status": "success",
        "transcript": transcript,
        "available": bool(transcript)
    })

@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "video_processor": "initialized" if video_processor else "failed",
        "processed_videos": len(processed_videos)
    })

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Multimodal Video Analysis System...")
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"), 
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "False").lower() == "true"
    )