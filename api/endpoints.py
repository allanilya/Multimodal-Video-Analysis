from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from services.video_service import VideoService
from models.schemas import VideoURL, ChatMessage, SearchQuery

router = APIRouter()
templates = Jinja2Templates(directory="templates")
video_service = VideoService()

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/api/process-video")
async def process_video(video_data: VideoURL):
    return await video_service.process_video(video_data)

@router.post("/api/chat")
async def chat_with_video(chat_data: ChatMessage):
    return await video_service.chat_with_video(chat_data)

@router.post("/api/search")
async def search_video_content(search_data: SearchQuery):
    return await video_service.search_video_content(search_data)

@router.get("/api/video/{video_id}")
async def get_video_data(video_id: str):
    return await video_service.get_video_data(video_id)

@router.get("/api/debug/rag/{video_id}")
async def debug_rag(video_id: str):
    # Access the processed_videos dict from the video service
    video_data = video_service.processed_videos.get(video_id)
    if not video_data:
        raise HTTPException(status_code=404, detail="Video not found")
    # Collect transcript chunks and visual descriptions
    text_chunks = video_data["embeddings_data"].get("text_metadata", [])
    visual_chunks = video_data["embeddings_data"].get("visual_metadata", [])
    return JSONResponse({
        "status": "success",
        "text_chunks": text_chunks,
        "visual_chunks": visual_chunks
    }) 