from fastapi.responses import JSONResponse
from fastapi import HTTPException
from video_processor import VideoProcessor
from models.schemas import VideoURL, ChatMessage, SearchQuery

class VideoService:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.processed_videos = {}

    async def process_video(self, video_data: VideoURL):
        video_id = self.video_processor.extract_video_id(video_data.url)
        if video_id in self.processed_videos:
            return JSONResponse({
                "status": "success",
                "message": "Video already processed",
                "video_id": video_id,
                "video_data": self.processed_videos[video_id]["metadata"]
            })
        try:
            video_metadata = self.video_processor.download_youtube_video(video_data.url)
            frames_data = self.video_processor.extract_frames(video_metadata["video_path"])
            # Only use Whisper for transcript
            transcript = self.video_processor.get_transcript(video_id, video_metadata["video_path"])
            sections = self.video_processor.generate_section_breakdown(transcript, video_metadata)
            embeddings_data = self.video_processor.create_embeddings(sections, frames_data)
            self.processed_videos[video_id] = {
                "metadata": video_metadata,
                "sections": sections,
                "frames_data": frames_data,
                "embeddings_data": embeddings_data
            }
            return JSONResponse({
                "status": "success",
                "message": "Video processed successfully",
                "video_id": video_id,
                "video_data": video_metadata,
                "sections": sections
            })
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": str(e)
            }, status_code=500)

    async def chat_with_video(self, chat_data: ChatMessage):
        if chat_data.video_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        video_data = self.processed_videos[chat_data.video_id]
        response = self.video_processor.chat_with_video(
            chat_data.message,
            video_data["metadata"],
            video_data["embeddings_data"]
        )
        return JSONResponse({
            "status": "success",
            "response": response
        })

    async def search_video_content(self, search_data: SearchQuery):
        if search_data.video_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        video_data = self.processed_videos[search_data.video_id]
        results = self.video_processor.search_content(
            search_data.query,
            video_data["embeddings_data"],
            k=10
        )
        return JSONResponse({
            "status": "success",
            "results": results
        })

    async def get_video_data(self, video_id: str):
        if video_id not in self.processed_videos:
            raise HTTPException(status_code=404, detail="Video not found")
        video_data = self.processed_videos[video_id]
        return JSONResponse({
            "status": "success",
            "video_data": video_data["metadata"],
            "sections": video_data["sections"]
        }) 