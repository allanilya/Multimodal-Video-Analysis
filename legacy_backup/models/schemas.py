from pydantic import BaseModel

class VideoURL(BaseModel):
    url: str

class ChatMessage(BaseModel):
    video_id: str
    message: str

class SearchQuery(BaseModel):
    video_id: str
    query: str 