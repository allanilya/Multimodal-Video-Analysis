"""
Data models for the application
"""
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import timedelta

@dataclass
class VideoSection:
    """Represents a timestamped section of the video"""
    start_time: float
    end_time: float
    title: str
    summary: str
    keywords: List[str]

    def to_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            **asdict(self),
            'start_timestamp': self.to_timestamp(self.start_time),
            'end_timestamp': self.to_timestamp(self.end_time)
        }

@dataclass
class ProcessingResult:
    """Container for video processing results"""
    video_id: str
    title: str
    duration: float
    sections: List[VideoSection]
    transcript: List[Dict]
    processing_time: float
    thumbnail_url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'video_id': self.video_id,
            'title': self.title,
            'duration': self.duration,
            'sections': [s.to_dict() for s in self.sections],
            'transcript': self.transcript,
            'processing_time': self.processing_time,
            'thumbnail_url': self.thumbnail_url,
            'has_transcript': len(self.transcript) > 0
        }

@dataclass
class ChatMessage:
    """Chat message with context"""
    message: str
    video_id: str
    use_visual_context: bool = True

@dataclass
class SearchQuery:
    """Search query"""
    query: str
    video_id: str
    search_type: str = 'both'  # 'visual', 'text', or 'both'
