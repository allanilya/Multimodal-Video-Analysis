"""
YouTube utilities for downloading and extracting information
"""
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)

class YouTubeService:
    """Handle YouTube video operations"""

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        raise ValueError("Invalid YouTube URL")

    @staticmethod
    def get_video_info(url: str) -> Dict:
        """Get video metadata using yt-dlp"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'author': info.get('uploader', 'Unknown'),
                    'views': info.get('view_count', 0),
                    'description': (info.get('description') or '')[:500],
                    'thumbnail': info.get('thumbnail', '')
                }
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            raise

    @staticmethod
    def download_video(url: str, output_path: Path, quality: str = 'worst[ext=mp4]/worst'):
        """Download video using yt-dlp"""
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Video already exists: {output_path}")
            return

        ydl_opts = {
            'format': quality,
            'outtmpl': str(output_path),
            'quiet': True,
            'no_warnings': True,
            'no_playlist': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Video downloaded: {output_path}")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise

    @staticmethod
    def get_transcript(video_id: str) -> List[Dict]:
        """Get video transcript"""
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

            # Try to get English transcript first
            try:
                transcript = transcript_list.find_transcript(['en', 'en-US'])
            except:
                # Get any available transcript
                transcript = next(iter(transcript_list))

            return transcript.fetch()
        except Exception as e:
            logger.warning(f"No transcript available for {video_id}: {e}")
            return []
