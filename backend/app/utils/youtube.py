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
    def download_video(url: str, output_path: Path, quality: str = 'worst[height<=480][ext=mp4]/worst[ext=mp4]/worst'):
        """Download video using yt-dlp optimized for frame extraction

        Uses lowest quality (480p or less) for faster download since we only need frames.
        Research shows 480p is sufficient for frame extraction and visual analysis.
        """
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info(f"Video already exists: {output_path}")
            return

        ydl_opts = {
            'format': quality,  # Prioritize 480p or lower for speed
            'outtmpl': str(output_path.with_suffix('')),  # Remove .mp4, yt-dlp adds it
            'quiet': True,
            'no_warnings': True,
            'no_playlist': True,
            'preferredcodec': 'mp4',
            'preferredquality': '480p',  # Lower quality = faster download
            'restrictfilenames': True,  # Avoid special characters in filename
        }

        try:
            logger.info(f"Downloading video (optimized for frame extraction)...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            logger.info(f"Video downloaded: {output_path}")
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            raise

    @staticmethod
    def get_transcript(video_id: str) -> List[Dict]:
        """Get video transcript using youtube-transcript-api v1.2+"""
        try:
            # Use new API (v1.2+)
            api = YouTubeTranscriptApi()

            # Try to list available transcripts first
            try:
                available_transcripts = api.list(video_id)
                logger.info(f"Available transcripts for {video_id}: {available_transcripts}")
            except Exception as e:
                logger.warning(f"Could not list transcripts: {e}")

            # Fetch transcript (automatically selects best available)
            result = api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])

            # Convert new format to old format for compatibility
            segments = []
            for snippet in result:
                segments.append({
                    'text': snippet.text,
                    'start': snippet.start,
                    'duration': snippet.duration
                })

            logger.info(f"Successfully fetched {len(segments)} transcript segments")
            return segments

        except Exception as e:
            logger.error(f"Failed to get transcript for {video_id}: {e}", exc_info=True)
            return []

    @staticmethod
    def extract_frame_at_timestamp(url: str, timestamp: float) -> Optional['Image.Image']:
        """Extract a single frame at a specific timestamp by streaming (no full download)

        Uses yt-dlp to seek to specific timestamp and extract just that frame.
        Much faster than downloading entire video.
        """
        import tempfile
        import cv2
        from PIL import Image

        try:
            # Create temporary file for single frame
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name

            # yt-dlp options to extract single frame at timestamp
            ydl_opts = {
                'format': 'worst[height<=480][ext=mp4]/worst[ext=mp4]/worst',  # Low quality for speed
                'quiet': True,
                'no_warnings': True,
                'no_playlist': True,
                'skip_download': True,  # Don't download video
                'postprocessors': [{
                    'key': 'FFmpegThumbnail',
                    'format': 'jpg',
                }],
                'writethumbnail': True,
                'outtmpl': temp_path,
            }

            # Use ffmpeg through yt-dlp to extract frame at specific time
            import subprocess

            # Get video URL with yt-dlp - request a specific video format
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'format': 'worst[ext=mp4]/worst[height>=144]/best[height<=480]'  # Get actual video, not storyboard
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                # Get the direct video URL - filter for actual video formats
                video_url = None
                if 'url' in info and info.get('ext') not in ['jpg', 'png', 'webp']:
                    video_url = info['url']
                elif 'formats' in info and info['formats']:
                    # Find first actual video format (not storyboard/image)
                    for fmt in info['formats']:
                        ext = fmt.get('ext', '')
                        vcodec = fmt.get('vcodec', '')
                        # Skip images/storyboards
                        if ext in ['jpg', 'png', 'webp', 'mhtml']:
                            continue
                        if vcodec == 'none':
                            continue
                        if fmt.get('url'):
                            video_url = fmt['url']
                            logger.info(f"Selected format: {fmt.get('format_id')} ({fmt.get('ext')}, {fmt.get('height')}p)")
                            break

                if not video_url:
                    logger.error(f"Could not extract video URL from info for {url}")
                    return None

            # Use ffmpeg to extract frame at timestamp
            ffmpeg_cmd = [
                'ffmpeg',
                '-ss', str(timestamp),  # Seek to timestamp
                '-i', video_url,        # Input video stream
                '-vframes', '1',        # Extract 1 frame
                '-q:v', '2',           # Quality
                '-y',                  # Overwrite
                temp_path
            ]

            subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30, check=True)

            # Load and return PIL Image
            if Path(temp_path).exists():
                pil_image = Image.open(temp_path)
                pil_image = pil_image.convert('RGB')

                # Resize for efficiency
                pil_image.thumbnail((800, 800), Image.Resampling.LANCZOS)

                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)

                return pil_image

            return None

        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s: {e}")
            return None
