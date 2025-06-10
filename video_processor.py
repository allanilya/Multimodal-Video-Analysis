# Updated video_processor.py with fixes for the Gemini API changes

import os
import json
import tempfile
from typing import List, Dict, Any
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import google.generativeai as genai
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from moviepy.editor import VideoFileClip
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with API clients and models."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
            elif "embed/" in url:
                return url.split("embed/")[1].split("?")[0]
        raise ValueError("Invalid YouTube URL")
        
    def download_video(self, url: str, output_path: str) -> str:
        """Download video from YouTube using yt-dlp."""
        try:
            # Use yt-dlp instead of pytube for better reliability
            import yt_dlp
            
            video_id = self.extract_video_id(url)
            output_filepath = os.path.join(output_path, f"{video_id}.mp4")
            
            # Check if already downloaded
            if os.path.exists(output_filepath):
                logger.info(f"Video already exists: {output_filepath}")
                return output_filepath
            
            ydl_opts = {
                'outtmpl': output_filepath,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading video: {url}")
                ydl.download([url])
                
            logger.info(f"Video downloaded: {output_filepath}")
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            # Fallback to pytube if yt-dlp fails
            try:
                yt = YouTube(url)
                stream = yt.streams.get_highest_resolution()
                video_path = stream.download(output_path=output_path)
                logger.info(f"Video downloaded with pytube: {video_path}")
                return video_path
            except Exception as pytube_error:
                logger.error(f"Both yt-dlp and pytube failed: {pytube_error}")
                raise

    def extract_transcript(self, video_url: str) -> List[Dict[str, Any]]:
        """Extract transcript from YouTube video."""
        try:
            video_id = self.extract_video_id(video_url)
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            logger.info(f"Transcript extracted: {len(transcript)} segments")
            return transcript
        except Exception as e:
            logger.warning(f"Could not extract transcript: {e}")
            return []

    def extract_frames(self, video_path: str, interval: int = 5) -> List[Dict[str, Any]]:
        """Extract frames from video at specified intervals."""
        frames_data = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
                
            timestamp = frame_count / fps
            if int(timestamp) % interval == 0 and timestamp > 0:
                # Save frame temporarily
                temp_path = f"temp/frame_{int(timestamp)}.jpg"
                cv2.imwrite(temp_path, frame)
                frames_data.append({
                    "timestamp": timestamp,
                    "path": temp_path
                })
                
            frame_count += 1
            
        video.release()
        logger.info(f"Extracted {len(frames_data)} frames")
        return frames_data

    def analyze_frame_with_gemini(self, image_path: str) -> str:
        """Analyze a single frame using Gemini Vision API."""
        try:
            # Upload the image file to Gemini
            uploaded_file = genai.upload_file(image_path)
            
            # Generate content with the uploaded file
            response = self.gemini_model.generate_content([
                "Describe what you see in this image in detail. Focus on objects, people, actions, and the overall scene.",
                uploaded_file
            ])
            
            # Clean up the uploaded file
            genai.delete_file(uploaded_file.name)
            
            return response.text
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return "Could not analyze frame"

    def generate_section_breakdown(self, transcript: List[Dict], frame_descriptions: List[Dict]) -> List[Dict]:
        """Generate video sections using GPT-4."""
        # Prepare context
        transcript_text = "\n".join([f"{t['start']}: {t['text']}" for t in transcript[:50]])  # Limit for context
        
        prompt = f"""
        Analyze this video content and create logical sections with timestamps.
        
        Transcript (first 50 segments):
        {transcript_text}
        
        Create 5-8 main sections that represent the video's structure.
        Return as JSON array with format:
        [
            {{
                "title": "Section Title",
                "start_time": 0,
                "end_time": 120,
                "summary": "Brief description of what happens in this section"
            }}
        ]
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            sections = json.loads(response.choices[0].message.content)
            return sections.get("sections", [])
        except Exception as e:
            logger.error(f"Error generating sections: {e}")
            return []

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text using sentence transformers."""
        embeddings = self.sentence_model.encode(texts)
        return embeddings

    def build_search_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        """Build FAISS index for similarity search."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        return index

    def process_video(self, video_url: str) -> Dict[str, Any]:
        """Main processing pipeline for video analysis."""
        video_id = self.extract_video_id(video_url)
        
        # Create directories
        os.makedirs("uploads", exist_ok=True)
        os.makedirs("temp", exist_ok=True)
        
        # Download video
        video_path = self.download_video(video_url, "uploads")
        
        # Extract transcript
        transcript = self.extract_transcript(video_url)
        
        # Extract frames
        frames_data = self.extract_frames(video_path)
        
        # Analyze frames with Gemini
        frame_descriptions = []
        for frame in frames_data[:10]:  # Limit to first 10 frames for demo
            description = self.analyze_frame_with_gemini(frame["path"])
            frame_descriptions.append({
                "timestamp": frame["timestamp"],
                "description": description
            })
            
        # Generate sections
        sections = self.generate_section_breakdown(transcript, frame_descriptions)
        
        # Create embeddings for search
        all_texts = []
        metadata = []
        
        # Add transcript segments
        for segment in transcript:
            all_texts.append(segment["text"])
            metadata.append({
                "type": "transcript",
                "timestamp": segment["start"],
                "content": segment["text"]
            })
            
        # Add frame descriptions
        for frame_desc in frame_descriptions:
            all_texts.append(frame_desc["description"])
            metadata.append({
                "type": "visual",
                "timestamp": frame_desc["timestamp"],
                "content": frame_desc["description"]
            })
            
        # Build search index
        if all_texts:
            embeddings = self.create_embeddings(all_texts)
            search_index = self.build_search_index(embeddings)
        else:
            search_index = None
            embeddings = None
            
        # Clean up temporary frame files
        for frame in frames_data:
            try:
                os.remove(frame["path"])
            except:
                pass
                
        return {
            "video_id": video_id,
            "video_path": video_path,
            "transcript": transcript,
            "frame_descriptions": frame_descriptions,
            "sections": sections,
            "search_index": search_index,
            "embeddings": embeddings,
            "metadata": metadata
        }

# Example usage:
if __name__ == "__main__":
    processor = VideoProcessor()
    # result = processor.process_video("https://www.youtube.com/watch?v=VIDEO_ID")
