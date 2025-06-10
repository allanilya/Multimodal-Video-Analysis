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
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, InvalidVideoId
from moviepy.editor import VideoFileClip
import logging
from dotenv import load_dotenv
from pathlib import Path
import base64
import whisper
import subprocess
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        """Initialize the video processor with API clients and models."""
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)
        self.processed_videos = {}
        self.openai_client = OpenAI()
        self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        self.whisper_model = whisper.load_model("base")
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

    def extract_chapters_from_description(self, description: str) -> List[Dict[str, Any]]:
        """Extract chapters from video description text."""
        chapters = []
        # Regex to find timestamps like HH:MM:SS or MM:SS at the start of a line
        # and capture the following text as a title.
        # Example: 00:00:00 Introduction or 00:00 Introduction
        chapter_pattern = re.compile(r'^(\d{1,2}:)?\d{1,2}:\d{2}\s+(.*)$', re.MULTILINE)
        
        current_start_time = 0
        matches = list(chapter_pattern.finditer(description))

        for i, match in enumerate(matches):
            timestamp_str = match.group(0).split(' ')[0]
            title = ' '.join(match.group(0).split(' ')[1:]).strip()
            
            # Convert timestamp string to seconds
            parts = list(map(int, timestamp_str.split(':')))
            if len(parts) == 3:
                start_seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
            elif len(parts) == 2:
                start_seconds = parts[0] * 60 + parts[1]
            else:
                continue # Skip invalid timestamp format

            if i > 0:
                # Set end time of previous chapter
                chapters[-1]["end"] = start_seconds
            
            chapters.append({
                "title": title,
                "start": start_seconds,
                "end": 0 # Placeholder, will be filled by next chapter's start or video end
            })
            
        # Set end time for the last chapter as video duration (if available) or an assumed duration
        # This needs to be handled outside this function with actual video duration.
        # For now, we'll leave end time as 0, to be filled later.

        logger.info(f"Extracted {len(chapters)} chapters from description.")
        return chapters

    def extract_chapters(self, video_url: str) -> List[Dict[str, Any]]:
        """Extract chapters from YouTube video using pytube or description parsing."""
        try:
            yt = YouTube(video_url)
            # Try pytube's built-in chapters first
            if hasattr(yt, 'chapters') and yt.chapters:
                chapters = []
                for i, chapter in enumerate(yt.chapters):
                    chapters.append({
                        "title": chapter['title'],
                        "start": chapter['start_time'],
                        "end": chapter['end_time'] if 'end_time' in chapter else (chapter['start_time'] + 30) # Assume 30s if end not present
                    })
                logger.info(f"Successfully extracted {len(chapters)} chapters from pytube.")
                return chapters
            else:
                logger.info(f"No chapters found via pytube for video: {video_url}. Attempting to parse description.")
                # Fallback to description parsing
                if yt.description:
                    chapters_from_desc = self.extract_chapters_from_description(yt.description)
                    if chapters_from_desc:
                        logger.info(f"Successfully extracted {len(chapters_from_desc)} chapters from description.")
                        return chapters_from_desc
                    else:
                        logger.info(f"No chapters found in description for video: {video_url}")
                        return []
                else:
                    logger.info(f"No description available for video: {video_url}. Cannot extract chapters from description.")
                    return []
        except Exception as e:
            logger.warning(f"Error extracting chapters from YouTube (pytube or description parsing): {e}")
            return []

    def extract_transcript(self, video_url: str) -> List[Dict[str, Any]]:
        """Extract transcript from YouTube video with robust error handling and language fallbacks."""
        video_id = self.extract_video_id(video_url)
        
        # Use a flag to decide if we should proceed with Whisper after YouTube attempts
        should_use_whisper = False

        # Attempt 1: Try to fetch auto-generated English directly (most common and desired)
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'], preserve_formatting=True)
            logger.info(f"Successfully fetched auto-generated English transcript: {len(transcript_data)} segments")
            return transcript_data
        except (NoTranscriptFound, TranscriptsDisabled, InvalidVideoId) as e:
            logger.info(f"Auto-generated English transcript not found or disabled for video {video_id}: {e}. Trying other languages.")
        except Exception as e:
            # Catch parsing errors or unexpected issues specifically here
            if "no element found" in str(e).lower():
                logger.warning(f"Parsing error encountered for auto-generated English transcript for {video_id}. Falling back to Whisper: {e}")
                should_use_whisper = True
            else:
                logger.warning(f"Unexpected error when fetching auto-generated English transcript for {video_id}: {e}")
        
        # If a parsing error occurred in attempt 1, skip further YouTube API calls
        if should_use_whisper:
            return [] # Return empty to trigger Whisper fallback

        # Attempt 2: List all available transcripts and try to fetch them systematically
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Ordered preference for fetching:
            # 1. Manually created English
            # 2. Any other English (auto or manual)
            # 3. Any available transcript (fallback)
            
            # Try manually created English first
            try:
                manual_en_transcript = transcript_list.find_manually_created_transcript(language_codes=['en'])
                transcript_data = manual_en_transcript.fetch()
                logger.info(f"Successfully fetched manually created English transcript: {len(transcript_data)} segments")
                return transcript_data
            except (NoTranscriptFound, TranscriptsDisabled, InvalidVideoId) as e:
                logger.info(f"No manually created English transcript found for video {video_id}: {e}. Trying other available English.")
            except Exception as e:
                if "no element found" in str(e).lower():
                    logger.warning(f"Parsing error encountered for manual English transcript for {video_id}. Falling back to Whisper: {e}")
                    should_use_whisper = True
                else:
                    logger.warning(f"Failed to fetch manually created English transcript (unexpected error) for {video_id}: {e}")
            
            if should_use_whisper:
                return []

            # Try finding any English transcript (auto-generated or other locales)
            try:
                any_en_transcript = transcript_list.find_transcript(language_codes=['en', 'en-US', 'en-GB'])
                transcript_data = any_en_transcript.fetch()
                logger.info(f"Successfully fetched an English variant transcript: {len(transcript_data)} segments")
                return transcript_data
            except NoTranscriptFound:
                logger.info("No other English variant transcripts found. Trying any available transcript.")
            except Exception as e:
                if "no element found" in str(e).lower():
                    logger.warning(f"Parsing error encountered for other English variant transcript for {video_id}. Falling back to Whisper: {e}")
                    should_use_whisper = True
                else:
                    logger.warning(f"Failed to fetch other English variant transcript (unexpected error) for {video_id}: {e}")
            
            if should_use_whisper:
                return []

            # Fallback: Loop through all available transcripts and try to fetch them individually
            for t_obj in transcript_list:
                try:
                    transcript_data = t_obj.fetch()
                    logger.info(f"Successfully fetched fallback transcript in {t_obj.language_code}: {len(transcript_data)} segments")
                    return transcript_data
                except Exception as e:
                    if "no element found" in str(e).lower():
                        logger.warning(f"Parsing error encountered in fallback loop for {t_obj.language_code} for {video_id}. Falling back to Whisper: {e}")
                        should_use_whisper = True
                        break # Stop trying other transcripts
                    else:
                        logger.warning(f"Failed to fetch transcript in {t_obj.language_code} (in fallback loop): {e}")
            
            if should_use_whisper:
                return []

            logger.warning(f"No suitable transcript found after exhaustive search for video {video_id}.")
            return []

        except (TranscriptsDisabled, InvalidVideoId) as e:
            logger.warning(f"YouTube transcript API known error for video {video_id}: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unhandled error during YouTube transcript extraction for video {video_id}")
            return []

    def extract_frames(self, video_path: str, interval: int = 30) -> List[Dict[str, Any]]:
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return frames_data
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"Video info: {duration:.1f}s duration, {fps:.1f} fps")
        frame_count = 0
        for i in range(0, int(duration), interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                print(f"Extracted frame at {i:.1f}s ({frame_count})")
                # Convert frame to base64 for Gemini
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_data.append({
                    "timestamp": i,
                    "frame_base64": frame_base64
                })
        cap.release()
        print(f"Extracted {frame_count} frames from video (every {interval}s)")
        return frames_data

    def analyze_frame_with_gemini(self, frame_base64: str, timestamp: float) -> Dict[str, Any]:
        """Analyze a single frame using Gemini."""
        try:
            image_part = {
                "mime_type": "image/jpeg",
                "data": frame_base64
            }
            
            prompt = f"""Analyze this video frame at timestamp {timestamp:.1f}s and provide:
1. A brief title (max 5 words)
2. A detailed description of what's happening
3. Any notable objects, people, or actions
4. The overall mood or atmosphere

Format the response as:
Title: [title]
Description: [description]"""

            response = self.gemini_model.generate_content([prompt, image_part])
            if response and response.text:
                lines = response.text.split('\n')
                title = next((line.replace('Title:', '').strip() for line in lines if 'Title:' in line), 'Unknown Scene')
                description = next((line.replace('Description:', '').strip() for line in lines if 'Description:' in line), 'No description available')
                return {
                    "title": title,
                    "description": description,
                    "timestamp": timestamp
                }
        except Exception as e:
            logger.warning(f"Error analyzing frame with Gemini: {str(e)}")
        return None

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video with optimized settings."""
        try:
            # Use FFmpeg with optimized settings for better audio quality
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '16000',  # 16kHz sample rate (optimal for Whisper)
                '-ac', '1',  # Mono audio
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Audio extracted successfully to {output_path}")
                return True
            else:
                logger.error(f"Error extracting audio: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error in audio extraction: {str(e)}")
            return False

    def whisper_transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio using Whisper with optimized settings."""
        try:
            # Configure Whisper for better performance
            options = {
                "language": "en",  # Force English for better accuracy
                "task": "transcribe",
                "fp16": False,  # Disable FP16 since we're on CPU
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 5,  # Take best of 5 samples
                "temperature": 0.0,  # Disable temperature for more consistent results
                "condition_on_previous_text": True,  # Use previous text for context
                "initial_prompt": "This is a transcription of a video. Please transcribe it accurately."  # Help with context
            }
            
            # Load model with optimized settings (using device="cpu" explicitly)
            model = whisper.load_model("base", device="cpu")
            
            # Transcribe with optimized settings
            result = model.transcribe(
                audio_path,
                **options
            )
            
            # Process and clean up the transcript
            segments = result["segments"]
            for segment in segments:
                # Ensure duration is present
                if "duration" not in segment:
                    segment["duration"] = segment["end"] - segment["start"]
                
                # Clean up text
                segment["text"] = segment["text"].strip()
                
                # Add confidence score if available
                if "confidence" in segment:
                    segment["confidence"] = float(segment["confidence"])
            
            logger.info(f"Whisper transcription complete: {len(segments)} segments.")
            return segments
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {str(e)}")
            return []

    def generate_title_from_text(self, text: str) -> str:
        """Generate a title from text using Gemini."""
        try:
            prompt = f"""Generate a brief, descriptive title (max 5 words) for this text:
            {text[:500]}  # Limit text length for efficiency
            
            Return ONLY the title, nothing else."""
            
            response = self.gemini_model.generate_content(prompt)
            if response and response.text:
                return response.text.strip()
            return "Untitled Section"
        except Exception as e:
            logger.warning(f"Error generating title: {str(e)}")
            return "Untitled Section"

    def generate_section_breakdown(self, transcript: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        sections = []
        video_path = video_metadata["video_path"]
        chapters = video_metadata.get("chapters", []) # Get chapters from metadata

        if chapters:
            logger.info("Generating sections based on YouTube chapters.")
            for chapter in chapters:
                section_start = chapter["start"]
                section_end = chapter["end"]
                section_title = chapter["title"]

                # Get visual description for the chapter (using a frame around the start of the chapter)
                chapter_frames = self.extract_frames(video_path, interval=1, start_time=section_start, end_time=section_start + 1)
                visual_description = ""
                if chapter_frames:
                    analysis = self.analyze_frame_with_gemini(chapter_frames[0]["frame_base64"], chapter_frames[0]["timestamp"])
                    if analysis:
                        visual_description = analysis["description"]

                # Get relevant transcript text for the chapter
                transcript_text_for_chapter = " ".join([s['text'] for s in transcript if s['start'] >= section_start and s['start'] < section_end])

                combined_description = f"Visual: {visual_description}\nAudio: {transcript_text_for_chapter.strip()}"
                if not visual_description:
                    combined_description = f"Audio: {transcript_text_for_chapter.strip()}"
                if not transcript_text_for_chapter:
                    combined_description = f"Visual: {visual_description}"
                if not visual_description and not transcript_text_for_chapter:
                    combined_description = "No specific visual or audio content analyzed for this chapter."

                sections.append({
                    "title": section_title,
                    "start": section_start,
                    "end": section_end,
                    "description": combined_description.strip(),
                    "source": "chapter"
                })
        else:
            # Fallback to previous visual and audio analysis if no chapters
            logger.info("No YouTube chapters available, generating sections from visual and audio analysis.")
            
            # 1. Get visual analysis from Gemini
            logger.info("Generating sections from visual analysis...")
            frames = self.extract_frames(video_path, interval=120)  # 2 minutes
            visual_sections = []
            for frame in frames:
                logger.info(f"[Gemini] Analyzing frame at {frame['timestamp']:.1f}s...")
                analysis = self.analyze_frame_with_gemini(frame["frame_base64"], frame["timestamp"])
                if analysis:
                    visual_sections.append({
                        "title": analysis["title"],
                        "start": frame['timestamp'],
                        "end": frame['timestamp'] + 120,  # Assume 2 minutes per section for visual only
                        "description": analysis["description"],
                        "source": "visual"
                    })

            # 2. Get audio transcript sections
            audio_sections = []
            if transcript:
                logger.info("Processing audio transcript into sections...")
                # Aggregate transcript segments into larger, more meaningful sections
                current_text_block = ""
                current_start = transcript[0]["start"]
                
                # Define a reasonable section length, e.g., 60 seconds or 1000 characters
                max_section_duration = 60 # seconds
                max_section_chars = 1000

                for i, segment in enumerate(transcript):
                    segment_text = segment["text"].strip()
                    segment_duration = segment.get("duration", segment["end"] - segment["start"])
                    
                    # Check if adding this segment would exceed limits or if it's a good natural break
                    if (len(current_text_block) + len(segment_text) > max_section_chars) or \
                       ((segment["start"] - current_start) > max_section_duration and current_text_block):
                        
                        # Generate title using Gemini for the aggregated text block
                        title = self.generate_title_from_text(current_text_block)
                        audio_sections.append({
                            "title": title,
                            "start": current_start,
                            "end": transcript[i-1]["end"] if i > 0 else current_start + segment_duration,
                            "description": current_text_block,
                            "source": "audio"
                        })
                        # Start new section
                        current_text_block = segment_text
                        current_start = segment["start"]
                    else:
                        current_text_block += (" " if current_text_block else "") + segment_text
                
                # Add the last accumulated section
                if current_text_block:
                    title = self.generate_title_from_text(current_text_block)
                    audio_sections.append({
                        "title": title,
                        "start": current_start,
                        "end": transcript[-1]["end"],
                        "description": current_text_block,
                        "source": "audio"
                    })

            # 3. Merge and align sections from visual and audio analysis
            logger.info("Merging visual and audio sections...")
            all_sections = visual_sections + audio_sections
            all_sections.sort(key=lambda x: x["start"])
            
            merged_sections = []
            current_section = None
            
            for section in all_sections:
                if not current_section:
                    current_section = section.copy()
                elif section["start"] <= current_section["end"]:
                    # Sections overlap, merge them
                    current_section["end"] = max(current_section["end"], section["end"])
                    # Combine descriptions, indicating source
                    if section["source"] == "audio":
                        current_section["description"] = f"{current_section['description']}\nAudio: {section['description']}"
                    else:
                        current_section["description"] = f"{current_section['description']}\nVisual: {section['description']}"
                else:
                    # No overlap, add current section and start new one
                    merged_sections.append(current_section)
                    current_section = section.copy()
            
            if current_section:
                merged_sections.append(current_section)

            sections = merged_sections

        return sections

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
        
        # Get YouTube chapters
        chapters = self.extract_chapters(video_url)

        # Get transcript from YouTube or Whisper
        transcript = self.extract_transcript(video_url)
        if not transcript:
            logger.info("No YouTube transcript available, using Whisper...")
            audio_path = str(self.temp_dir / f"{video_id}.wav")
            if self.extract_audio(video_path, audio_path):
                transcript = self.whisper_transcribe(audio_path)
        
        # Extract and analyze frames (if not using chapters for primary sections, or for chapter enrichment)
        frames_data = self.extract_frames(video_path, interval=120)  # 2 minutes interval for general visual analysis
        frame_descriptions = []
        for frame in frames_data:
            analysis = self.analyze_frame_with_gemini(frame["frame_base64"], frame["timestamp"])
            if analysis:
                frame_descriptions.append({
                    "timestamp": frame["timestamp"],
                    "description": analysis["description"]
                })
        
        # Generate sections combining all available data (chapters, visual, audio)
        sections = self.generate_section_breakdown(transcript, {"video_path": video_path, "chapters": chapters})
        
        # Create embeddings for search
        all_texts = []
        metadata = []
        
        # Add sections to search index
        for section in sections:
            all_texts.append(f"{section['title']} {section['description']}")
            metadata.append({
                "title": section["title"],
                "start": section["start"],
                "end": section["end"],
                "description": section["description"]
            })
        
        # Create embeddings and search index
        embeddings = self.create_embeddings(all_texts)
        search_index = self.build_search_index(embeddings)
        
        return {
            "video_id": video_id,
            "video_path": video_path,
            "transcript": transcript,
            "chapters": chapters, # Include chapters in response
            "frame_descriptions": frame_descriptions,
            "sections": sections,
            "search_index": search_index,
            "metadata": metadata,
            "embeddings": embeddings
        }

# Example usage:
if __name__ == "__main__":
    processor = VideoProcessor()
    # result = processor.process_video("https://www.youtube.com/watch?v=VIDEO_ID")