import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from pytubefix import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from openai import OpenAI
from dotenv import load_dotenv
import subprocess
import whisper

load_dotenv()

class VideoProcessor:
    def __init__(self):
        # Check API keys first
        openai_key = os.getenv("OPENAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if not openai_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        if not google_key:
            raise Exception("GOOGLE_API_KEY not found in environment variables")
            
        self.openai_client = OpenAI(api_key=openai_key)
        genai.configure(api_key=google_key)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./uploads"))
        self.temp_dir = Path(os.getenv("TEMP_DIR", "./temp"))
        self.temp_dir.mkdir(exist_ok=True)
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)
        self.processed_videos = {}
        self.whisper_model = whisper.load_model("base")
        
    def extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "youtube.com/watch?v=" in youtube_url:
            return youtube_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            return youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
    
    def download_youtube_video(self, youtube_url: str) -> Dict[str, Any]:
        """Download YouTube video and extract metadata with fallback to transcript-only mode and Whisper"""
        try:
            video_id = self.extract_video_id(youtube_url)
            print(f"Attempting to download video ID: {video_id}")
            yt = None
            for attempt in range(3):
                try:
                    print(f"Info retrieval attempt {attempt + 1}")
                    yt = YouTube(youtube_url, use_oauth=False, allow_oauth_cache=False)
                    title = yt.title
                    duration = yt.length
                    description = yt.description or "No description available"
                    print(f"Video info retrieved: {title}")
                    break
                except Exception as e:
                    print(f"Info attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:
                        print("Falling back to transcript-only mode...")
                        return self.transcript_only_fallback(video_id, youtube_url)
                    continue
            video_path = None
            try:
                streams = yt.streams.filter(file_extension='mp4', progressive=True)
                if not streams:
                    streams = yt.streams.filter(file_extension='mp4', adaptive=True)
                if not streams:
                    print("No streams available - using transcript-only mode")
                    return self.create_transcript_only_response(video_id, yt, youtube_url)
                video_stream = streams.get_highest_resolution() or streams.first()
                print(f"Selected stream: {video_stream.resolution}")
                video_path = self.upload_dir / f"{video_id}.mp4"
                print(f"Downloading to: {video_path}")
                video_stream.download(
                    output_path=str(self.upload_dir), 
                    filename=f"{video_id}.mp4"
                )
                if not video_path.exists():
                    raise Exception("Video file was not created")
                print(f"Downloaded successfully: {video_path.stat().st_size} bytes")
            except Exception as e:
                print(f"Video download failed: {str(e)}")
                print("Continuing with transcript-only mode...")
                return self.create_transcript_only_response(video_id, yt, youtube_url)
            print("Getting transcript...")
            transcript = self.get_transcript(video_id, str(video_path))
            return {
                "video_id": video_id,
                "title": yt.title,
                "description": yt.description or "No description available",
                "duration": yt.length,
                "video_path": str(video_path) if video_path else None,
                "transcript": transcript,
                "url": youtube_url,
                "mode": "full"
            }
        except Exception as e:
            print(f"Error in download_youtube_video: {str(e)}")
            return self.transcript_only_fallback(video_id if 'video_id' in locals() else None, youtube_url)
    
    def transcript_only_fallback(self, video_id: str, youtube_url: str) -> Dict[str, Any]:
        """Fallback to transcript-only mode when video download fails, with Whisper support"""
        try:
            if not video_id:
                video_id = self.extract_video_id(youtube_url)
            print(f"Attempting transcript-only mode for: {video_id}")
            # Try to get transcript from YouTube, fallback to Whisper if needed
            transcript = self.get_transcript(video_id)
            if not transcript:
                # Try to download video for Whisper
                try:
                    yt = YouTube(youtube_url, use_oauth=False, allow_oauth_cache=False)
                    streams = yt.streams.filter(file_extension='mp4', progressive=True)
                    if not streams:
                        streams = yt.streams.filter(file_extension='mp4', adaptive=True)
                    if streams:
                        video_stream = streams.get_highest_resolution() or streams.first()
                        video_path = self.upload_dir / f"{video_id}.mp4"
                        video_stream.download(
                            output_path=str(self.upload_dir), 
                            filename=f"{video_id}.mp4"
                        )
                        transcript = self.get_transcript(video_id, str(video_path))
                except Exception as e:
                    print(f"Could not download video for Whisper fallback: {str(e)}")
            if not transcript:
                raise Exception("No transcript available for this video (YouTube or Whisper)")
            try:
                yt = YouTube(youtube_url, use_oauth=False, allow_oauth_cache=False)
                title = yt.title
                description = yt.description or "No description available"
                duration = yt.length
            except:
                title = f"Video {video_id}"
                description = "Description unavailable"
                duration = 0
            return {
                "video_id": video_id,
                "title": title,
                "description": description,
                "duration": duration,
                "video_path": None,
                "transcript": transcript,
                "url": youtube_url,
                "mode": "transcript_only"
            }
        except Exception as e:
            raise Exception(f"Could not process video in any mode: {str(e)}")
    
    def create_transcript_only_response(self, video_id: str, yt: YouTube, youtube_url: str) -> Dict[str, Any]:
        """Create response for transcript-only mode with YouTube object"""
        print("Creating transcript-only response...")
        transcript = self.get_transcript(video_id)
        
        return {
            "video_id": video_id,
            "title": yt.title,
            "description": yt.description or "No description available",
            "duration": yt.length,
            "video_path": None,
            "transcript": transcript,
            "url": youtube_url,
            "mode": "transcript_only"
        }
    
    def get_transcript(self, video_id: str, video_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if video_path and Path(video_path).exists():
            print(f"[Transcript] Using Whisper transcription for video ID: {video_id}")
            audio_path = str(self.temp_dir / f"{video_id}.wav")
            if self.extract_audio(video_path, audio_path):
                whisper_transcript = self.whisper_transcribe(audio_path)
                if whisper_transcript:
                    print(f"[Transcript] Whisper transcript generated with {len(whisper_transcript)} segments.")
                    return whisper_transcript
        print("[Transcript] No transcript available for this video.")
        return []
    
    def extract_frames(self, video_path: str) -> List[Dict[str, Any]]:
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return frames_data
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"Video info: {duration:.1f}s duration, {fps:.1f} fps")
        frame_interval = 40  # Extract a frame every 60 seconds (optimized for speed)
        frame_count = 0
        for i in range(0, int(duration), frame_interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, i * 1000)
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                print(f"Extracted frame at {i:.1f}s ({frame_count})")
                frames_data.append({
                    "timestamp": i,
                    "frame": frame
                })
        cap.release()
        print(f"Extracted {frame_count} frames from video (every {frame_interval}s)")
        return frames_data
    
    def generate_section_breakdown(self, transcript: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not transcript:
            print("No transcript available - generating sections from visual analysis")
            frames = self.extract_frames(video_metadata["video_path"])
            if not frames:
                return []
            sections = []
            for frame in frames:
                print(f"[Gemini] Analyzing frame at {frame['timestamp']:.1f}s...")
                prompt = f"Analyze this video frame and provide a title and description. Frame timestamp: {frame['timestamp']:.1f}s"
                response = self.gemini_model.generate_content(prompt)
                if response and response.text:
                    sections.append({
                        "title": response.text.split('\n')[0].replace('**Title:**', '').strip(),
                        "start": frame['timestamp'],
                        "end": frame['timestamp'] + 30,  # Assume 30s per section
                        "transcript_text": f"Visual content at {frame['timestamp']:.1f}s: {response.text}"
                    })
            return sections
        else:
            # Use OpenAI to generate sections from transcript
            # Limit transcript to first 500 characters to avoid context length errors
            limited_transcript = transcript[:500]
            prompt = f"Generate a section breakdown for this video transcript:\n{limited_transcript}"
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a simpler model for faster processing
                messages=[{"role": "user", "content": prompt}]
            )
            sections = []
            for line in response.choices[0].message.content.split('\n'):
                if '|' in line:
                    title, time_range = line.split('|')
                    start, end = time_range.split('-')
                    sections.append({
                        "title": title.strip(),
                        "start": float(start.replace('s', '')),
                        "end": float(end.replace('s', '')),
                        "transcript_text": f"Transcript section: {title.strip()}"
                    })
            return sections
    
    def create_embeddings(self, sections: List[Dict], frames_data: List[Dict]) -> Dict[str, Any]:
        """Create embeddings for text and visual content using scikit-learn"""
        # Text embeddings
        text_embeddings = []
        text_metadata = []
        
        for section in sections:
            embedding = self.embedding_model.encode(section["transcript_text"])
            text_embeddings.append(embedding)
            text_metadata.append({
                "type": "text",
                "title": section["title"],
                "summary": section["summary"],
                "start_time": section["start_time"],
                "end_time": section["end_time"],
                "text": section["transcript_text"]
            })
        
        # Visual embeddings using Gemini
        visual_embeddings = []
        visual_metadata = []
        
        for frame_data in frames_data:
            try:
                # Generate description using Gemini
                image_part = genai.types.BlobDict(
                    mime_type="image/jpeg",
                    data=base64.b64decode(frame_data["frame_base64"])
                )
                
                response = self.gemini_model.generate_content([
                    "Describe this video frame in detail, focusing on objects, people, activities, text, and scene composition:",
                    image_part
                ])
                
                description = response.text if response.text else "No description available"
                
                # Create embedding from description
                embedding = self.embedding_model.encode(description)
                visual_embeddings.append(embedding)
                visual_metadata.append({
                    "type": "visual",
                    "timestamp": frame_data["timestamp"],
                    "description": description,
                    "frame_path": frame_data["frame_path"]
                })
                
            except Exception as e:
                print(f"Error processing frame at {frame_data['timestamp']}: {str(e)}")
        
        # Store embeddings as numpy arrays
        if text_embeddings:
            text_embeddings = np.array(text_embeddings).astype('float32')
            text_embeddings = normalize(text_embeddings, norm='l2')
        else:
            text_embeddings = None
        
        if visual_embeddings:
            visual_embeddings = np.array(visual_embeddings).astype('float32')
            visual_embeddings = normalize(visual_embeddings, norm='l2')
        else:
            visual_embeddings = None
        
        print(f"[Embeddings] Created {len(text_embeddings) if text_embeddings is not None else 0} text embeddings and {len(visual_embeddings) if visual_embeddings is not None else 0} visual embeddings.")
        return {
            "text_embeddings": text_embeddings,
            "text_metadata": text_metadata,
            "visual_embeddings": visual_embeddings,
            "visual_metadata": visual_metadata
        }
    
    def search_content(self, query: str, embeddings_data: Dict, k: int = 5) -> List[Dict[str, Any]]:
        """Search both text and visual content using cosine similarity"""
        query_embedding = self.embedding_model.encode([query]).astype('float32')
        query_embedding = normalize(query_embedding, norm='l2')
        
        results = []
        
        # Search text content
        if embeddings_data["text_embeddings"] is not None:
            similarities = cosine_similarity(query_embedding, embeddings_data["text_embeddings"])[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in top_indices:
                if idx < len(embeddings_data["text_metadata"]):
                    result = embeddings_data["text_metadata"][idx].copy()
                    result["score"] = float(similarities[idx])
                    results.append(result)
        
        # Search visual content
        if embeddings_data["visual_embeddings"] is not None:
            similarities = cosine_similarity(query_embedding, embeddings_data["visual_embeddings"])[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            
            for idx in top_indices:
                if idx < len(embeddings_data["visual_metadata"]):
                    result = embeddings_data["visual_metadata"][idx].copy()
                    result["score"] = float(similarities[idx])
                    results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        print(f"[Search] Retrieved {len(results)} results for query '{query}'. Types: {[r['type'] for r in results]}")
        return results[:k]
    
    def chat_with_video(self, question: str, video_metadata: Dict, embeddings_data: Dict) -> Dict[str, Any]:
        """Chat with video using context from search results and visual analysis"""
        # Search for relevant content
        relevant_content = self.search_content(question, embeddings_data, k=5)
        
        # Build context from both text and visual content
        context_parts = []
        citations = []
        
        # If we have very little content, analyze more frames
        if len(relevant_content) < 2:
            print("Limited content found, analyzing additional frames...")
            additional_frames = self.extract_frames(video_metadata["video_path"])
            
            for frame in additional_frames[:3]:  # Analyze first 3 frames
                try:
                    image_part = genai.types.BlobDict(
                        mime_type="image/jpeg",
                        data=base64.b64decode(frame["frame_base64"])
                    )
                    
                    response = self.gemini_model.generate_content([
                        f"Analyze this frame from '{video_metadata['title']}' and describe what you see in detail:",
                        image_part
                    ])
                    
                    if response.text:
                        context_parts.append(f"Visual at {frame['timestamp']:.0f}s: {response.text}")
                        citations.append({
                            "timestamp": frame["timestamp"],
                            "description": response.text[:100] + "...",
                            "type": "visual"
                        })
                        
                except Exception as e:
                    print(f"Error analyzing additional frame: {str(e)}")
        
        # Add existing relevant content
        for content in relevant_content:
            if content["type"] == "text":
                context_parts.append(f"Section: {content['title']}\nTime: {content['start_time']:.0f}s-{content['end_time']:.0f}s\nContent: {content['text'][:500]}...")
                citations.append({
                    "title": content["title"],
                    "start_time": content["start_time"],
                    "end_time": content["end_time"],
                    "type": "text"
                })
            else:
                context_parts.append(f"Visual at {content['timestamp']:.0f}s: {content['description']}")
                citations.append({
                    "timestamp": content["timestamp"],
                    "description": content["description"],
                    "type": "visual"
                })
        
        context = "\n\n".join(context_parts)
        
        # Generate response using OpenAI
        try:
            system_message = f"""You are a helpful assistant that answers questions about a video titled '{video_metadata['title']}'. 
            
            The video is {video_metadata.get('duration', 0)//60} minutes long. Use the provided context to answer questions accurately. 
            When referencing specific content, mention the timestamp. If the context is limited, acknowledge this but still provide helpful information based on what's available.
            
            Video description: {video_metadata.get('description', 'No description available')}"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": f"Context from video:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            print(f"[Chat] Context built from {len(relevant_content)} items. Types: {[c['type'] for c in relevant_content]}")
            return {
                "answer": answer,
                "citations": citations,
                "relevant_content": relevant_content
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "citations": citations,
                "relevant_content": relevant_content
            }
    
    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video using ffmpeg"""
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Audio extracted to {audio_path}")
            return True
        except Exception as e:
            print(f"Audio extraction failed: {str(e)}")
            return False
    
    def whisper_transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        try:
            print(f"Transcribing audio with Whisper: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            if result and 'segments' in result:
                return result['segments']
            else:
                print("Whisper transcription failed: No segments found.")
                return []
        except Exception as e:
            print(f"Whisper transcription failed: {str(e)}")
            return []