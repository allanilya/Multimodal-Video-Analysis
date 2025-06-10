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
        self.upload_dir.mkdir(exist_ok=True)
        self.temp_dir.mkdir(exist_ok=True)
        
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
        # Only use Whisper for audio transcription
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
    
    def extract_frames(self, video_path: str, interval: int = 15, max_frames: int = 40) -> List[Dict[str, Any]]:
        """Extract frames from video at specified intervals (default 15s, max 40 frames)"""
        if not video_path or not Path(video_path).exists():
            print("No video file available - skipping frame extraction")
            return []
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"Video info: {duration:.1f}s duration, {fps:.1f} fps")
        frame_count = 0
        extracted_count = 0
        extracted_timestamps = []
        while True:
            ret, frame = cap.read()
            if not ret or len(frames_data) >= max_frames:
                break
            timestamp = frame_count / fps
            # Extract frame at intervals
            if frame_count % (fps * interval) == 0:
                frame_path = self.temp_dir / f"frame_{int(timestamp)}.jpg"
                cv2.imwrite(str(frame_path), frame)
                # Encode frame as base64 for Gemini
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                frames_data.append({
                    "timestamp": timestamp,
                    "frame_path": str(frame_path),
                    "frame_base64": frame_base64
                })
                extracted_count += 1
                extracted_timestamps.append(timestamp)
                print(f"Extracted frame at {timestamp:.1f}s ({extracted_count}/{max_frames})")
            frame_count += 1
        cap.release()
        print(f"Extracted {len(frames_data)} frames from video (every {interval}s, max {max_frames})")
        print(f"Extracted frame timestamps: {[f'{t:.1f}' for t in extracted_timestamps]}")
        return frames_data
    
    def generate_section_breakdown(self, transcript: List[Dict], video_metadata: Dict) -> List[Dict[str, Any]]:
        """Generate section breakdown using OpenAI and visual analysis if no transcript"""
        
        # If we have transcript, use it
        if transcript and len(transcript) > 0:
            return self.generate_sections_from_transcript(transcript, video_metadata)
        
        # If no transcript, use visual analysis
        print("No transcript available - generating sections from visual analysis")
        return self.generate_sections_from_frames(video_metadata)
    
    def generate_sections_from_transcript(self, transcript: List[Dict], video_metadata: Dict) -> List[Dict[str, Any]]:
        """Generate sections from transcript (original method)"""
        text_chunks = []
        current_chunk = ""
        chunk_start_time = 0
        
        for entry in transcript:
            if len(current_chunk) > 1000:  # ~1000 char chunks
                text_chunks.append({
                    "text": current_chunk,
                    "start_time": chunk_start_time,
                    "end_time": entry["start"]
                })
                current_chunk = entry["text"]
                chunk_start_time = entry["start"]
            else:
                current_chunk += " " + entry["text"]
        
        if current_chunk:
            text_chunks.append({
                "text": current_chunk,
                "start_time": chunk_start_time,
                "end_time": transcript[-1]["start"] + transcript[-1]["duration"]
            })
        
        # Generate sections using OpenAI
        sections = []
        for i, chunk in enumerate(text_chunks):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are analyzing a video transcript. Create a concise, descriptive title for this section (max 8 words) and a brief summary (max 50 words)."
                        },
                        {
                            "role": "user",
                            "content": f"Video title: {video_metadata['title']}\n\nTranscript section:\n{chunk['text']}"
                        }
                    ],
                    temperature=0.3
                )
                
                content = response.choices[0].message.content
                lines = content.strip().split('\n')
                title = lines[0] if lines else f"Section {i+1}"
                summary = lines[1] if len(lines) > 1 else ""
                
                sections.append({
                    "title": title.replace("Title:", "").replace("**", "").strip(),
                    "summary": summary.replace("Summary:", "").replace("**", "").strip(),
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "transcript_text": chunk["text"]
                })
            except Exception as e:
                print(f"Error generating section {i}: {str(e)}")
                sections.append({
                    "title": f"Section {i+1}",
                    "summary": "Content analysis unavailable",
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "transcript_text": chunk["text"]
                })
        
        return sections
    
    def generate_sections_from_frames(self, video_metadata: Dict) -> List[Dict[str, Any]]:
        """Generate sections using visual analysis when no transcript is available (denser frames, limited)"""
        try:
            # Extract more frames for analysis (every 15 seconds, max 40 frames)
            frames = self.extract_frames(video_metadata["video_path"], interval=15, max_frames=40)
            if not frames:
                print("No frames extracted for visual section generation.")
                return []
            print(f"Generating sections from {len(frames)} frames...")
            sections = []
            duration = video_metadata.get("duration", 0)
            for i, frame in enumerate(frames):
                try:
                    print(f"[Gemini] Analyzing frame {i+1}/{len(frames)} at {frame['timestamp']:.1f}s...")
                    image_part = genai.types.BlobDict(
                        mime_type="image/jpeg",
                        data=base64.b64decode(frame["frame_base64"])
                    )
                    response = self.gemini_model.generate_content([
                        f"Analyze this frame from a video titled '{video_metadata['title']}'. Provide a brief title (max 6 words) and description (max 40 words) of what's happening:",
                        image_part
                    ])
                    print(f"[Gemini] Response for frame {i+1}: {response.text}")
                    if response.text:
                        lines = response.text.strip().split('\n')
                        title = lines[0] if lines else f"Visual Section {i+1}"
                        description = lines[1] if len(lines) > 1 else "Visual content analysis"
                        start_time = frame["timestamp"]
                        end_time = frames[i+1]["timestamp"] if i+1 < len(frames) else duration
                        sections.append({
                            "title": title.replace("Title:", "").replace("**", "").strip(),
                            "summary": description.replace("Description:", "").replace("**", "").strip(),
                            "start_time": start_time,
                            "end_time": end_time,
                            "transcript_text": f"Visual content at {start_time:.0f}s: {description}"
                        })
                        print(f"Generated visual section: {title} ({start_time:.1f}s - {end_time:.1f}s)")
                except Exception as e:
                    print(f"Error analyzing frame {i}: {str(e)}")
            print(f"Total sections generated: {len(sections)}")
            for sec in sections:
                print(f"Section: {sec['title']} | {sec['start_time']:.1f}s - {sec['end_time']:.1f}s")
            return sections
        except Exception as e:
            print(f"Error generating visual sections: {str(e)}")
            return []
    
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
            additional_frames = self.extract_frames(video_metadata["video_path"], interval=120)  # Every 2 minutes
            
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
    
    def whisper_transcribe(self, audio_path: str) -> list:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            client = self.openai_client
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json"
                )
            print(f"Whisper transcript received with {len(transcript['segments'])} segments")
            # Convert Whisper segments to youtube-transcript-api format
            return [
                {"text": seg["text"], "start": seg["start"], "duration": seg["end"] - seg["start"]}
                for seg in transcript["segments"]
            ]
        except Exception as e:
            print(f"Whisper transcription failed: {str(e)}")
            return []