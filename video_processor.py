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
        """Download YouTube video and extract metadata with better error handling"""
        try:
            video_id = self.extract_video_id(youtube_url)
            print(f"Attempting to download video ID: {video_id}")
            
            # Try different approaches for YouTube download
            yt = None
            for attempt in range(3):
                try:
                    print(f"Download attempt {attempt + 1}")
                    yt = YouTube(youtube_url, use_oauth=False, allow_oauth_cache=False)
                    
                    # Test if we can access video info
                    title = yt.title
                    print(f"Video title: {title}")
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == 2:
                        raise Exception(f"Failed to access video after 3 attempts: {str(e)}")
                    continue
            
            if not yt:
                raise Exception("Could not create YouTube object")
            
            # Get available streams
            try:
                streams = yt.streams.filter(file_extension='mp4', progressive=True)
                if not streams:
                    # Try adaptive streams if progressive not available
                    streams = yt.streams.filter(file_extension='mp4', adaptive=True)
                    
                if not streams:
                    raise Exception("No MP4 streams available for this video")
                
                # Get the best quality stream
                video_stream = streams.get_highest_resolution()
                if not video_stream:
                    video_stream = streams.first()
                    
                print(f"Selected stream: {video_stream.resolution}, {video_stream.filesize} bytes")
                
            except Exception as e:
                raise Exception(f"Failed to get video streams: {str(e)}")
            
            # Download video
            try:
                video_path = self.upload_dir / f"{video_id}.mp4"
                print(f"Downloading to: {video_path}")
                
                video_stream.download(
                    output_path=str(self.upload_dir), 
                    filename=f"{video_id}.mp4"
                )
                
                # Verify file was downloaded
                if not video_path.exists():
                    raise Exception("Video file was not created")
                    
                file_size = video_path.stat().st_size
                print(f"Downloaded successfully: {file_size} bytes")
                
            except Exception as e:
                raise Exception(f"Failed to download video file: {str(e)}")
            
            # Get transcript
            print("Getting transcript...")
            transcript = self.get_transcript(video_id)
            
            return {
                "video_id": video_id,
                "title": yt.title,
                "description": yt.description or "No description available",
                "duration": yt.length,
                "video_path": str(video_path),
                "transcript": transcript,
                "url": youtube_url
            }
            
        except Exception as e:
            print(f"Error in download_youtube_video: {str(e)}")
            raise Exception(f"Error downloading video: {str(e)}")
    
    def get_transcript(self, video_id: str) -> List[Dict[str, Any]]:
        """Get transcript using youtube-transcript-api"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return transcript_list
        except Exception as e:
            print(f"Could not retrieve transcript: {str(e)}")
            return []
    
    def extract_frames(self, video_path: str, interval: int = 30) -> List[Dict[str, Any]]:
        """Extract frames from video at specified intervals"""
        frames_data = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
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
            
            frame_count += 1
        
        cap.release()
        return frames_data
    
    def generate_section_breakdown(self, transcript: List[Dict], video_metadata: Dict) -> List[Dict[str, Any]]:
        """Generate section breakdown using OpenAI"""
        if not transcript:
            return []
        
        # Combine transcript into text chunks
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
        return results[:k]
    
    def chat_with_video(self, question: str, video_metadata: Dict, embeddings_data: Dict) -> Dict[str, Any]:
        """Chat with video using context from search results"""
        # Search for relevant content
        relevant_content = self.search_content(question, embeddings_data, k=3)
        
        # Build context
        context_parts = []
        citations = []
        
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
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"You are a helpful assistant that answers questions about a video titled '{video_metadata['title']}'. Use the provided context to answer questions accurately. When referencing specific content, mention the timestamp."
                    },
                    {
                        "role": "user",
                        "content": f"Context from video:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
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