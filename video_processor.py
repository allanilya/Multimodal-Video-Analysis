import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
import requests
import time

load_dotenv()

class VideoProcessor:
    def __init__(self):
        # Check API keys
        openai_key = os.getenv("OPENAI_API_KEY")
        google_key = os.getenv("GOOGLE_API_KEY")
        
        if not openai_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        if not google_key:
            raise Exception("GOOGLE_API_KEY not found in environment variables")
            
        self.openai_client = OpenAI(api_key=openai_key)
        genai.configure(api_key=google_key)
        
        # Use Gemini 1.5 Pro for video analysis
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
        self.upload_dir = Path("./uploads")
        self.upload_dir.mkdir(exist_ok=True)
        
    def extract_video_id(self, youtube_url: str) -> str:
        """Extract video ID from YouTube URL"""
        if "youtube.com/watch?v=" in youtube_url:
            return youtube_url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            return youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
    
    def upload_video_to_gemini(self, youtube_url: str) -> Dict[str, Any]:
        """Upload video directly to Gemini for analysis"""
        try:
            video_id = self.extract_video_id(youtube_url)
            print(f"Processing video with Gemini: {video_id}")
            
            # Upload video file to Gemini
            print("Uploading video to Google AI...")
            video_file = genai.upload_file(
                path=youtube_url,  # Gemini can handle YouTube URLs directly
                display_name=f"video_{video_id}"
            )
            
            print(f"Upload complete: {video_file.uri}")
            
            # Wait for processing
            print("Waiting for video processing...")
            while video_file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed")
            
            print("\nVideo processing complete!")
            
            # Get basic video info
            video_info = self.get_video_info(youtube_url)
            
            return {
                "video_id": video_id,
                "gemini_file": video_file,
                "url": youtube_url,
                **video_info
            }
            
        except Exception as e:
            print(f"Error uploading to Gemini: {str(e)}")
            # Fallback to YouTube URL analysis
            return self.analyze_youtube_url(youtube_url)
    
    def get_video_info(self, youtube_url: str) -> Dict[str, Any]:
        """Get basic video information"""
        try:
            # Try to get basic info without downloading
            from pytubefix import YouTube
            yt = YouTube(youtube_url)
            return {
                "title": yt.title,
                "description": yt.description or "No description available",
                "duration": yt.length
            }
        except:
            return {
                "title": "Video Analysis",
                "description": "Description unavailable",
                "duration": 0
            }
    
    def analyze_youtube_url(self, youtube_url: str) -> Dict[str, Any]:
        """Fallback: Analyze YouTube URL directly with Gemini"""
        try:
            video_id = self.extract_video_id(youtube_url)
            print(f"Analyzing YouTube URL directly: {video_id}")
            
            # Create a video part for Gemini
            video_part = genai.types.FileDataPart(
                file_data=genai.types.FileData(
                    file_uri=youtube_url,
                    mime_type="video/mp4"
                )
            )
            
            video_info = self.get_video_info(youtube_url)
            
            return {
                "video_id": video_id,
                "video_part": video_part,
                "url": youtube_url,
                **video_info
            }
            
        except Exception as e:
            raise Exception(f"Could not analyze video: {str(e)}")
    
    def generate_comprehensive_analysis(self, video_data: Dict) -> Dict[str, Any]:
        """Generate comprehensive video analysis using Gemini"""
        try:
            print("Generating comprehensive video analysis...")
            
            # Prepare video for analysis
            if "gemini_file" in video_data:
                video_input = video_data["gemini_file"]
            else:
                video_input = video_data["video_part"]
            
            # Generate comprehensive analysis
            analysis_prompt = f"""
            Analyze this video titled "{video_data.get('title', 'Unknown')}" comprehensively:
            
            1. TRANSCRIPT: Provide a detailed transcript of all spoken content with timestamps
            2. SECTIONS: Break the video into 5-8 logical sections with start times and descriptions
            3. VISUAL CONTENT: Describe key visual elements, scenes, and activities
            4. SUMMARY: Provide a comprehensive summary of the video content
            
            Format your response as JSON with these keys:
            - transcript: [{{time: "MM:SS", text: "spoken content"}}]
            - sections: [{{start_time: seconds, title: "section title", description: "description"}}]
            - visual_content: [{{timestamp: seconds, description: "visual description"}}]
            - summary: "comprehensive summary"
            """
            
            response = self.gemini_model.generate_content([
                analysis_prompt,
                video_input
            ])
            
            if not response.text:
                raise Exception("No response from Gemini video analysis")
            
            print("Analysis complete!")
            
            # Parse response
            try:
                analysis = json.loads(response.text)
            except:
                # If JSON parsing fails, create structured response
                analysis = self.parse_text_response(response.text)
            
            return analysis
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return self.fallback_analysis(video_data)
    
    def parse_text_response(self, text: str) -> Dict[str, Any]:
        """Parse text response into structured format"""
        lines = text.split('\n')
        
        sections = []
        transcript = []
        visual_content = []
        summary = text[:500] + "..." if len(text) > 500 else text
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for section markers
            if any(word in line.lower() for word in ["section", "part", "minute", ":"]):
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    "start_time": len(sections) * 60,  # Estimate
                    "title": line[:50],
                    "description": line
                }
        
        if current_section:
            sections.append(current_section)
        
        return {
            "transcript": transcript,
            "sections": sections,
            "visual_content": visual_content,
            "summary": summary
        }
    
    def fallback_analysis(self, video_data: Dict) -> Dict[str, Any]:
        """Fallback analysis when comprehensive fails"""
        return {
            "transcript": [],
            "sections": [{
                "start_time": 0,
                "title": "Video Content",
                "description": f"Analysis of {video_data.get('title', 'video content')}"
            }],
            "visual_content": [],
            "summary": f"Video titled: {video_data.get('title', 'Unknown')}"
        }
    
    def chat_with_video(self, question: str, video_data: Dict, analysis: Dict) -> Dict[str, Any]:
        """Chat about video using comprehensive analysis"""
        try:
            # Build context from analysis
            context_parts = []
            
            # Add transcript
            if analysis.get("transcript"):
                transcript_text = "\n".join([f"{t['time']}: {t['text']}" for t in analysis["transcript"][:10]])
                context_parts.append(f"TRANSCRIPT:\n{transcript_text}")
            
            # Add sections
            if analysis.get("sections"):
                sections_text = "\n".join([f"{s['start_time']}s: {s['title']} - {s['description']}" for s in analysis["sections"]])
                context_parts.append(f"SECTIONS:\n{sections_text}")
            
            # Add visual content
            if analysis.get("visual_content"):
                visual_text = "\n".join([f"{v['timestamp']}s: {v['description']}" for v in analysis["visual_content"][:5]])
                context_parts.append(f"VISUAL CONTENT:\n{visual_text}")
            
            # Add summary
            if analysis.get("summary"):
                context_parts.append(f"SUMMARY:\n{analysis['summary']}")
            
            context = "\n\n".join(context_parts)
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are an AI assistant that can answer questions about a video titled "{video_data.get('title', 'Unknown')}".
                        
                        Use the provided video analysis to answer questions accurately. Always reference specific timestamps when relevant.
                        If you mention content from the video, include the timestamp (e.g., "At 2:30...").
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Video Analysis:\n{context}\n\nQuestion: {question}"
                    }
                ],
                temperature=0.7
            )
            
            return {
                "answer": response.choices[0].message.content,
                "context_used": len(context_parts),
                "analysis_available": bool(analysis.get("transcript") or analysis.get("sections"))
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error: {str(e)}. Please try rephrasing your question.",
                "context_used": 0,
                "analysis_available": False
            }
    
    def search_content(self, query: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Search video content using analysis"""
        results = []
        
        # Search transcript
        if analysis.get("transcript"):
            for item in analysis["transcript"]:
                if query.lower() in item.get("text", "").lower():
                    results.append({
                        "type": "transcript",
                        "time": item["time"],
                        "content": item["text"],
                        "score": 0.9
                    })
        
        # Search sections
        if analysis.get("sections"):
            for section in analysis["sections"]:
                if query.lower() in section.get("description", "").lower():
                    results.append({
                        "type": "section",
                        "start_time": section["start_time"],
                        "title": section["title"],
                        "content": section["description"],
                        "score": 0.8
                    })
        
        # Search visual content
        if analysis.get("visual_content"):
            for visual in analysis["visual_content"]:
                if query.lower() in visual.get("description", "").lower():
                    results.append({
                        "type": "visual",
                        "timestamp": visual["timestamp"],
                        "content": visual["description"],
                        "score": 0.7
                    })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:10]