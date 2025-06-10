# Multimodal Video Analysis System

A comprehensive AI-powered system that allows users to upload YouTube videos and interact with them through natural language chat, timestamped navigation, and visual content search.

## Features

- **Video Processing**: Download and analyze YouTube videos
- **Section Breakdown**: Automatically generate timestamped sections with summaries
- **Interactive Chat**: Ask questions about video content with cited responses
- **Visual Search**: Search for specific visual content within the video
- **Timestamp Navigation**: Click on sections or citations to jump to specific moments
- **Modern UI**: Responsive web interface with real-time updates

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **OpenAI GPT-4**: For chat responses and section analysis
- **Google Gemini**: For visual content understanding
- **FAISS**: Vector similarity search
- **SentenceTransformers**: Text embeddings
- **MoviePy**: Video processing
- **PyTube**: YouTube video downloading
- **youtube-transcript-api**: Transcript extraction

### Frontend
- **Vanilla JavaScript**: Interactive user interface
- **CSS3**: Modern styling with gradients and animations
- **Font Awesome**: Icon library

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd multimodal-video-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.template .env
   ```
   
   Edit the `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

5. **Create required directories**
   ```bash
   mkdir -p uploads temp static templates
   ```

## API Keys Setup

### OpenAI API Key
1. Go to [OpenAI API](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add it to your `.env` file

### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file

## Running the Application

1. **Start the server**
   ```bash
   python main.py
   ```

2. **Access the application**
   Open your browser and go to `http://127.0.0.1:8000`

## Usage

1. **Upload a Video**
   - Enter a YouTube URL in the input field
   - Click "Process Video" and wait for analysis to complete

2. **Explore Sections**
   - View automatically generated video sections in the left panel
   - Click on any section to jump to that timestamp on YouTube

3. **Chat with Video**
   - Use the chat tab to ask questions about the video content
   - Responses include timestamped citations
   - Click on citations to jump to relevant moments

4. **Visual Search**
   - Use the search tab to find specific visual content
   - Describe what you're looking for (e.g., "person wearing red shirt")
   - Results show matching video segments with timestamps

## API Endpoints

### Process Video
```
POST /api/process-video
Body: {"url": "youtube_url"}
```

### Chat with Video
```
POST /api/chat
Body: {"video_id": "video_id", "message": "your_question"}
```

### Search Content
```
POST /api/search
Body: {"video_id": "video_id", "query": "search_query"}
```

### Get Video Data
```
GET /api/video/{video_id}
```

## Project Structure

```
multimodal-video-analysis/
├── main.py                 # FastAPI application
├── video_processor.py      # Core video processing logic
├── requirements.txt        # Python dependencies
├── .env.template          # Environment variables template
├── README.md              # This file
├── static/
│   ├── style.css         # Frontend styling
│   └── script.js         # Frontend JavaScript
├── templates/
│   └── index.html        # Main HTML template
├── uploads/              # Downloaded videos
└── temp/                # Temporary frame files
```

## Key Components

### VideoProcessor Class
- Downloads YouTube videos using PyTube
- Extracts transcripts using youtube-transcript-api
- Generates section breakdowns using OpenAI GPT-4
- Creates embeddings for text and visual content
- Implements vector search using FAISS

### Frontend Features
- Responsive design with mobile support
- Real-time chat interface
- Tab-based navigation
- Loading indicators and error handling
- Timestamp-based YouTube navigation

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all API keys are correctly set in `.env`
   - Check API key permissions and quotas

2. **Video Download Issues**
   - Some videos may be restricted or private
   - Try with different YouTube URLs

3. **Transcript Unavailable**
   - Not all videos have transcripts
   - The system will work with limited functionality

4. **Memory Issues**
   - Large videos may require significant memory
   - Consider processing shorter videos for testing

### Performance Optimization

- Frame extraction interval can be adjusted in `extract_frames()`
- Text chunk size can be modified in `generate_section_breakdown()`
- Number of search results can be limited in API calls

## Dependencies

All dependencies are listed in `requirements.txt` with compatible versions that work together:

- **FastAPI & Uvicorn**: Web framework and ASGI server
- **OpenAI**: GPT-4 integration for chat and analysis
- **Google Generative AI**: Gemini model for visual understanding
- **PyTube**: YouTube video downloading
- **youtube-transcript-api**: Extract video transcripts
- **MoviePy**: Video file processing and frame extraction
- **OpenCV (cv2)**: Computer vision and image processing
- **SentenceTransformers**: Text embedding generation
- **FAISS**: Efficient similarity search and clustering
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Requests**: HTTP library
- **Python-dotenv**: Environment variable management
- **Pydantic**: Data validation
- **Jinja2**: Template engine
- **Aiofiles**: Async file operations

## Demo Walkthrough

When presenting your demo, make sure to explain how you tackled each requirement:

### 1. Frontend UI for YouTube Video Upload
- **Implementation**: Clean, modern web interface with URL input validation
- **Features**: Real-time feedback, loading indicators, error handling
- **Technology**: HTML5, CSS3 with responsive design, vanilla JavaScript

### 2. Section Breakdown with Hyperlinked Timestamps
- **Implementation**: Used OpenAI GPT-4 to analyze transcript chunks and generate meaningful section titles and summaries
- **Process**: 
  1. Split transcript into ~1000 character chunks
  2. Generate descriptive titles and summaries for each section
  3. Create clickable sections that link to YouTube timestamps
- **User Experience**: Left panel shows organized sections, click to jump to YouTube at specific time

### 3. Chat with Timestamp Citations
- **Implementation**: 
  1. Vector search to find relevant content for user questions
  2. Context-aware responses using OpenAI GPT-4
  3. Citation system that links back to specific timestamps
- **Process**: Query → Vector Search → Context Building → GPT-4 Response → Citation Formatting
- **User Experience**: Natural conversation with clickable timestamp references

### 4. Visual Content Search
- **Implementation**:
  1. Extract video frames at 30-second intervals using OpenCV
  2. Generate descriptions using Google Gemini vision model
  3. Create embeddings and enable semantic search
- **Process**: Frame Extraction → Gemini Analysis → Embedding Creation → FAISS Indexing → Search Interface
- **User Experience**: Natural language queries to find visual content (e.g., "person in red shirt")

## Architecture Overview

```
User Input (YouTube URL)
        ↓
Video Download (PyTube)
        ↓
Parallel Processing:
├─ Transcript Extraction → Section Analysis (GPT-4) → Text Embeddings
└─ Frame Extraction → Visual Analysis (Gemini) → Visual Embeddings
        ↓
FAISS Vector Store Creation
        ↓
Interactive Interface:
├─ Chat System (GPT-4 + Vector Search)
└─ Visual Search (Semantic Similarity)
```

## Limitations and Future Improvements

### Current Limitations
- Requires videos with available transcripts
- Processing time increases with video length
- Memory usage scales with video size
- API rate limits may affect processing speed

### Potential Improvements
- Add support for uploaded video files
- Implement audio analysis for videos without transcripts
- Add video summarization features
- Support for multiple languages
- Real-time processing progress indicators
- Batch processing capabilities
- Video segment preview in search results

## License

This project is for educational and demonstration purposes. Please ensure compliance with YouTube's Terms of Service and API usage policies when using this system.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation for OpenAI and Google
3. Ensure all environment variables are properly set
4. Verify internet connection for video downloads