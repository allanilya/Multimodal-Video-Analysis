export interface VideoSection {
  start_time: number;
  end_time: number;
  title: string;
  summary: string;
  keywords: string[];
  start_timestamp: string;
  end_timestamp: string;
}

export interface TranscriptSegment {
  start: number;
  duration: number;
  text: string;
}

export interface VideoData {
  video_id: string;
  title: string;
  duration: number;
  sections: VideoSection[];
  transcript: TranscriptSegment[];
  processing_time: number;
  thumbnail_url?: string;
  has_transcript: boolean;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  timestamp: Date;
}

export interface Citation {
  timestamp: string;
  seconds: number;
  section?: string;
}

export interface SearchResult {
  timestamp: number;
  formatted_time: string;
  score: number;
  type: 'text' | 'visual';
  text?: string;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
}

export interface ProcessingStatus {
  video_id: string;
  status: 'processing' | 'completed' | 'error';
  message: string;
  data?: VideoData;
}
