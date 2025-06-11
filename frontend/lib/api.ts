import axios from 'axios';
import { VideoData, ChatMessage, SearchResponse } from '@/types';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const videoApi = {
  async processVideo(url: string, forceReprocess = false) {
    const response = await api.post('/api/video/process', {
      url,
      force_reprocess: forceReprocess,
    });
    return response.data;
  },

  async getVideoInfo(videoId: string): Promise<VideoData> {
    const response = await api.get(`/api/video/${videoId}`);
    return response.data;
  },

  async getStatus(videoId: string) {
    const response = await api.get(`/api/video/${videoId}/status`);
    return response.data;
  },

  async deleteVideo(videoId: string) {
    const response = await api.delete(`/api/video/${videoId}`);
    return response.data;
  },
};

export const chatApi = {
  async sendMessage(videoId: string, message: string, useVisualContext = true) {
    const response = await api.post('/api/chat/', {
      video_id: videoId,
      message,
      use_visual_context: useVisualContext,
    });
    return response.data;
  },
};

export const searchApi = {
  async search(
    videoId: string,
    query: string,
    searchType: 'visual' | 'text' | 'both' = 'both'
  ): Promise<SearchResponse> {
    const response = await api.post('/api/search/', {
      video_id: videoId,
      query,
      search_type: searchType,
    });
    return response.data;
  },
};

export default api;
