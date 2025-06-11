'use client';

import React, { useState, useEffect } from 'react';
import { Youtube, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { io, Socket } from 'socket.io-client';
import VideoPlayer from '@/components/VideoPlayer';
import VideoSections from '@/components/VideoSections';
import ChatInterface from '@/components/ChatInterface';
import SearchInterface from '@/components/SearchInterface';
import { VideoData, ProcessingStatus } from '@/types';
import { videoApi } from '@/lib/api';
import { extractVideoId } from '@/lib/utils';

export default function Home() {
  const [url, setUrl] = useState('');
  const [videoData, setVideoData] = useState<VideoData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [seekTo, setSeekTo] = useState<number | undefined>(undefined);
  const [currentTime, setCurrentTime] = useState(0);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [processingMessage, setProcessingMessage] = useState('');

  useEffect(() => {
    // Initialize socket connection
    const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:5001';
    const newSocket = io(WS_URL);

    newSocket.on('processing_status', (status: ProcessingStatus) => {
      setProcessingMessage(status.message);

      if (status.status === 'completed' && status.data) {
        setVideoData(status.data);
        setLoading(false);
        setError('');
      } else if (status.status === 'error') {
        setError(status.message);
        setLoading(false);
      }
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!url.trim()) return;

    const videoId = extractVideoId(url);
    if (!videoId) {
      setError('Invalid YouTube URL');
      return;
    }

    setLoading(true);
    setError('');
    setVideoData(null);
    setProcessingMessage('Starting video processing...');

    try {
      const response = await videoApi.processVideo(url);

      if (response.status === 'completed' && response.data) {
        setVideoData(response.data);
        setLoading(false);
      }
      // If status is 'processing', we'll wait for socket updates
    } catch (err: any) {
      setError(err.response?.data?.error || 'Failed to process video');
      setLoading(false);
    }
  };

  const handleSeek = (seconds: number) => {
    setSeekTo(seconds);
    // Reset seekTo after a short delay to allow multiple seeks to the same timestamp
    setTimeout(() => setSeekTo(undefined), 100);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <Youtube className="text-red-600" size={36} />
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Multimodal Video Analysis
              </h1>
              <p className="text-gray-600">
                AI-powered video analysis with chat and visual search
              </p>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        {/* URL Input */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                YouTube Video URL
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={loading}
                />
                <button
                  type="submit"
                  disabled={loading || !url.trim()}
                  className="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center gap-2"
                >
                  {loading ? (
                    <>
                      <Loader2 className="animate-spin" size={20} />
                      Processing...
                    </>
                  ) : (
                    'Analyze Video'
                  )}
                </button>
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2 text-red-600 bg-red-50 p-3 rounded-lg">
                <XCircle size={20} />
                <span>{error}</span>
              </div>
            )}

            {loading && processingMessage && (
              <div className="flex items-center gap-2 text-blue-600 bg-blue-50 p-3 rounded-lg">
                <Loader2 className="animate-spin" size={20} />
                <span>{processingMessage}</span>
              </div>
            )}
          </form>
        </div>

        {/* Video Analysis Interface */}
        {videoData && (
          <div className="space-y-6">
            {/* Success Message */}
            <div className="flex items-center gap-2 text-green-600 bg-green-50 p-3 rounded-lg">
              <CheckCircle size={20} />
              <span>
                Video processed in {videoData.processing_time.toFixed(2)}s
              </span>
            </div>

            {/* Video Title */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                {videoData.title}
              </h2>
              <div className="flex gap-4 text-sm text-gray-600">
                <span>Duration: {Math.floor(videoData.duration / 60)}m</span>
                <span>Sections: {videoData.sections.length}</span>
                {videoData.has_transcript && <span>Transcript available</span>}
              </div>
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left Column - Video Player & Sections */}
              <div className="lg:col-span-2 space-y-6">
                <VideoPlayer
                  videoId={videoData.video_id}
                  onTimeUpdate={setCurrentTime}
                  seekTo={seekTo}
                />

                <VideoSections
                  sections={videoData.sections}
                  onSectionClick={handleSeek}
                  currentTime={currentTime}
                />
              </div>

              {/* Right Column - Chat & Search */}
              <div className="space-y-6">
                <div className="h-[500px]">
                  <ChatInterface
                    videoId={videoData.video_id}
                    onCitationClick={handleSeek}
                  />
                </div>

                <SearchInterface
                  videoId={videoData.video_id}
                  onResultClick={handleSeek}
                />
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!videoData && !loading && (
          <div className="text-center py-16">
            <Youtube className="mx-auto text-gray-300 mb-4" size={64} />
            <h2 className="text-2xl font-bold text-gray-700 mb-2">
              Start by entering a YouTube URL
            </h2>
            <p className="text-gray-600">
              Analyze videos with AI-powered section breakdown, chat, and visual search
            </p>
          </div>
        )}
      </main>
    </div>
  );
}
