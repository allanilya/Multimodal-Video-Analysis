'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, MessageSquare, Loader2 } from 'lucide-react';
import { ChatMessage } from '@/types';
import { chatApi } from '@/lib/api';

interface ChatInterfaceProps {
  videoId: string;
  onCitationClick: (seconds: number) => void;
}

export default function ChatInterface({ videoId, onCitationClick }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await chatApi.sendMessage(videoId, input);

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.response,
        citations: response.citations,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage: ChatMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your question. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-white rounded-lg shadow-lg">
      {/* Header */}
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <MessageSquare className="text-red-600" size={24} />
          Ask About This Video
        </h2>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-8">
            <MessageSquare size={48} className="mx-auto mb-2 opacity-50" />
            <p>Ask any question about the video content</p>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.role === 'user'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              <p className="whitespace-pre-wrap">{message.content}</p>

              {message.citations && message.citations.length > 0 && (
                <div className="mt-2 pt-2 border-t border-gray-300">
                  <p className="text-xs font-semibold mb-1">Citations:</p>
                  <div className="flex flex-wrap gap-1">
                    {message.citations.map((citation, i) => (
                      <button
                        key={i}
                        onClick={() => onCitationClick(citation.seconds)}
                        className="text-xs px-2 py-1 bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                      >
                        {citation.timestamp}
                        {citation.section && ` - ${citation.section}`}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-lg p-3">
              <Loader2 className="animate-spin text-red-600" size={20} />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a question about the video..."
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 text-gray-900"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={20} />
          </button>
        </div>
      </form>
    </div>
  );
}
