'use client';

import React, { useState } from 'react';
import { Search, Loader2, Eye, FileText } from 'lucide-react';
import { SearchResult } from '@/types';
import { searchApi } from '@/lib/api';

interface SearchInterfaceProps {
  videoId: string;
  onResultClick: (seconds: number) => void;
}

export default function SearchInterface({ videoId, onResultClick }: SearchInterfaceProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [searchType, setSearchType] = useState<'both' | 'visual' | 'text'>('both');

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    try {
      const response = await searchApi.search(videoId, query, searchType);
      console.log('Search response:', response);
      console.log('Results count:', response.results?.length || 0);
      setResults(response.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-[#262626] rounded-lg shadow-lg p-4 border border-gray-200 dark:border-red-600/20">
      <h2 className="text-xl font-bold text-gray-800 dark:text-white flex items-center gap-2 mb-4">
        <Search className="text-red-600 dark:text-red-500" size={24} />
        Search Video Content
      </h2>

      <form onSubmit={handleSearch} className="space-y-3">
        <div className="flex gap-2">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search for visual or text content..."
            className="flex-1 px-4 py-2 bg-white dark:bg-[#1a1a1a] border border-gray-300 dark:border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-500 text-gray-900 dark:text-white placeholder-gray-400 dark:placeholder-gray-500"
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-300 dark:disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            {loading ? <Loader2 className="animate-spin" size={20} /> : <Search size={20} />}
            Search
          </button>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setSearchType('both')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
              searchType === 'both'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            Both
          </button>
          <button
            type="button"
            onClick={() => setSearchType('visual')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${
              searchType === 'visual'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <Eye size={14} />
            Visual
          </button>
          <button
            type="button"
            onClick={() => setSearchType('text')}
            className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors flex items-center gap-1 ${
              searchType === 'text'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
          >
            <FileText size={14} />
            Text
          </button>
        </div>
      </form>

      {!loading && query && results.length === 0 && (
        <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-950/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">No results found for "{query}"</p>
        </div>
      )}

      {results.length > 0 && (
        <div className="mt-4 space-y-2 max-h-96 overflow-y-auto">
          <p className="text-sm text-gray-600 dark:text-gray-400">{results.length} results found</p>
          {results.map((result, index) => (
            <button
              key={index}
              onClick={() => onResultClick(result.timestamp)}
              className="w-full text-left p-3 bg-gray-50 dark:bg-[#1a1a1a] hover:bg-gray-100 dark:hover:bg-[#202020] rounded-lg border border-gray-200 dark:border-gray-700 transition-colors"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    {result.type === 'visual' ? (
                      <Eye size={14} className="text-purple-600 dark:text-purple-400" />
                    ) : (
                      <FileText size={14} className="text-green-600 dark:text-green-400" />
                    )}
                    <span className="text-sm font-medium text-red-600 dark:text-red-500">
                      {result.formatted_time}
                    </span>
                    <span
                      className={`text-xs px-2 py-0.5 rounded-full ${
                        result.type === 'visual'
                          ? 'bg-purple-100 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300'
                          : 'bg-green-100 dark:bg-green-950/50 text-green-700 dark:text-green-300'
                      }`}
                    >
                      {result.type}
                    </span>
                  </div>
                  {result.text && (
                    <p className="text-sm text-gray-700 dark:text-gray-300">{result.text}</p>
                  )}
                </div>
                <div className="text-xs text-gray-500">
                  {Math.round(result.score * 100)}%
                </div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
