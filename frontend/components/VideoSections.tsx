'use client';

import React from 'react';
import { VideoSection } from '@/types';
import { Clock, ChevronRight } from 'lucide-react';

interface VideoSectionsProps {
  sections: VideoSection[];
  onSectionClick: (seconds: number) => void;
  currentTime?: number;
}

export default function VideoSections({ sections, onSectionClick, currentTime = 0 }: VideoSectionsProps) {
  return (
    <div className="space-y-3">
      <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
        <Clock className="text-red-600" size={24} />
        Video Sections
      </h2>

      <div className="space-y-2">
        {sections.map((section, index) => {
          const isActive = currentTime >= section.start_time && currentTime <= section.end_time;

          return (
            <button
              key={index}
              onClick={() => onSectionClick(section.start_time)}
              className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                isActive
                  ? 'border-red-500 bg-red-50 shadow-md'
                  : 'border-gray-200 bg-white hover:border-red-300 hover:shadow-sm'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-red-600">
                      {section.start_timestamp}
                    </span>
                    <ChevronRight size={14} className="text-gray-400" />
                    <span className="text-sm font-medium text-red-600">
                      {section.end_timestamp}
                    </span>
                  </div>

                  <h3 className="font-semibold text-gray-900 mb-1">
                    {section.title}
                  </h3>

                  <p className="text-sm text-gray-600 mb-2">
                    {section.summary}
                  </p>

                  {section.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {section.keywords.map((keyword, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-gray-100 text-gray-700 rounded-full"
                        >
                          {keyword}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
