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
      <h2 className="text-xl font-bold text-white flex items-center gap-2">
        <Clock className="text-red-500" size={24} />
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
                  ? 'border-red-500 bg-red-950/30 shadow-md'
                  : 'border-gray-700 bg-[#262626] hover:border-red-600/50 hover:shadow-sm'
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-medium text-red-500">
                      {section.start_timestamp}
                    </span>
                    <ChevronRight size={14} className="text-gray-600" />
                    <span className="text-sm font-medium text-red-500">
                      {section.end_timestamp}
                    </span>
                  </div>

                  <h3 className="font-semibold text-white mb-1">
                    {section.title}
                  </h3>

                  <p className="text-sm text-gray-400 mb-2">
                    {section.summary}
                  </p>

                  {section.keywords.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {section.keywords.map((keyword, i) => (
                        <span
                          key={i}
                          className="text-xs px-2 py-1 bg-gray-800 text-gray-300 rounded-full"
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
