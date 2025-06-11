'use client';

import React, { useRef, useState } from 'react';
import ReactPlayer from 'react-player/youtube';
import { Play, Pause, Volume2, VolumeX, Maximize } from 'lucide-react';
import { formatTime } from '@/lib/utils';

interface VideoPlayerProps {
  videoId: string;
  onTimeUpdate?: (seconds: number) => void;
  seekTo?: number;
}

export default function VideoPlayer({ videoId, onTimeUpdate, seekTo }: VideoPlayerProps) {
  const playerRef = useRef<ReactPlayer>(null);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(0.8);
  const [muted, setMuted] = useState(false);
  const [played, setPlayed] = useState(0);
  const [duration, setDuration] = useState(0);

  React.useEffect(() => {
    if (seekTo !== undefined && playerRef.current) {
      playerRef.current.seekTo(seekTo, 'seconds');
      setPlaying(true);
    }
  }, [seekTo]);

  const handleProgress = (state: any) => {
    setPlayed(state.played);
    if (onTimeUpdate) {
      onTimeUpdate(state.playedSeconds);
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newPlayed = parseFloat(e.target.value);
    setPlayed(newPlayed);
    playerRef.current?.seekTo(newPlayed);
  };

  const url = `https://www.youtube.com/watch?v=${videoId}`;

  return (
    <div className="relative bg-black rounded-lg overflow-hidden shadow-2xl">
      <div className="aspect-video">
        <ReactPlayer
          ref={playerRef}
          url={url}
          playing={playing}
          volume={volume}
          muted={muted}
          width="100%"
          height="100%"
          onProgress={handleProgress}
          onDuration={setDuration}
          progressInterval={100}
        />
      </div>

      {/* Custom Controls */}
      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/90 to-transparent p-4">
        {/* Progress Bar */}
        <input
          type="range"
          min={0}
          max={1}
          step={0.001}
          value={played}
          onChange={handleSeek}
          className="w-full h-1 mb-3 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
        />

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Play/Pause */}
            <button
              onClick={() => setPlaying(!playing)}
              className="text-white hover:text-blue-400 transition-colors"
            >
              {playing ? <Pause size={24} /> : <Play size={24} />}
            </button>

            {/* Volume */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => setMuted(!muted)}
                className="text-white hover:text-blue-400 transition-colors"
              >
                {muted ? <VolumeX size={20} /> : <Volume2 size={20} />}
              </button>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="w-20 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
              />
            </div>

            {/* Time */}
            <span className="text-white text-sm">
              {formatTime(played * duration)} / {formatTime(duration)}
            </span>
          </div>

          {/* Fullscreen */}
          <button className="text-white hover:text-blue-400 transition-colors">
            <Maximize size={20} />
          </button>
        </div>
      </div>
    </div>
  );
}
