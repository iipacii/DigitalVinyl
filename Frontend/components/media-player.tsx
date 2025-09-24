"use client"

import { useState, useRef, useEffect } from "react"
import type { Track } from "@/types/track"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Play, Pause, SkipBack, SkipForward, Volume2 } from "lucide-react"
import { formatTime } from "@/lib/utils"

interface MediaPlayerProps {
  track: Track
  isPlaying: boolean
  onPlayPause: (playing: boolean) => void
  onTrackChange: (direction: "next" | "prev") => void
}

export function MediaPlayer({ track, isPlaying, onPlayPause, onTrackChange }: MediaPlayerProps) {
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(80)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  useEffect(() => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.play().catch((err) => console.error("Playback failed:", err))
      } else {
        audioRef.current.pause()
      }
    }
  }, [isPlaying, track.id])

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume / 100
    }
  }, [volume])

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleSeek = (value: number[]) => {
    if (audioRef.current) {
      audioRef.current.currentTime = value[0]
      setCurrentTime(value[0])
    }
  }

  const togglePlay = () => {
    onPlayPause(!isPlaying)
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-zinc-900 border-t border-zinc-800 p-3 z-50">
      <div className="container mx-auto flex items-center gap-4">
        <div className="relative h-12 w-12 overflow-hidden rounded-md flex-shrink-0">
          <img src={track.imageUrl || "/placeholder.svg"} alt="" className="h-full w-full object-cover" />
        </div>

        <div className="flex-1 min-w-0 mr-4">
          <div className="flex justify-between items-center mb-1">
            <div>
              <h3 className="font-medium text-sm truncate text-white">{track.title}</h3>
              <p className="text-zinc-400 text-xs transform -skew-x-12">{track.artist}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-zinc-400 transform -skew-x-12">{formatTime(currentTime)}</span>
            <div className="flex-1 relative">
              <Slider
                value={[currentTime]}
                max={duration || 100}
                step={0.1}
                onValueChange={handleSeek}
                className="z-10 relative [&_.absolute]:bg-amber-400/30 [&_[role=slider]]:bg-amber-400 [&_[role=slider]]:border-amber-400/50 [&_.relative]:bg-amber-400/20"
              />
              {/* Track duration background indicator */}
              <div
                className="absolute top-1/2 left-0 h-1 bg-zinc-800 -translate-y-1/2 rounded-full"
                style={{ width: "100%" }}
              ></div>
            </div>
            <span className="text-xs text-zinc-400 transform -skew-x-12">{formatTime(duration)}</span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            onClick={() => onTrackChange("prev")}
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-zinc-400 hover:text-amber-400 hover:bg-zinc-800"
          >
            <SkipBack size={16} />
          </Button>

          <Button
            onClick={togglePlay}
            variant="ghost"
            size="icon"
            className="h-10 w-10 rounded-full bg-zinc-800 text-amber-400 hover:bg-zinc-700"
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
          </Button>

          <Button
            onClick={() => onTrackChange("next")}
            variant="ghost"
            size="icon"
            className="h-8 w-8 text-zinc-400 hover:text-amber-400 hover:bg-zinc-800"
          >
            <SkipForward size={16} />
          </Button>
        </div>

        <div className="flex items-center gap-2 w-32">
          <Volume2 size={14} className="text-zinc-400" />
          <Slider 
            value={[volume]} 
            max={100} 
            step={1} 
            onValueChange={(value) => setVolume(value[0])} 
            className="[&_.absolute]:bg-amber-400/30 [&_[role=slider]]:bg-amber-400 [&_[role=slider]]:border-amber-400/50 [&_.relative]:bg-amber-400/20"
          />
        </div>

        <audio
          ref={audioRef}
          src={track.audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => onPlayPause(false)}
          className="hidden"
        />
      </div>
    </div>
  )
}
