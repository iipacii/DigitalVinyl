"use client"

import { useState } from "react"
import type { Track } from "@/types/track"
import { TrackCard } from "@/components/track-card"
import { Input } from "@/components/ui/input"
import { Search, Plus } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { getGenreColor } from "@/lib/utils"
import { EmotionSelector } from "@/components/emotion-selector"

interface LibraryProps {
  tracks: Track[]
  onPlayTrack: (track: Track) => void
  currentTrackId?: string
  isPlaying: boolean
  onTogglePlay: () => void
  onUploadClick: () => void
  activeGenre: string | null
  onGenreSelect: (genre: string | null) => void
  genres: string[]
}

export function Library({
  tracks,
  onPlayTrack,
  currentTrackId,
  isPlaying,
  onTogglePlay,
  onUploadClick,
  activeGenre,
  onGenreSelect,
  genres,
}: LibraryProps) {
  const [searchQuery, setSearchQuery] = useState("")
  const [targetEmotion, setTargetEmotion] = useState<{ arousal: number; valence: number } | null>(null)

  // Calculate emotional distance between two points
  const getEmotionalDistance = (emotion1: { arousal: number; valence: number }, emotion2: { arousal: number; valence: number }) => {
    const dA = emotion1.arousal - emotion2.arousal
    const dV = emotion1.valence - emotion2.valence
    return Math.sqrt(dA * dA + dV * dV) // Euclidean distance
  }

  const filteredTracks = tracks.filter((track) => {
    // Filter by search query
    const matchesSearch =
      track.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      track.artist.toLowerCase().includes(searchQuery.toLowerCase())

    // Filter by genre if active
    const matchesGenre = !activeGenre || track.genre.toLowerCase() === activeGenre.toLowerCase()

    return matchesSearch && matchesGenre
  }).sort((a, b) => {
    // If no emotion target is selected, keep original order
    if (!targetEmotion) return 0

    // Sort by emotional distance
    const distanceA = getEmotionalDistance(targetEmotion, a.emotion)
    const distanceB = getEmotionalDistance(targetEmotion, b.emotion)
    return distanceA - distanceB
  })

  return (
    <div>
      <div className="flex items-center gap-4 mb-8">
        {/* Left side: Search and Emotion Selector */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400" />
            <Input
              type="text"
              placeholder=""
              className="pl-8 pr-3 py-2 h-10 w-40 bg-zinc-800 border-zinc-700 text-white rounded-full"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <EmotionSelector 
            onEmotionChange={(emotion) => setTargetEmotion(emotion)} 
          />
        </div>

        {/* Right side: Genre filters */}
        <div className="flex flex-wrap gap-2">
          {genres.length > 0 && (
            <Badge
              className={`cursor-pointer transform -skew-x-12 ${!activeGenre ? "bg-amber-400 text-black" : "bg-zinc-800 hover:bg-zinc-700 text-white"}`}
              onClick={() => onGenreSelect(null)}
            >
              all
            </Badge>
          )}

          {genres.map((genre) => (
            <Badge
              key={genre}
              className={`cursor-pointer transform -skew-x-12 ${
                activeGenre === genre ? "bg-amber-400 text-black" : getGenreColor(genre)
              }`}
              onClick={() => onGenreSelect(genre)}
            >
              {genre}
            </Badge>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {/* Upload card */}
        <div
          className="aspect-square bg-zinc-900 border border-zinc-800 hover:border-zinc-700 rounded-lg flex flex-col items-center justify-center cursor-pointer transition-all hover:bg-zinc-800"
          onClick={onUploadClick}
        >
          <div className="rounded-full h-16 w-16 bg-zinc-800 flex items-center justify-center">
            <Plus size={32} className="text-amber-400" />
          </div>
        </div>

        {/* Track cards */}
        {filteredTracks.map((track) => (
          <TrackCard
            key={track.id}
            track={track}
            onPlay={() => onPlayTrack(track)}
            isActive={track.id === currentTrackId}
            isPlaying={(isActive) => isActive && isPlaying}
            onTogglePlay={onTogglePlay}
          />
        ))}

        {/* Empty state - only show if no tracks and not filtered */}
        {tracks.length === 0 && searchQuery === "" && !activeGenre && (
          <div className="col-span-full text-center py-12">
            <p className="text-zinc-400 transform -skew-x-12">click the + to add your first track</p>
          </div>
        )}
      </div>
    </div>
  )
}
