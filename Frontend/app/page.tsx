"use client"

import { useState, useEffect } from "react"
import { Library } from "@/components/library"
import { MediaPlayer } from "@/components/media-player"
import { UploadTrack } from "@/components/upload-track"
import type { Track } from "@/types/track"
import Image from "next/image"

// Create a silent audio blob for placeholder
const createSilentAudio = () => {
  // Create a silent audio context
  const audioContext = new AudioContext()
  const buffer = audioContext.createBuffer(1, 44100, 44100)
  const source = audioContext.createBufferSource()
  source.buffer = buffer
  source.connect(audioContext.destination)

  // Export to a blob
  const offlineContext = new OfflineAudioContext(1, 44100, 44100)
  const offlineSource = offlineContext.createBufferSource()
  offlineSource.buffer = buffer
  offlineSource.connect(offlineContext.destination)
  offlineSource.start()

  return offlineContext.startRendering().then((renderedBuffer) => {
    const wavBlob = bufferToWave(renderedBuffer, 44100)
    return URL.createObjectURL(wavBlob)
  })
}

// Helper function to convert AudioBuffer to WAV Blob
function bufferToWave(abuffer: AudioBuffer, len: number) {
  const numOfChan = abuffer.numberOfChannels
  const length = len * numOfChan * 2 + 44
  const buffer = new ArrayBuffer(length)
  const view = new DataView(buffer)
  const channels = []
  let i, sample
  let offset = 0

  // write WAVE header
  setUint32(0x46464952) // "RIFF"
  setUint32(length - 8) // file length - 8
  setUint32(0x45564157) // "WAVE"
  setUint32(0x20746d66) // "fmt " chunk
  setUint32(16) // length = 16
  setUint16(1) // PCM (uncompressed)
  setUint16(numOfChan)
  setUint32(abuffer.sampleRate)
  setUint32(abuffer.sampleRate * 2 * numOfChan) // avg. bytes/sec
  setUint16(numOfChan * 2) // block-align
  setUint16(16) // 16-bit
  setUint32(0x61746164) // "data" - chunk
  setUint32(length - offset - 4) // chunk length

  // write interleaved data
  for (i = 0; i < abuffer.numberOfChannels; i++) {
    channels.push(abuffer.getChannelData(i))
  }

  while (offset < length) {
    for (i = 0; i < numOfChan; i++) {
      // interleave channels
      sample = Math.max(-1, Math.min(1, channels[i][offset / (numOfChan * 2)])) // clamp
      sample = (0.5 + sample < 0 ? sample * 32768 : sample * 32767) | 0 // scale to 16-bit signed int
      view.setInt16(offset, sample, true) // write 16-bit sample
      offset += 2
    }
  }

  function setUint16(data: number) {
    view.setUint16(offset, data, true)
    offset += 2
  }

  function setUint32(data: number) {
    view.setUint32(offset, data, true)
    offset += 4
  }

  return new Blob([buffer], { type: "audio/wav" })
}

export default function Home() {
  const [tracks, setTracks] = useState<Track[]>([])
  const [currentTrack, setCurrentTrack] = useState<Track | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [activeGenre, setActiveGenre] = useState<string | null>(null)
  const [placeholderAudio, setPlaceholderAudio] = useState<string | null>(null)

  // Generate placeholder audio on mount - FIXED: changed useState to useEffect
  useEffect(() => {
    console.log("Generating placeholder audio...")
    createSilentAudio().then((url) => {
      console.log("Placeholder audio created:", url)
      setPlaceholderAudio(url)
    })
  }, []) // Empty dependency array means this runs once on mount

  // Get unique genres from tracks
  const genres = Array.from(new Set(tracks.map((track) => track.genre.toLowerCase())))

  const addTrack = (track: Track) => {
    console.log("Adding track to library:", track)
    console.log("Image URL type:", typeof track.imageUrl)
    console.log("Image URL starts with:", track.imageUrl.substring(0, 30) + "...")
    
    // If the track uses a placeholder audio URL, replace it with our generated one
    if (track.audioUrl === "/placeholder.mp3" && placeholderAudio) {
      console.log("Replacing placeholder audio with generated silent audio")
      track.audioUrl = placeholderAudio
    }

    // Ensure the imageUrl is properly set and log it
    if (!track.imageUrl || track.imageUrl === "undefined") {
      console.warn("Track has no valid imageUrl, using placeholder")
      track.imageUrl = "/placeholder.svg"
    } else if (track.imageUrl.startsWith("data:image/")) {
      console.log("Track has a valid base64 image")
    } else {
      console.log("Track has a regular URL image:", track.imageUrl)
    }

    setTracks((prev) => {
      console.log("Library now has", prev.length + 1, "tracks")
      return [track, ...prev]
    })
  }

  const handlePlayTrack = (track: Track) => {
    setCurrentTrack(track)
    setIsPlaying(true)
  }

  return (
    <main className="min-h-screen bg-black text-white pb-24">
      <div className="container mx-auto px-4 py-8">
        <div className="flex justify-between items-center mb-12">
          <div className="flex items-center gap-3">
            <Image 
              src="/Stamp.png"
              alt="Vinyl logo"
              width={48}
              height={48}
              className="object-contain"
            />
            <h1 className="text-6xl font-bold">vinyl</h1>
          </div>
        </div>

        <Library
          tracks={tracks}
          onPlayTrack={handlePlayTrack}
          currentTrackId={currentTrack?.id}
          isPlaying={isPlaying}
          onTogglePlay={() => setIsPlaying(!isPlaying)}
          onUploadClick={() => setIsUploading(true)}
          activeGenre={activeGenre}
          onGenreSelect={setActiveGenre}
          genres={genres}
        />

        {isUploading && <UploadTrack onTrackAdded={addTrack} onClose={() => setIsUploading(false)} />}

        {currentTrack && (
          <MediaPlayer
            track={currentTrack}
            isPlaying={isPlaying}
            onPlayPause={setIsPlaying}
            onTrackChange={(track) => {
              const currentIndex = tracks.findIndex((t) => t.id === currentTrack.id)
              if (track === "next" && currentIndex < tracks.length - 1) {
                setCurrentTrack(tracks[currentIndex + 1])
              } else if (track === "prev" && currentIndex > 0) {
                setCurrentTrack(tracks[currentIndex - 1])
              }
            }}
          />
        )}
      </div>
    </main>
  )
}
