"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import type { Track } from "@/types/track"
import { Music, UploadIcon, ImageIcon } from "lucide-react"

interface UploadProps {
  onTrackAdded: (track: Track) => void
}

export function Upload({ onTrackAdded }: UploadProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)

  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0])
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!audioFile) return

    setIsProcessing(true)

    // Simulate processing with progress
    let currentProgress = 0
    const interval = setInterval(() => {
      currentProgress += 5
      setProgress(currentProgress)

      if (currentProgress >= 100) {
        clearInterval(interval)

        // Create a new track with fixed emotion values as specified
        const newTrack: Track = {
          id: Date.now().toString(),
          title: audioFile.name.replace(/\.[^/.]+$/, ""),
          artist: "Unknown Artist",
          audioUrl: URL.createObjectURL(audioFile),
          imageUrl: imageFile ? URL.createObjectURL(imageFile) : "/placeholder.svg",
          emotion: {
            arousal: 0.8,
            valence: 0.6,
          },
          genre: "Electronic",
          uploadedAt: new Date(),
        }

        onTrackAdded(newTrack)
        setIsProcessing(false)
        setAudioFile(null)
        setImageFile(null)
        setProgress(0)
      }
    }, 100)
  }

  return (
    <div className="bg-amber-300 rounded-lg p-6 shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Upload Your Music</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="audio" className="flex items-center gap-2">
            <Music size={18} /> Audio File (MP3, WAV)
          </Label>
          <Input
            id="audio"
            type="file"
            accept=".mp3,.wav"
            onChange={handleAudioChange}
            className="bg-amber-50"
            disabled={isProcessing}
          />
          {audioFile && <p className="text-sm font-medium">Selected: {audioFile.name}</p>}
        </div>

        <div className="space-y-2">
          <Label htmlFor="image" className="flex items-center gap-2">
            <ImageIcon size={18} /> Cover Image (Optional)
          </Label>
          <Input
            id="image"
            type="file"
            accept="image/*"
            onChange={handleImageChange}
            className="bg-amber-50"
            disabled={isProcessing}
          />
          {imageFile && <p className="text-sm font-medium">Selected: {imageFile.name}</p>}
        </div>

        {isProcessing ? (
          <div className="space-y-2">
            <p className="font-medium">{progress < 50 ? "Analyzing audio..." : "Encoding digital vinyl..."}</p>
            <Progress value={progress} className="h-2 bg-amber-200" />
          </div>
        ) : (
          <Button type="submit" className="w-full bg-black hover:bg-gray-800 text-amber-400" disabled={!audioFile}>
            <UploadIcon className="mr-2 h-4 w-4" /> Upload & Process
          </Button>
        )}
      </form>
    </div>
  )
}
