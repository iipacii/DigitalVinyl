"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Dialog, DialogContent } from "@/components/ui/dialog"
import type { Track } from "@/types/track"
import { Music, UploadIcon, ImageIcon } from "lucide-react"

interface UploadDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  onTrackAdded: (track: Track) => void
}

export function UploadDialog({ open, onOpenChange, onTrackAdded }: UploadDialogProps) {
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
          artist: "unknown artist",
          audioUrl: URL.createObjectURL(audioFile),
          imageUrl: imageFile ? URL.createObjectURL(imageFile) : "/placeholder.svg",
          emotion: {
            arousal: 0.8,
            valence: 0.6,
          },
          genre: "electronic",
          uploadedAt: new Date(),
        }

        onTrackAdded(newTrack)
        setIsProcessing(false)
        setAudioFile(null)
        setImageFile(null)
        setProgress(0)
        onOpenChange(false)
      }
    }, 100)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-zinc-900 border-zinc-800 text-white">
        <form onSubmit={handleSubmit} className="space-y-6 pt-4">
          <div className="flex items-center justify-center gap-6">
            <div className="flex flex-col items-center">
              <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center mb-2">
                <Music size={24} className="text-amber-400" />
              </div>
              <label className="transform -skew-x-12 text-xs text-zinc-400">audio</label>
              <Input
                id="audio"
                type="file"
                accept=".mp3,.wav"
                onChange={handleAudioChange}
                className="sr-only"
                disabled={isProcessing}
              />
              <Button
                type="button"
                onClick={() => document.getElementById("audio")?.click()}
                variant="outline"
                className="mt-2 h-8 bg-zinc-800 border-zinc-700 hover:bg-zinc-700"
                disabled={isProcessing}
              >
                {audioFile ? "✓" : "+"}
              </Button>
            </div>

            <div className="flex flex-col items-center">
              <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center mb-2">
                <ImageIcon size={24} className="text-amber-400" />
              </div>
              <label className="transform -skew-x-12 text-xs text-zinc-400">cover</label>
              <Input
                id="image"
                type="file"
                accept="image/*"
                onChange={handleImageChange}
                className="sr-only"
                disabled={isProcessing}
              />
              <Button
                type="button"
                onClick={() => document.getElementById("image")?.click()}
                variant="outline"
                className="mt-2 h-8 bg-zinc-800 border-zinc-700 hover:bg-zinc-700"
                disabled={isProcessing}
              >
                {imageFile ? "✓" : "+"}
              </Button>
            </div>
          </div>

          {isProcessing ? (
            <div className="space-y-2">
              <Progress value={progress} className="h-1 bg-zinc-800" />
            </div>
          ) : (
            <Button type="submit" className="w-full bg-amber-400 hover:bg-amber-300 text-black" disabled={!audioFile}>
              <UploadIcon className="mr-2 h-4 w-4" />
              <span className="transform -skew-x-12">upload</span>
            </Button>
          )}
        </form>
      </DialogContent>
    </Dialog>
  )
}
