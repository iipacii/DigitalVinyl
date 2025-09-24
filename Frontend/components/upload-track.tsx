"use client"

import type React from "react"

import { useState, useRef, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import type { Track } from "@/types/track"
import { Music, ImageIcon, X, Upload, FileAudio } from "lucide-react"

interface UploadTrackProps {
  onTrackAdded: (track: Track) => void
  onClose: () => void
}

export function UploadTrack({ onTrackAdded, onClose }: UploadTrackProps) {
  // Standard upload states
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [imageBase64, setImageBase64] = useState<string>("")

  // Encoded image upload states
  const [encodedImageBase64, setEncodedImageBase64] = useState<string>("")

  // Common states
  const [isProcessing, setIsProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [step, setStep] = useState<"select" | "audio" | "cover" | "encoded" | "processing">("select")
  const [emotion, setEmotion] = useState<{ arousal: number; valence: number } | null>(null)

  // Refs
  const audioInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const encodedImageInputRef = useRef<HTMLInputElement>(null)
  const dropZoneRef = useRef<HTMLDivElement>(null)
  const encodedDropZoneRef = useRef<HTMLDivElement>(null)

  // Complete standard upload (audio + image) with a specific image
  const completeStandardUploadWithImage = async (imageData: string) => {
    if (!audioFile || !emotion) {
      console.error("Cannot complete upload: missing required data", { 
        hasAudioFile: !!audioFile,
        hasEmotion: !!emotion
      })
      return
    }

    setIsProcessing(true)
    setProgress(0)

    try {
      // Create FormData with the files
      const formData = new FormData()
      formData.append('audio_file', audioFile)
      
      // If we have image data, convert base64 to blob and append
      if (imageData && imageData.startsWith('data:image')) {
        const response = await fetch(imageData)
        const blob = await response.blob()
        formData.append('cover_image', blob)
      }
      
      formData.append('song_name', audioFile.name.replace(/\.[^/.]+$/, ""))

      // Send to encode endpoint
      const encodeResponse = await fetch('http://localhost:8000/encode', {
        method: 'POST',
        body: formData
      })

      if (!encodeResponse.ok) {
        throw new Error('Failed to encode audio')
      }

      const { encoded_image_url, genre = "unknown", genre_confidence, valence, arousal } = await encodeResponse.json()

      // Create a new track with the encoded image URL and emotion values from backend
      const newTrack: Track = {
        id: Date.now().toString(),
        title: audioFile.name.replace(/\.[^/.]+$/, ""),
        artist: "unknown artist",
        audioUrl: URL.createObjectURL(audioFile), // Keep original audio for immediate playback
        imageUrl: `http://localhost:8000${encoded_image_url}`,
        emotion: {
          arousal: arousal,
          valence: valence
        },
        genre: genre,  // Using the genre from the backend
        uploadedAt: new Date(),
      }

      onTrackAdded(newTrack)
      onClose()
    } catch (error) {
      console.error('Error during upload:', error)
      setIsProcessing(false)
    }
  }

  // Extract audio from an encoded image
  const extractAudioFromImage = async (file: File, imageBase64: string) => {
    setIsProcessing(true)
    setProgress(0)

    try {
      // Create FormData with the encoded image
      const formData = new FormData()
      formData.append('encoded_image', file)

      // Send to decode endpoint
      const decodeResponse = await fetch('http://localhost:8000/decode', {
        method: 'POST',
        body: formData
      })

      if (!decodeResponse.ok) {
        throw new Error('Failed to decode audio')
      }

      const { decoded_audio_url, song_name, genre = "unknown", valence, arousal } = await decodeResponse.json()

      // Create a new track with the decoded audio and emotion values from backend
      const newTrack: Track = {
        id: Date.now().toString(),
        title: song_name || file.name.replace(/\.[^/.]+$/, ""),
        artist: "decoded vinyl",
        audioUrl: `http://localhost:8000${decoded_audio_url}`,
        imageUrl: imageBase64,
        emotion: {
          arousal: arousal,
          valence: valence
        },
        genre: genre,
        uploadedAt: new Date(),
      }

      onTrackAdded(newTrack)
      onClose()
    } catch (error) {
      console.error('Error during decode:', error)
      setIsProcessing(false)
    }
  }

  // Standard image upload handler
  const handleImageFile = useCallback((file: File) => {
    console.log("handleImageFile called with file:", file.name, "type:", file.type, "size:", file.size)

    // Convert image to base64
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        const base64 = e.target.result as string
        console.log("Image converted to base64, size:", base64.length, "starts with:", base64.substring(0, 50) + "...")

        // Store the base64 data in state
        setImageBase64(base64)

        // IMPORTANT: We need to use the base64 value directly rather than depending on state update
        // because React state updates are asynchronous and might not be available immediately
        if (emotion) {
          console.log("Emotion data already present, completing upload with base64 image")
          completeStandardUploadWithImage(base64)
        } else {
          console.log("Waiting for emotion data before completing upload")
        }
      }
    }
    reader.onerror = (error) => {
      console.error("Error reading image file:", error)
    }
    reader.readAsDataURL(file)
  }, [emotion, completeStandardUploadWithImage])

  // Encoded image upload handler
  const handleEncodedImageFile = useCallback((file: File) => {
    setIsProcessing(true)
    setProgress(0)

    // Convert image to base64
    const reader = new FileReader()
    reader.onload = (e) => {
      if (e.target?.result) {
        const base64 = e.target.result as string
        setEncodedImageBase64(base64)
        extractAudioFromImage(file, base64)
      }
    }
    reader.readAsDataURL(file)
  }, [extractAudioFromImage])

  // Handle drag and drop for standard image
  useEffect(() => {
    const dropZone = dropZoneRef.current
    if (!dropZone || step !== "cover") return

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.add("border-amber-400")
    }

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.remove("border-amber-400")
    }

    const handleDrop = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.remove("border-amber-400")

      if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0]
        if (file.type.startsWith("image/")) {
          handleImageFile(file)
        }
      }
    }

    dropZone.addEventListener("dragover", handleDragOver)
    dropZone.addEventListener("dragleave", handleDragLeave)
    dropZone.addEventListener("drop", handleDrop)

    return () => {
      dropZone.removeEventListener("dragover", handleDragOver)
      dropZone.removeEventListener("dragleave", handleDragLeave)
      dropZone.removeEventListener("drop", handleDrop)
    }
  }, [step, handleImageFile])

  // Handle drag and drop for encoded image
  useEffect(() => {
    const dropZone = encodedDropZoneRef.current
    if (!dropZone || step !== "encoded") return

    const handleDragOver = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.add("border-amber-400")
    }

    const handleDragLeave = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.remove("border-amber-400")
    }

    const handleDrop = (e: DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      dropZone.classList.remove("border-amber-400")

      if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
        const file = e.dataTransfer.files[0]
        if (file.type.startsWith("image/")) {
          handleEncodedImageFile(file)
        }
      }
    }

    dropZone.addEventListener("dragover", handleDragOver)
    dropZone.addEventListener("dragleave", handleDragLeave)
    dropZone.addEventListener("drop", handleDrop)

    return () => {
      dropZone.removeEventListener("dragover", handleDragOver)
      dropZone.removeEventListener("dragleave", handleDragLeave)
      dropZone.removeEventListener("drop", handleDrop)
    }
  }, [step, handleEncodedImageFile])

  // Standard audio upload handler
  const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setAudioFile(file)
      setStep("cover")
      processAudio()
    }
  }

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleImageFile(e.target.files[0])
    }
  }

  const handleEncodedImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleEncodedImageFile(e.target.files[0])
    }
  }

  // Process audio for standard upload
  const processAudio = () => {
    setIsProcessing(true)

    // Simulate audio analysis to get emotion values
    let currentProgress = 0
    const interval = setInterval(() => {
      currentProgress += 5
      setProgress(currentProgress)

      if (currentProgress >= 100) {
        clearInterval(interval)

        // Generate random emotion values (in a real app, this would come from analysis)
        const newEmotion = {
          arousal: Math.random(),
          valence: Math.random(),
        }
        setEmotion(newEmotion)
        setIsProcessing(false)

        // If we already have an image, complete the process
        // but use the direct state value to avoid timing issues
        if (imageBase64) {
          console.log("Image already uploaded, completing with existing image")
          completeStandardUploadWithImage(imageBase64)
        }
      }
    }, 100)
  }

  // Original function kept for backward compatibility
  const completeStandardUpload = () => {
    if (imageBase64) {
      completeStandardUploadWithImage(imageBase64)
    } else {
      console.warn("No image data in state, using placeholder")
      completeStandardUploadWithImage("")
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50 p-4">
      <div className="bg-zinc-900 border border-zinc-800 rounded-lg p-6 max-w-md w-full relative">
        <Button
          variant="ghost"
          size="icon"
          className="absolute right-2 top-2 text-zinc-500 hover:text-white"
          onClick={onClose}
        >
          <X size={18} />
        </Button>

        <h2 className="text-xl mb-6 transform -skew-x-12">upload track</h2>

        {step === "select" && (
          <div className="space-y-6">
            <p className="text-zinc-400 text-sm mb-4">Choose how you want to add music to your library:</p>

            <div className="grid grid-cols-2 gap-4">
              <div
                className="flex flex-col items-center justify-center p-4 border border-zinc-800 rounded-lg hover:border-amber-400 cursor-pointer transition-colors"
                onClick={() => setStep("audio")}
              >
                <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center mb-2">
                  <FileAudio size={24} className="text-amber-400" />
                </div>
                <p className="text-sm text-center transform -skew-x-12">upload audio + cover</p>
              </div>

              <div
                className="flex flex-col items-center justify-center p-4 border border-zinc-800 rounded-lg hover:border-amber-400 cursor-pointer transition-colors"
                onClick={() => setStep("encoded")}
              >
                <div className="h-12 w-12 rounded-full bg-zinc-800 flex items-center justify-center mb-2">
                  <Upload size={24} className="text-amber-400" />
                </div>
                <p className="text-sm text-center transform -skew-x-12">upload encoded image</p>
              </div>
            </div>
          </div>
        )}

        {step === "audio" && (
          <div className="space-y-4">
            <div className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg hover:border-amber-400 transition-colors">
              <Music size={32} className="text-zinc-500 mb-2" />
              <input
                ref={audioInputRef}
                type="file"
                accept=".mp3,.wav"
                onChange={handleAudioChange}
                className="hidden"
              />
              <Button variant="ghost" onClick={() => audioInputRef.current?.click()} className="mt-2">
                <span className="transform -skew-x-12">browse</span>
              </Button>
            </div>
          </div>
        )}

        {step === "cover" && (
          <div className="space-y-4">
            <div
              ref={dropZoneRef}
              className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg hover:border-amber-400 transition-colors"
            >
              {imageBase64 ? (
                <div className="w-full h-full relative">
                  <img
                    src={imageBase64 || "/placeholder.svg"}
                    alt="Cover preview"
                    className="w-full h-full object-contain"
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    className="absolute top-2 right-2 bg-black bg-opacity-50 hover:bg-opacity-70 rounded-full h-8 w-8 p-0"
                    onClick={() => {
                      setImageBase64("")
                    }}
                  >
                    <X size={14} />
                  </Button>
                </div>
              ) : (
                <>
                  <ImageIcon size={32} className="text-zinc-500 mb-2" />
                  <p className="text-zinc-400 transform -skew-x-12">drag & drop or select cover</p>
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageChange}
                    className="hidden"
                  />
                  <Button variant="ghost" onClick={() => imageInputRef.current?.click()} className="mt-2">
                    <span className="transform -skew-x-12">browse</span>
                  </Button>
                </>
              )}
            </div>

            <div className="flex justify-between">
              <Button
                variant="ghost"
                onClick={() => {
                  if (emotion) {
                    completeStandardUpload()
                  } else {
                    setStep("processing")
                  }
                }}
              >
                <span className="transform -skew-x-12">skip</span>
              </Button>

              {isProcessing && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-zinc-500 transform -skew-x-12">analyzing audio</span>
                  <Progress value={progress} className="w-24 h-1 bg-zinc-800" />
                </div>
              )}
            </div>
          </div>
        )}

        {step === "encoded" && (
          <div className="space-y-4">
            <div
              ref={encodedDropZoneRef}
              className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-zinc-800 rounded-lg hover:border-amber-400 transition-colors"
            >
              {encodedImageBase64 ? (
                <div className="w-full h-full relative">
                  <img
                    src={encodedImageBase64 || "/placeholder.svg"}
                    alt="Encoded image preview"
                    className="w-full h-full object-contain"
                  />
                  {isProcessing && (
                    <div className="absolute inset-0 bg-black bg-opacity-50 flex flex-col items-center justify-center">
                      <p className="text-white text-sm mb-2 transform -skew-x-12">
                        {progress < 50 ? "Decoding audio..." : "Extracting data..."}
                      </p>
                      <Progress value={progress} className="w-32 h-1 bg-zinc-800" />
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <Upload size={32} className="text-zinc-500 mb-2" />
                  <p className="text-zinc-400 transform -skew-x-12">drag & drop or select encoded image</p>
                  <p className="text-zinc-500 text-xs text-center mt-1 px-4">
                    Upload an image with embedded audio data
                  </p>
                  <input
                    ref={encodedImageInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleEncodedImageChange}
                    className="hidden"
                  />
                  <Button variant="ghost" onClick={() => encodedImageInputRef.current?.click()} className="mt-2">
                    <span className="transform -skew-x-12">browse</span>
                  </Button>
                </>
              )}
            </div>
          </div>
        )}

        {step === "processing" && (
          <div className="space-y-4">
            <div className="text-center py-8">
              <p className="text-zinc-400 transform -skew-x-12 mb-4">analyzing audio...</p>
              <Progress value={progress} className="h-1 bg-zinc-800" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
