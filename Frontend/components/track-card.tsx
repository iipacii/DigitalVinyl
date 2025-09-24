"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import type { Track } from "@/types/track"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Play, Pause, Download, Disc, Info } from "lucide-react"
import { EmotionVisualizer } from "@/components/emotion-visualizer"
import { Particles } from "@/components/particles"
import { getGenreColor } from "@/lib/utils"

interface TrackCardProps {
  track: Track
  onPlay: () => void
  isActive: boolean
  isPlaying: (isActive: boolean) => boolean
  onTogglePlay: () => void
}

export function TrackCard({ track, onPlay, isActive, isPlaying, onTogglePlay }: TrackCardProps) {
  const [rotation, setRotation] = useState(0)
  const [isFlipped, setIsFlipped] = useState(false)
  const [titleWidth, setTitleWidth] = useState(0)
  const [containerWidth, setContainerWidth] = useState(0)
  const titleRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const animationRef = useRef<number | null>(null)
  const lastTimeRef = useRef<number>(0)
  const isCurrentlyPlaying = isPlaying(isActive)
  const [titlePosition, setTitlePosition] = useState(0)
  const [titleDirection, setTitleDirection] = useState(-1) // -1 for left, 1 for right
  const [needsScroll, setNeedsScroll] = useState(false)

  // Measure title and container widths
  useEffect(() => {
    if (titleRef.current && containerRef.current) {
      const titleWidth = titleRef.current.scrollWidth
      const containerWidth = containerRef.current.clientWidth
      setTitleWidth(titleWidth)
      setContainerWidth(containerWidth)
      setNeedsScroll(titleWidth > containerWidth)
    }
  }, [track.title])

  // Reset rotation when track changes or stops playing
  useEffect(() => {
    if (!isCurrentlyPlaying) {
      // Reset lastTime when stopped
      lastTimeRef.current = 0
      // Reset title position when stopped
      setTitlePosition(0)
    }
  }, [isCurrentlyPlaying])

  // Handle title scrolling
  useEffect(() => {
    if (isCurrentlyPlaying && needsScroll) {
      const scrollTitle = () => {
        setTitlePosition((prev) => {
          // Calculate new position
          const newPos = prev + titleDirection * 0.5

          // Check if we need to change direction
          if (newPos <= -(titleWidth - containerWidth + 20) && titleDirection === -1) {
            setTitleDirection(1)
            return -(titleWidth - containerWidth + 20)
          } else if (newPos >= 0 && titleDirection === 1) {
            setTitleDirection(-1)
            return 0
          }

          return newPos
        })

        requestAnimationFrame(scrollTitle)
      }

      const animationId = requestAnimationFrame(scrollTitle)
      return () => cancelAnimationFrame(animationId)
    }
  }, [isCurrentlyPlaying, needsScroll, titleWidth, containerWidth, titleDirection])

  // Handle rotation animation
  useEffect(() => {
    // Clean up previous animation frame
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }

    if (isCurrentlyPlaying && !isFlipped) {
      const animate = (time: number) => {
        if (lastTimeRef.current === 0) {
          lastTimeRef.current = time
        }

        const delta = time - lastTimeRef.current
        lastTimeRef.current = time

        // Faster rotation - 45 RPM
        setRotation((prev) => (prev + delta / 100) % 360)
        animationRef.current = requestAnimationFrame(animate)
      }

      animationRef.current = requestAnimationFrame(animate)
    } else if (!isCurrentlyPlaying && !isFlipped) {
      // Deceleration animation when stopped
      const slowDown = () => {
        setRotation((prev) => {
          if (Math.abs(prev) < 0.5) return 0
          return prev * 0.95 // Slow down factor
        })

        if (rotation !== 0) {
          animationRef.current = requestAnimationFrame(slowDown)
        }
      }

      if (rotation !== 0) {
        animationRef.current = requestAnimationFrame(slowDown)
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isCurrentlyPlaying, isFlipped, rotation])

  const handleDownload = (e: React.MouseEvent) => {
    e.stopPropagation()

    // For base64 images, we need to handle download differently
    if (track.imageUrl.startsWith("data:")) {
      // Create a temporary link
      const link = document.createElement("a")
      link.href = track.imageUrl
      link.download = `${track.title}-vinyl.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } else {
      // For regular URLs, open in new tab
      window.open(track.imageUrl, "_blank")
    }
  }

  const handleCardClick = () => {
    if (!isActive) {
      onPlay()
    } else {
      onTogglePlay()
    }
  }

  return (
    <Card
      className={`overflow-hidden bg-zinc-900 border-zinc-800 hover:border-zinc-700 transition-all ${
        isActive ? "ring-1 ring-amber-400" : ""
      }`}
    >
      <div className="relative aspect-square overflow-hidden bg-black group">
        {/* Flip card container */}
        <div
          className={`w-full h-full transition-all duration-700 ${isFlipped ? "rotate-y-180" : ""}`}
          style={{ transformStyle: "preserve-3d" }}
        >
          {/* Front side - Album cover */}
          <div className="absolute w-full h-full backface-hidden" style={{ backfaceVisibility: "hidden" }}>
            {/* Particles effect when playing */}
            {isCurrentlyPlaying && !isFlipped && <Particles />}

            <div
              className={`w-full h-full transition-all duration-500 ${isCurrentlyPlaying && !isFlipped ? "rounded-full" : ""}`}
              style={{
                transform: isFlipped ? "none" : `rotate(${rotation}deg)`,
                transformOrigin: "center",
                transition:
                  isCurrentlyPlaying && !isFlipped ? "none" : "transform 0.5s ease-out, border-radius 0.5s ease-out",
              }}
            >
              <img
                src={track.imageUrl || "/placeholder.svg"}
                alt={track.title}
                className="w-full h-full object-cover"
                onError={(e) => {
                  console.error("Image failed to load:", track.imageUrl)
                  ;(e.target as HTMLImageElement).src = "/placeholder.svg"
                }}
              />

              {/* Vinyl grooves */}
              {(isCurrentlyPlaying || isActive) && (
                <div className="absolute inset-0 pointer-events-none">
                  {/* Concentric circles to simulate vinyl grooves */}
                  <div className="absolute inset-0 rounded-full border border-black opacity-10"></div>
                  <div className="absolute inset-[10%] rounded-full border border-black opacity-10"></div>
                  <div className="absolute inset-[20%] rounded-full border border-black opacity-10"></div>
                  <div className="absolute inset-[30%] rounded-full border border-black opacity-10"></div>
                  <div className="absolute inset-[40%] rounded-full border border-black opacity-10"></div>
                  <div className="absolute inset-[45%] rounded-full border-2 border-black opacity-20"></div>
                  <div className="absolute inset-[48%] rounded-full border border-black opacity-30"></div>
                </div>
              )}
            </div>

            {/* Center hole when playing */}
            {isCurrentlyPlaying && !isFlipped && (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-4 h-4 bg-black rounded-full z-10"></div>
            )}

            {/* Vinyl label */}
            {isCurrentlyPlaying && !isFlipped && (
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-16 h-16 bg-zinc-900 rounded-full z-5 flex items-center justify-center">
                <div className="text-[8px] text-amber-400 transform -skew-x-12 text-center">
                  <div>vinyl</div>
                  <div className="text-[6px] text-zinc-500">33⅓ rpm</div>
                </div>
              </div>
            )}
          </div>

          {/* Back side - Emotion data */}
          <div
            className="absolute w-full h-full backface-hidden rotate-y-180 bg-zinc-800 p-4 flex flex-col justify-center items-center"
            style={{ backfaceVisibility: "hidden" }}
          >
            <div className="text-center mb-4">
              <h3 className="font-medium text-white mb-1">{track.title}</h3>
              <p className="text-zinc-400 text-sm transform -skew-x-12">{track.artist}</p>
            </div>

            <div className="w-full max-w-[200px] mx-auto">
              <EmotionVisualizer arousal={track.emotion.arousal} valence={track.emotion.valence} large={true} />
            </div>

            <div className="mt-4 text-center">
              <Badge variant="custom" className={`${getGenreColor(track.genre)} transform -skew-x-12`}>{track.genre}</Badge>
            </div>
          </div>
        </div>

        {/* Control buttons */}
        <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="flex gap-3">
            <Button
              onClick={(e) => {
                e.stopPropagation()
                handleCardClick()
              }}
              variant="secondary"
              size="icon"
              className="rounded-full h-12 w-12 bg-white/20 hover:bg-white/30 backdrop-blur-sm"
            >
              {isCurrentlyPlaying && !isFlipped ? <Pause size={20} /> : <Play size={20} />}
            </Button>

            <Button
              onClick={(e) => {
                e.stopPropagation()
                setIsFlipped(!isFlipped)
              }}
              variant="secondary"
              size="icon"
              className="rounded-full h-12 w-12 bg-white/20 hover:bg-white/30 backdrop-blur-sm"
            >
              {isFlipped ? <Disc size={20} /> : <Info size={20} />}
            </Button>
          </div>
        </div>
      </div>

      <div className="p-3 flex justify-between items-center">
        <div ref={containerRef} className="min-w-0 flex-1 mr-2 overflow-hidden">
          <div
            ref={titleRef}
            className="font-medium text-sm text-white whitespace-nowrap"
            style={{
              transform: isCurrentlyPlaying && needsScroll ? `translateX(${titlePosition}px)` : "none",
              transition: !isCurrentlyPlaying ? "transform 0.5s ease-out" : "none",
            }}
          >
            {track.title}
          </div>
          <p className="text-zinc-400 text-xs transform -skew-x-12 truncate">{track.artist}</p>
        </div>

        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge variant="custom" className={`${getGenreColor(track.genre)} text-xs transform -skew-x-12`}>{track.genre}</Badge>

          <Button
            onClick={handleDownload}
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 text-zinc-300 hover:text-white hover:bg-zinc-800"
          >
            <Download className="h-3 w-3" />
          </Button>
        </div>
      </div>
    </Card>
  )
}
