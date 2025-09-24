"use client"

import { useRef, useEffect, MouseEvent } from "react"

interface EmotionVisualizerProps {
  arousal: number
  valence: number
  onChange?: (arousal: number, valence: number) => void
  mini?: boolean
  expanded?: boolean
  onExpand?: () => void
}

const EMOTION_LABELS = [
  { text: "Excited", x: 0.8, y: 0.2 },
  { text: "Happy", x: 0.9, y: 0.3 },
  { text: "Calm", x: 0.6, y: 0.7 },
  { text: "Sleepy", x: 0.3, y: 0.9 },
  { text: "Sad", x: 0.2, y: 0.8 },
  { text: "Angry", x: 0.1, y: 0.2 }
]

export function EmotionVisualizer({ 
  arousal, 
  valence, 
  onChange, 
  mini = false,
  expanded = false,
  onExpand
}: EmotionVisualizerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const isDraggingRef = useRef(false)

  // Calculate position in the 2D grid
  const xPos = `${valence * 100}%`
  const yPos = `${(1 - arousal) * 100}%` // Invert Y-axis for visual representation

  const updatePosition = (e: MouseEvent) => {
    if (!containerRef.current || !onChange) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width))
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height))
    onChange(1 - y, x) // Invert Y-axis back for arousal
  }

  const handleMouseDown = (e: MouseEvent) => {
    if (onChange) {
      isDraggingRef.current = true
      updatePosition(e)
    }
  }

  const handleMouseMove = (e: MouseEvent) => {
    if (isDraggingRef.current) {
      updatePosition(e)
    }
  }

  const handleMouseUp = () => {
    isDraggingRef.current = false
  }

  useEffect(() => {
    document.addEventListener('mouseup', handleMouseUp)
    return () => document.removeEventListener('mouseup', handleMouseUp)
  }, [])

  if (mini && !expanded) {
    return (
      <div 
        className="w-10 h-10 bg-zinc-900 rounded-md overflow-hidden cursor-pointer hover:scale-105 transition-all"
        onClick={onExpand}
      >
        <div className="relative w-full h-full">
          <div className="absolute top-1/2 left-0 right-0 h-px bg-zinc-700"></div>
          <div className="absolute top-0 bottom-0 left-1/2 w-px bg-zinc-700"></div>
          <div
            className="absolute h-2 w-2 bg-amber-400 rounded-full"
            style={{ left: xPos, top: yPos }}
          ></div>
        </div>
      </div>
    )
  }

  return (
    <div 
      ref={containerRef}
      className="relative h-48 bg-zinc-900 rounded-md overflow-hidden"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      style={{ cursor: onChange ? 'crosshair' : 'default' }}
    >
      {/* Grid lines */}
      <div className="absolute top-1/2 left-0 right-0 h-px bg-zinc-700"></div>
      <div className="absolute top-0 bottom-0 left-1/2 w-px bg-zinc-700"></div>

      {/* Emotion labels */}
      {EMOTION_LABELS.map(({ text, x, y }) => (
        <div
          key={text}
          className="absolute text-xs text-zinc-500 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none"
          style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
        >
          {text}
        </div>
      ))}

      {/* Emotion marker */}
      <div
        className="absolute h-3 w-3 bg-amber-400 rounded-full"
        style={{ 
          left: xPos, 
          top: yPos,
          cursor: onChange ? 'grab' : 'default'
        }}
      ></div>

      {/* Values */}
      <div className="absolute bottom-2 right-2 text-xs text-zinc-400">
        <span className="transform -skew-x-12">v: {valence.toFixed(2)} a: {arousal.toFixed(2)}</span>
      </div>
    </div>
  )
}
