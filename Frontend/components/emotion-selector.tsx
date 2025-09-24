"use client"

import { useState, useCallback } from "react"
import { EmotionVisualizer } from "./emotion-visualizer"
import { Dialog, DialogContent } from "@/components/ui/dialog"

interface EmotionSelectorProps {
  onEmotionChange: (emotion: { arousal: number; valence: number }) => void
}

export function EmotionSelector({ onEmotionChange }: EmotionSelectorProps) {
  const [selectedEmotion, setSelectedEmotion] = useState<{ arousal: number; valence: number }>({
    arousal: 0.5,
    valence: 0.5
  })
  const [isExpanded, setIsExpanded] = useState(false)

  const handleEmotionChange = useCallback((arousal: number, valence: number) => {
    setSelectedEmotion({ arousal, valence })
    onEmotionChange({ arousal, valence })
  }, [onEmotionChange])

  return (
    <div className="inline-block">
      <EmotionVisualizer 
        arousal={selectedEmotion.arousal} 
        valence={selectedEmotion.valence}
        onChange={handleEmotionChange}
        mini={true}
        expanded={isExpanded}
        onExpand={() => setIsExpanded(true)}
      />
      
      <Dialog open={isExpanded} onOpenChange={setIsExpanded}>
        <DialogContent className="sm:max-w-lg">
          <EmotionVisualizer 
            arousal={selectedEmotion.arousal} 
            valence={selectedEmotion.valence}
            onChange={handleEmotionChange}
          />
        </DialogContent>
      </Dialog>
    </div>
  )
}