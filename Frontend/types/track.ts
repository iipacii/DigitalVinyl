export interface Track {
  id: string
  title: string
  artist: string
  audioUrl: string
  imageUrl: string
  emotion: {
    arousal: number // 0 to 1, energy/intensity
    valence: number // 0 to 1, positivity/negativity
  }
  genre: string
  uploadedAt: Date
}
