import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTime(seconds: number): string {
  if (isNaN(seconds)) return "0:00"

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.floor(seconds % 60)

  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`
}

export function getGenreColor(genre: string): string {
  const genres: Record<string, string> = {
    electronic: "bg-cyan-500 text-white",
    rock: "bg-red-500 text-white",
    pop: "bg-pink-500 text-white",
    "hip hop": "bg-purple-500 text-white",
    "hip-hop": "bg-purple-500 text-white", // Alternative format
    rap: "bg-purple-500 text-white", // Common alternative
    jazz: "bg-blue-500 text-white",
    classical: "bg-green-500 text-white",
    ambient: "bg-teal-500 text-white",
    folk: "bg-amber-500 text-white",
    blues: "bg-indigo-500 text-white",
    country: "bg-yellow-500 text-black",
    metal: "bg-zinc-800 text-white",
    reggae: "bg-green-600 text-white",
    soul: "bg-orange-500 text-white",
    rnb: "bg-orange-500 text-white",
    "r&b": "bg-orange-500 text-white",
  }

  // Case-insensitive lookup, trim any whitespace
  const normalizedGenre = genre.toLowerCase().trim()
  
  return genres[normalizedGenre] || "bg-zinc-600 text-white"
}
