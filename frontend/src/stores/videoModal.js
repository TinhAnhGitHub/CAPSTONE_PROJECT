// stores/videoModal.js
import { create } from 'zustand'

export const useVideoModalStore = create((set) => ({
    video: null,
    isOpen: false,
    startTime: null, // seconds to seek to on open (optional)

    open: (video, startTime = null) => set({ video, isOpen: true, startTime }),
    close: () => set({ video: null, isOpen: false, startTime: null }),
}))