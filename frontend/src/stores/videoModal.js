// stores/videoModal.js
import { create } from 'zustand'

export const useVideoModalStore = create((set) => ({
    video: null,
    isOpen: false,

    open: (video) => set({ video, isOpen: true }),
    close: () => set({ video: null, isOpen: false }),
}))