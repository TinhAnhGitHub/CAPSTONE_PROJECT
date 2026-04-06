import { create } from "zustand";
import { persist } from "zustand/middleware";

export const useStore = create(
    persist(
        (set, get) => ({
            user: null,
            token: null,

            // set both token + usera
            login: (user, token) => set({ user, token }),

            // update user only
            setUser: (user) => set({ user }),

            // clear everything
            logout: () => set({ user: null, token: null, session_id: null, chatMessages: [], chatHistory: [], workspaceVideos: [] }),
        }),
        {
            name: "appState", // key in localStorage
        }
    )
);
