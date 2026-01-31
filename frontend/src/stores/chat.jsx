import socket from "@/api/socket";
import { create } from "zustand";
import { persist } from "zustand/middleware";

export const useStore = create(
    persist(

        (set, get) => ({
            session_id: null, // new chat
            setSessionId: (session_id) => {
                socket.emit('join_session', { session_id });
                set({ session_id })
            },
            getSessionId: () => get().session_id,
            ensureSessionId: () => {
                let session_id = get().session_id;
                if (session_id) return session_id;
                // get the first one in chatHistory
                const chatHistory = get().chatHistory;
                if (chatHistory.length > 0) {
                    session_id = chatHistory[0]._id;
                    set({ session_id });
                    return session_id;
                }
                return null;
            },

            chatMessages: [],
            setChatMessages: (messages) => set({ chatMessages: messages }),
            addChatMessage: (message) => set({ chatMessages: [...get().chatMessages, message] }),
            clearChatMessages: () => set({ chatMessages: [] }),


            chatHistory: [],
            setChatHistory: (history) => set({ chatHistory: history }),
            addChatHistory: (chat) => set({ chatHistory: [...get().chatHistory, chat] }),
            clearChatHistory: () => set({ chatHistory: [] }),


            workspaceVideos: [],
            setWorkspaceVideos: (videos) => set({ workspaceVideos: videos }),
            clearWorkspaceVideos: () => set({ workspaceVideos: [] }),
            removeWorkspaceVideo: (videoId) => set({ workspaceVideos: get().workspaceVideos.filter(v => v.id !== videoId) }),
            addWorkspaceVideo: (video) => set({ workspaceVideos: [...get().workspaceVideos, video] }),

            overrideVideos: [],
            setOverrideVideos: (videos) => set({ overrideVideos: videos }),
            clearOverrideVideos: () => set({ overrideVideos: [] }),
            isOverrideMode: () => get().overrideVideos.length > 0,

            currentGroup: null,
            setCurrentGroup: (group) => set({ currentGroup: group }),

            sidebarOpen: false,
            setSidebarOpen: (open) => set({ sidebarOpen: open }),
            toggleSidebar: () => set({ sidebarOpen: !get().sidebarOpen }),
        }),
        {
            name: "chatState", // key in localStorage
            partialize: (state) => ({
                session_id: state.session_id,
                currentGroup: state.currentGroup,
            })
        }
    )
);
