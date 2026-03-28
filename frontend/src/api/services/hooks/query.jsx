import api from "@/api/api";
import { useStore } from "@/stores/chat";
import { useMutation, useQuery, useQueryClient } from "react-query";

export function useVideos(groupId, sessionId) {
    return useQuery({
        queryKey: ["videos", groupId, sessionId],
        queryFn: async () => {
            const res = await api.get(`/api/user/videos`, {
                params: {
                    session_id: sessionId,
                    group: groupId,
                },
            });
            return res.data.videos;
        },
        enabled: !!groupId && !!sessionId, // avoids invalid calls
    });
}

export function useRenameVideo() {
    const queryClient = useQueryClient();
    return useMutation({
        mutationFn: ({ videoId, newName }) =>
            api.patch(`/api/user/video/${videoId}/rename`, {
                new_name: newName,
            }),

        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['videos'] });
        },
    });
}

export const useCreateNewChat = () => {
    const queryClient = useQueryClient();
    const setSessionId = useStore((state) => state.setSessionId);
    return useMutation(
        async () => {
            const response = await api.post('/api/user/new-chat')
            return response.data;
        },
        {
            onSuccess: (data) => {
                const newChatId = data.chat_session_id;
                setSessionId(newChatId);
            },
            onSettled: () => {
                // Refetch chat history
                queryClient.invalidateQueries(['chatHistory']);
            }
        }
    )
}
export const useDeleteSession = () => {
    const queryClient = useQueryClient();
    const session_id = useStore((state) => state.session_id);
    const setSessionId = useStore((state) => state.setSessionId);
    const chatHistory = useStore((state) => state.chatHistory);

    const createNewChat = useCreateNewChat(); // 👈 composed here

    return useMutation({
        mutationFn: (session) =>
            api.delete(`/api/user/session/${session._id}/delete`),

        onSuccess: async (data) => {
            const deletedId = data.data.session_id;

            if (session_id === deletedId) {
                const remaining = chatHistory.filter(
                    (chat) => chat._id !== deletedId
                );

                if (remaining.length === 0) {
                    // ✅ never allow 0 chats
                    await createNewChat.mutateAsync();
                    return;
                }

                const index = chatHistory.findIndex(
                    (chat) => chat._id === deletedId
                );

                const next =
                    chatHistory[index + 1] ||
                    chatHistory[index - 1] ||
                    null;

                setSessionId(next?._id ?? null);
            }
        },

        onSettled: () => {
            queryClient.invalidateQueries(['chatHistory']);
        },
    });
};


export const useGroups = () => {
    const currentGroup = useStore((state) => state.currentGroup);
    const setCurrentGroup = useStore((state) => state.setCurrentGroup);
    return useQuery({
        queryKey: ['groups'],
        queryFn: async () => {
            const res = await api.get('/api/user/groups');
            return res.data.groups
        },
        onSuccess: (data) => {
            if (data.length > 0) {
                if (!currentGroup) {
                    setCurrentGroup(data[0]._id);
                }
            }
        }
    });
}

export const useCreateGroup = () => {
    const queryClient = useQueryClient();
    const setCurrentGroup = useStore((state) => state.setCurrentGroup);
    return useMutation({
        mutationFn: (groupName) => {
            return api.post('/api/user/groups/create', { group_name: groupName })
        },
        onSuccess: (data) => {
            const new_group_id = data.data.group_id;
            setCurrentGroup(new_group_id);
        },
        onSettled: () => {
            queryClient.invalidateQueries('groups');
        }
    })
}

export const useDeleteGroup = () => {
    const queryClient = useQueryClient();
    const setCurrentGroup = useStore((state) => state.setCurrentGroup);
    const currentGroup = useStore((state) => state.currentGroup);
    const createGroup = useCreateGroup();
    return useMutation({
        mutationFn: (groupId) => {
            return api.delete(`/api/user/groups/${groupId}/delete`)
        },
        onSuccess: async (data) => {
            const groups = queryClient.getQueryData('groups') || [];
            if (groups.length <= 1) {
                // create a new group if no groups left
                await createGroup.mutateAsync();
                return;
            }
            if (currentGroup === data.data.group_id) {
                const index = groups.findIndex(chat => chat._id === data.data.group_id);
                const next = groups[index + 1] || groups[index - 1] || null;
                setCurrentGroup(next ? next._id : null);
            }
        },
        onSettled: () => {
            queryClient.invalidateQueries('groups');
        }
    })
}

export const useRenameGroup = () => {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: ({ groupId, newName }) =>
            api.patch(`/api/user/group/${groupId}/rename`, {
                new_name: newName,
            }),

        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ['groups'] });
        },
    });
};

export const useSearchChatHistory = (searchTerm) => {
    return useQuery({
        queryKey: ['searchChatHistory', searchTerm],
        queryFn: async () => {
            const res = await api.get('/api/user/chat-history/search', {
                params: { query_text: searchTerm },
            });
            return res.data.results;
        },
        enabled: !!searchTerm,
    });
}