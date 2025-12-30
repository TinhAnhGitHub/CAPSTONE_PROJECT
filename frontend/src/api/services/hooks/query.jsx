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
    return useMutation({
        mutationFn: (session) => {
            // if current session is being deleted, create new chat after deletion
            return api.delete(`/api/user/session/${session._id}/delete`);
        },
        onSuccess: (data) => {
            if (session_id === data.data.session_id) {
                setSessionId(null);
            }
        },
        onSettled: () => {
            queryClient.invalidateQueries('chatHistory');
        }
    })
}

export const useGroups = () => {
    return useQuery({
        queryKey: ['groups'],
        queryFn: async () => {
            const res = await api.get('/api/user/groups');
            // console.log(res.data)
            return res.data.groups
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
    return useMutation({
        mutationFn: (groupId) => {
            return api.delete(`/api/user/groups/${groupId}/delete`)
        },
        onSuccess: (data) => {
            if (currentGroup === data.data.group_id) {
                setCurrentGroup(null);
            }
        },
        onSettled: () => {
            queryClient.invalidateQueries('groups');
        }
    })
}