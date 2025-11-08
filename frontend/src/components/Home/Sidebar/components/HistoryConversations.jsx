import React, { useState } from 'react'
import { messagesConversations } from '@/mockdata/messages'
import { useMutation, useQuery, useQueryClient } from 'react-query';
import api from '@/api/api';
import { useStore } from '@/stores/user';
import { useStore as useStoreChat } from "@/stores/chat";

import clsx from 'clsx';
import SessionDropdownList from './SessionDropdownList';
import { PlusIcon } from '@heroicons/react/16/solid';

export default function HistoryConversations() {
  const chatHistory = useStoreChat((state) => state.chatHistory);
  const setChatHistory = useStoreChat((state) => state.setChatHistory);
  const setSessionId = useStoreChat((state) => state.setSessionId);
  const session_id = useStoreChat((state) => state.session_id);
  const setChatMessages = useStoreChat((state) => state.setChatMessages);
  const setWorkspaceVideos = useStoreChat((state) => state.setWorkspaceVideos);
  const user = useStore((state) => state.user);
  const queryClient = useQueryClient();

  useQuery(
    {
      queryKey: ["chatHistory", user],
      queryFn: async () => {
        const response = await api.get('/api/user/chat-history');
        const chats = response.data.chats;
        return chats;
      },
      onSuccess: (data) => {
        setChatHistory(data);
        if (data.length > 0) {
          if (!session_id) {
            setSessionId(data[0].session_id);
          }
        }
      },
      enabled: !!user,
    }
  );

  const createNewChatMutation = useMutation(
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
  function createNewChat() {
    createNewChatMutation.mutate();
  }

  function selectConversation(session_id) {
    // set session id 
    setSessionId(session_id);
  }

  return (
    <div className='relative flex flex-col h-full '>
      <div className='border-b border-gray-800 flex items-center justify-between sticky top-0 bg-black/90 p-2 '>
        <p className='text-sm p-2 text-gray-400/60'>Chats</p>
        <PlusIcon className="btn-icon" onClick={createNewChat} />
      </div>
      <div className='flex flex-col scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-300 overflow-y-auto'>
        {
          chatHistory.map((conv, idx) => (
            <div key={idx}
              className={clsx('relative m-1 py-2 px-4 hover:bg-gray-800 cursor-pointer rounded-lg',
                (session_id === conv._id) ? 'bg-gray-800' : '',
                "group"
              )}
              onClick={() => selectConversation(conv._id)}>
              <div className='text-sm '>{conv?._id?.slice(0, 8)}</div>
              <div className='absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1 hover:bg-gray-600 cursor-pointer hidden group-hover:block has-data-open:block'>
                <SessionDropdownList session={conv} />
              </div>
            </div>
          ))
        }
      </div>
    </div>
  )
}
