import React, { useEffect, useState } from 'react'
import { messagesConversations } from '@/mockdata/messages'
import { useMutation, useQuery, useQueryClient } from 'react-query';
import api from '@/api/api';
import { useStore } from '@/stores/user';
import { useStore as useStoreChat } from "@/stores/chat";

import clsx from 'clsx';
import SessionDropdownList from './SessionDropdownList';
import { PlusIcon } from '@heroicons/react/16/solid';
import { useCreateNewChat } from '@/api/services/hooks/query';
import ChatHistory from './ChatHistory';

export default function HistoryConversations() {
  const chatHistory = useStoreChat((state) => state.chatHistory);
  const setChatHistory = useStoreChat((state) => state.setChatHistory);
  const session_id = useStoreChat((state) => state.session_id);
  const setSessionId = useStoreChat((state) => state.setSessionId);
  const user = useStore((state) => state.user);

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
    }
  );

  const createNewChatMutation = useCreateNewChat();

  function createNewChat() {
    createNewChatMutation.mutate();
  }


  function handleEditChat(chatId, newName) {
    // Update local state optimistically
    // setChatHistory(prev => 
    //   prev.map(chat => 
    //     chat._id === chatId ? { ...chat, name: newName } : chat
    //   )
    // );
    
    // TODO: Call API to persist the change
    // api.patch(`/api/chat/${chatId}`, { name: newName });
  }

  const ensureSessionId = useStoreChat((state) => state.ensureSessionId);
  useEffect(() => {
    if (!ensureSessionId()) {
      createNewChat();
    }
  }, [session_id]);

  return (
    <div className='relative flex flex-col h-full '>
      <div className='border-b border-gray-800 flex items-center justify-between sticky top-0 bg-black/90 px-2 py-1 '>
        <p className='text-sm p-2 text-gray-400/60'>Chats</p>
        <PlusIcon className="btn-icon" onClick={createNewChat} />
      </div>
      <div className='flex flex-col scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-300 overflow-y-auto'>
        {
          chatHistory.map((conv, idx) => (
            <ChatHistory
              key={conv._id || idx}
              conv={conv}
              session_id={session_id}
              onEdit={handleEditChat}
            />
          ))
        }
      </div>
    </div>
  )
}
