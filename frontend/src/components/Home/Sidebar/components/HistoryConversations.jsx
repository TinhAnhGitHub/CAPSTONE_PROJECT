import React, { useEffect, useState } from 'react'
import { messagesConversations } from '@/mockdata/messages'
import { useMutation, useQuery, useQueryClient } from 'react-query';
import api from '@/api/api';
import { useStore } from '@/stores/user';
import { useStore as useStoreChat } from "@/stores/chat";

import clsx from 'clsx';
import SessionDropdownList from './SessionDropdownList';
import { PencilSquareIcon } from '@heroicons/react/20/solid';
import { useCreateNewChat } from '@/api/services/hooks/query';
import ChatHistory from './ChatHistory';

export default function HistoryConversations() {
  const chatHistory = useStoreChat((state) => state.chatHistory);

  // Mock data for testing - remove or set to [] in production
  const mockChatHistory = [
    { _id: '1', session_id: 'sess_1', name: 'How to use React hooks', created_at: new Date() },
    { _id: '2', session_id: 'sess_2', name: 'Video editing tips', created_at: new Date() },
    { _id: '3', session_id: 'sess_3', name: 'AI image generation', created_at: new Date() },
    { _id: '4', session_id: 'sess_4', name: 'Building a REST API', created_at: new Date() },
    { _id: '5', session_id: 'sess_5', name: 'CSS Grid layout tutorial', created_at: new Date() },
  ];

  // Use mock if chatHistory is empty
  const displayHistory = chatHistory.length > 0 ? chatHistory : mockChatHistory;

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
    <div className='relative flex flex-col h-full'>
      {/* New Chat Button */}
      <div className='sticky top-0 px-2 py-2 border-b border-surface-light'>
        <button
          onClick={createNewChat}
          className='flex itemeaes-center gap-2 w-full px-3 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium transition-colors cursor-pointer'
        >
          <PencilSquareIcon className="w-5 h-5" />
          <span>New Chat</span>
        </button>
      </div>

      {/* Your Chats Section */}
      <div className='flex flex-col flex-1 overflow-hidden'>
        <p className='text-xs text-text-muted uppercase tracking-wide px-4 py-2'>Your Chats</p>
        <div className='flex flex-col flex-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-auto'>
          {
            displayHistory.map((conv, idx) => (
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
    </div>
  )
}
