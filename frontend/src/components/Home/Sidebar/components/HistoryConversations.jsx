import React, { useEffect, useState } from 'react'
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
            setSessionId(data[0]._id);
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
    const newChatHistory = chatHistory.map(chat =>
      chat._id === chatId ? { ...chat, name: newName } : chat
    );
    setChatHistory(newChatHistory);
    // TODO: Call API to persist the change
    api.patch(`/api/user/session/${chatId}/rename`, { new_name: newName });
  }


  return (
    <div className='relative flex flex-col h-full'>
      {/* New Chat Button */}
      <div className='sticky flex top-0 px-2 py-2 border-b border-surface-light h-14'>
        <button
          onClick={createNewChat}
          className='flex items-center gap-2 self-center w-full px-3 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium transition-colors cursor-pointer'
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
    </div>
  )
}
