import React, { useState, useEffect, useRef, useLayoutEffect } from 'react'
import socket from '@/api/socket';
import { Textarea } from '@headlessui/react';
import clsx from 'clsx';
import { useStore } from '@/stores/user';
import { useStore as useStoreChat } from "@/stores/chat";

import { useQuery, useQueryClient } from 'react-query';
import api from '@/api/api';
import { useForm } from 'react-hook-form';
import parseChunkToBlock from '@/utils/chat/parseChunkToBlock';
import addBlockToMessages from '@/utils/chat/addBlockToMessages';
import BlockRenderer from './BlockRenderer';
import { useVideos } from '@/api/services/hooks/query';
import SendButton from './SendButton';

export default function Chat() {
  const {
    register,
    handleSubmit,
    getValues,
    reset,
    control,
  } = useForm()

  const chatMessages = useStoreChat((state) => state.chatMessages);
  const setChatMessages = useStoreChat((state) => state.setChatMessages);
  const addChatMessage = useStoreChat((state) => state.addChatMessage);

  const getSessionId = useStoreChat((state) => state.getSessionId);
  const session_id = useStoreChat((state) => state.session_id);
  const setSessionId = useStoreChat((state) => state.setSessionId);
  const user = useStore((state) => state.user);

  const userId = user?.id;

  const [thinkingMessage, setThinkingMessage] = useState('');

  const queryClient = useQueryClient();

  const groupId = useStoreChat((state) => state.currentGroup);
  const { data: videos = [] } = useVideos(groupId, session_id);
  const selectedVideosIds = videos.filter(video => video.selected).map(video => video.id);
  useQuery({
    queryKey: ["chatMessages", session_id],
    queryFn: async () => {
      const session_id = getSessionId();
      if (!session_id) return [];
      const response = await api.get(`/api/user/chat-history/${session_id}`);
      const chat = response.data.chat;
      return chat;
    },
    onSuccess: (data) => {
      setChatMessages(data);
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'auto' });
      });
    },
    enabled: !!session_id,
  })

  useEffect(() => {
    const handleStatus = (msg) => {
      if (!getSessionId()) setSessionId(msg.session_id);
      queryClient.invalidateQueries(['chatHistory']);
    };

    const handleStreamThinking = (msg) => {
      setThinkingMessage(msg.status);
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      });
    };


    const handleAnswer = (msg) => {
      console.log("stream_chunk", msg);
      setThinkingMessage('');
      const prev = useStoreChat.getState().chatMessages;
      
      const newBlock = parseChunkToBlock(msg.msg_type, msg.chunk)
      if (!newBlock) return;

      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
      });
    };

    const handleAnswerEnd = () => {
      // done
    };

    // handle session status
    socket.on('message_received', handleStatus);

    // handle stream thinking
    socket.on('stream_thinking', handleStreamThinking);

    // handle answer
    socket.on('stream_chunk', handleAnswer);

    // handle end
    socket.on('stream_end', handleAnswerEnd);


    return () => {
      socket.off('message_received', handleStatus);
      socket.off('stream_thinking', handleStreamThinking);
      socket.off('stream_chunk', handleAnswer);
      socket.off('stream_end', handleAnswerEnd);
    };
  }); // [] <- no deps, always up to date for dev

  const bottomRef = useRef(null);
  const chatRef = useRef(null);

  useEffect(() => {
    // if keyboard a-z or 0-9 is pressed, focus the chat input
    const handleKeyDown = (e) => {
      const key = e.key || e.keyCode;
      if ((key.length === 1 && key.match(/[a-z0-9]/i)) || (key >= 48 && key <= 90)) {
        chatRef.current?.focus();
      }
    }
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handlePrompt = async () => {
    const prompt = getValues('prompt').trim();
    if (!prompt) return;

    socket.emit('stream_chat', { userId, sessionId: getSessionId(), text: prompt, videos: selectedVideosIds });
    addChatMessage({
      role: 'user',
      timestamp: Date.now(),
      blocks: [
        {
          block_type: 'text',
          text_content: prompt,
        }
      ],
    });
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    })
    reset({ prompt: '' });
  };

  return (
    <div className='h-screen w-full flex flex-col justify-between'>
      <div className="flex flex-col w-full  h-[90vh] gap-12 px-4 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-300 overflow-y-auto">
        {/*  hiện tại chưa handle block, hard code! */}
        {chatMessages.map((m, i) => (
          <div key={i} className='w-full flex flex-col'>
            {m.blocks.map((block, j) => (
              <BlockRenderer key={j} block={block} role={m.role} />
            ))}
          </div>
        ))}

        {thinkingMessage && <div className='animate-pulse text-white flex flex-col pt-12'>{thinkingMessage}</div>}
        <div ref={bottomRef}></div>
      </div>

      <div className="flex flex-row w-full px-4 py-2 space-x-2 z-10">
        <Textarea
          {...register('prompt')}
          ref={(e) => {
            register('prompt').ref(e);
            chatRef.current = e; // preserve manual ref usage
          }}
          rows={1}
          className={clsx(
            'block w-full rounded-lg border-none bg-white/5 px-3 py-1.5 text-sm/6 text-white',
            'focus:outline-none resize-none',
            'whitespace-pre-wrap leading-relaxed',
            'max-h-[10rem] overflow-y-auto' // ⬅️ cap at ~5 lines + scroll
          )}
          onInput={(e) => {
            e.target.style.height = 'auto'; // reset
            e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px'; // cap to ~10rem (5 lines)
          }}
          onKeyDown={(e) => {
            const value = getValues('prompt')?.trim() || '';
            if (e.key === 'Enter' && !e.shiftKey && value) {
              e.preventDefault();
              handlePrompt();
              e.target.style.height = 'auto';
            }
          }}
          placeholder="Ask the agent..."
        />

          <SendButton control={control} handlePrompt={handlePrompt} />
      </div>
    </div>
  )
}
