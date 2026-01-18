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
import addBlocksToMessages, { addBlockToMessages } from '@/utils/chat/addBlockToMessages';
import BlockRenderer from './BlockRenderer';
import { useVideos } from '@/api/services/hooks/query';
import SendButton from './SendButton';
import AppBar from '../Appbar';
import Markdown from 'react-markdown';
import { ChatBubbleOvalLeftEllipsisIcon } from '@heroicons/react/24/solid';
import Thinking from './Chat/Thinking';
import Chip from '../common/components/Chip';

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
  const isOverrideMode = useStoreChat((state) => state.isOverrideMode);
  const overrideVideos = useStoreChat((state) => state.overrideVideos);
  const setOverrideVideos = useStoreChat((state) => state.setOverrideVideos);
  //   const [chatMessages, setChatMessages] = useState([{
  //     role: 'assistant',
  //     timestamp: Date.now(),
  //     blocks: [
  //       {
  //         block_type: 'text',
  //         text: 'Hello! How can I assist you today?',
  //       }
  //     ],
  //   },
  // {
  //     role: 'user',
  //     timestamp: Date.now(),
  //     blocks: [
  //       {
  //         block_type: 'text',
  //         text: 'Hi! I have a question about my order.',
  //       }
  //     ],
  // },
  // {
  //   role: 'assistant',
  //   timestamp: Date.now(),
  //   blocks: [
  //     {
  //       block_type: 'text',
  //       text: 'Sure! Could you please provide me with your order number so I can look into it for you?',
  //     },
  //     {
  //       block_type: 'image',
  //       url: ['http://100.120.22.90:5173/images/testImage.png', 'http://100.120.22.90:5173/images/testImage.png'],
  //     },
  //     {
  //       block_type: 'video',
  //       url: ['http://100.120.22.90:5173/videos/testVideo.mp4'],
  //     }
  //   ],
  // }]);
  const addChatMessage = useStoreChat((state) => state.addChatMessage);

  const getSessionId = useStoreChat((state) => state.getSessionId);
  const session_id = useStoreChat((state) => state.session_id);
  const setSessionId = useStoreChat((state) => state.setSessionId);
  const user = useStore((state) => state.user);

  const userId = user?.id;

  const [agentProgess, setAgentProgress] = useState(false);
  const [thinkingMessage, setThinkingMessage] = useState('Thinking');


  const queryClient = useQueryClient();

  const groupId = useStoreChat((state) => state.currentGroup);
  const { data: videos = [] } = useVideos(groupId, session_id);
  const selectedVideosIds = videos.filter(video => video.selected).map(video => video._id);
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
    refetchOnWindowFocus: false,
    refetchOnMount: false,
    staleTime: Infinity,
  })

  useEffect(() => {
    const handleStatus = (msg) => {
      if (!getSessionId()) setSessionId(msg.session_id);
      queryClient.invalidateQueries(['chatHistory']);
    };

    const handleThinking = (msg) => {
      setAgentProgress(false);
      setThinkingMessage(msg.content);
      scrollToBottomIfNeeded();
    };


    const handleResponse = (msg) => {
      setThinkingMessage('');
      const prev = useStoreChat.getState().chatMessages;

      const newBlock = parseChunkToBlock("text", msg.content_delta);
      if (!newBlock) return;

      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };
    const handleRunning = () => {
      setAgentProgress(true);
    }

    const handleMedia = (media) => {
      // done
      console.log("media received", media);
      setAgentProgress(false);
      const prev = useStoreChat.getState().chatMessages;
      if (!media || !media.media_type) return;
      const media_type = media.media_type;
      if (media_type !== 'image' && media_type !== 'video') return;

      const newBlocks = parseChunkToBlock(media_type, media.results)
      if (!newBlocks) return;

      // Fix: actually update state with the result
      const updated = addBlocksToMessages(prev, 'assistant', newBlocks);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };

    const handleFullResponse = (data) => {
      console.log("full_response", data);
    }

    // handle session status
    socket.on('message_received', handleStatus);

    // handle agent progress event
    socket.on("running", handleRunning);


    // handle stream thinking
    socket.on('thinking', handleThinking);

    // handle answer
    socket.on('response', handleResponse);

    // check save on database
    socket.on("full_response", handleFullResponse);

    // handle end
    socket.on('media', handleMedia);


    return () => {
      socket.off('message_received', handleStatus);
      socket.off('running', handleRunning);
      socket.off('thinking', handleThinking);
      socket.off('response', handleResponse);
      socket.off('media', handleMedia);
      socket.off('full_response', handleFullResponse);
    };
  }); // [] <- no deps, always up to date for dev

  const bottomRef = useRef(null);
  const chatRef = useRef(null);
  const chatContainerRef = useRef(null);
  const isNearBottomRef = useRef(true); // Track if user is near bottom

  // Helper to check if scrolled near bottom
  const checkIfNearBottom = () => {
    const container = chatContainerRef.current;
    if (!container) return true;
    const threshold = 100; // pixels from bottom to consider "at bottom"
    return container.scrollHeight - container.scrollTop - container.clientHeight < threshold;
  };

  // Auto-scroll only if user is near bottom
  const scrollToBottomIfNeeded = (behavior = 'smooth') => {
    if (isNearBottomRef.current) {
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior });
      });
    }
  };

  useEffect(() => {
    // if keyboard a-z or 0-9 is pressed, focus the chat input
    const handleKeyDown = (e) => {
      // Don't steal focus if modifier keys are pressed (Ctrl+C, Cmd+V, etc.)
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      // Don't steal focus if user is already typing in an input/textarea
      const activeEl = document.activeElement;
      const isTyping = activeEl?.tagName === 'INPUT' ||
        activeEl?.tagName === 'TEXTAREA' ||
        activeEl?.isContentEditable;
      if (isTyping) return;

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
          text: prompt,
        }
      ],
    });
    // Always scroll to bottom when user sends a message
    isNearBottomRef.current = true;
    scrollToBottomIfNeeded();
    reset({ prompt: '' });
    // Reset textarea height after clearing content
    requestAnimationFrame(() => {
      if (chatRef.current) {
        chatRef.current.style.height = 'auto';
      }
    });
  };

  return (
    <div className='h-screen w-full flex flex-col justify-between'>
      <AppBar />
      <div
        ref={chatContainerRef}
        onScroll={() => { isNearBottomRef.current = checkIfNearBottom(); }}
        className="flex flex-col w-full  h-[90vh] gap-12 px-4 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-300 overflow-y-auto"
      >
        {/*  hiện tại chưa handle block, hard code! */}
        {chatMessages.map((m, i) => (
          <div key={i} className='w-full flex flex-col'>
            {m.blocks.map((block, j) => (
              <BlockRenderer key={`${i}-${j}`} block={block} role={m.role} />
            ))}
          </div>
        ))}

        {/* test video block */}
        {
          <BlockRenderer block={{
            block_type: 'video',
            video_id: '2421946379',
            url: '/videos/testVideo.mp4',
            segments: [{ start_frame: 0, end_frame: 150 },
            { start_frame: 1000, end_frame: 2537 }], // in frames
            fps: 30,
          }} role={"assistant"} />
        }
        {/* test video block */}
        {
          <BlockRenderer block={{
            block_type: 'text',
            text: 'This is a test message to demonstrate the text block rendering in the chat interface. It should properly display the text content sent by the assistant role.',
          }} role={"assistant"} />
        }
        {/* test ảnh block */}
        {
          <BlockRenderer block={{
            block_type: 'image',
            url: ['/images/testImage.png', '/images/testImage.png', '/images/testImage.png'],
          }} role={"assistant"} />
        }


        {/* {thinkingMessage && <div className='flex pt-12 gap-2'>
          <div><ChatBubbleOvalLeftEllipsisIcon className="w-5 h-5 text-gray-400" /></div>
          <div className='animate-pulse text-white flex flex-col '>
            <Markdown>
              {thinkingMessage}
            </Markdown>
          </div>
        </div>} */}
        <Thinking />
        {agentProgess && <div className='animate-pulse text-white flex flex-col pt-12'>...</div>}
        <div ref={bottomRef}></div>
      </div>

      <div className="flex flex-row w-full px-4 py-2 space-x-2 z-10">
        <div className="flex-grow">
          <div className={clsx(
            'flex flex-col w-full rounded-lg bg-white/5',
            'focus-within:ring-2 focus-within:ring-white/20 transition-all'
          )}>
            {/* Chips inside the input container */}
            {isOverrideMode() && overrideVideos.length > 0 && (
              <div className='flex flex-wrap gap-2 px-3 pt-2'>
                {overrideVideos.map((video, index) => (
                  <Chip
                    key={index}
                    label={video.title}
                    size="sm"
                    onDelete={() => {
                      setOverrideVideos(overrideVideos.filter((v) => v.video_id !== video.video_id));
                    }}
                  />
                ))}
              </div>
            )}

            <Textarea
              {...register('prompt')}
              ref={(e) => {
                register('prompt').ref(e);
                chatRef.current = e;
              }}
              rows={1}
              className={clsx(
                'block w-full border-none bg-transparent px-3 py-1.5 text-sm/6 text-white',
                'focus:outline-none resize-none',
                'whitespace-pre-wrap leading-relaxed',
                'max-h-[10rem] overflow-y-auto'
              )}
              onInput={(e) => {
                e.target.style.height = 'auto';
                e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
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
          </div>
        </div>

        <SendButton control={control} handlePrompt={handlePrompt} />
      </div>
    </div>
  )
}
