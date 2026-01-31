import React, { useState, useEffect, useRef, useLayoutEffect } from 'react'
import socket from '@/api/socket';
import { Textarea } from '@headlessui/react';
import clsx from 'clsx';
import { useStore } from '@/stores/user';
import { useStore as useStoreChat } from "@/stores/chat";
import { ChevronDownIcon } from '@heroicons/react/24/solid';

import { useQuery, useQueryClient } from 'react-query';
import api from '@/api/api';
import { useForm } from 'react-hook-form';
import parseChunkToBlock from '@/utils/chat/parseChunkToBlock';
import addBlocksToMessages, { addBlockToMessages, updateToolCallBlock } from '@/utils/chat/addBlockToMessages';
import BlockRenderer from './BlockRenderer';
import { useVideos } from '@/api/services/hooks/query';
import SendButton from './SendButton';
import AppBar from '../Appbar';
import Markdown from 'react-markdown';
import { ChatBubbleOvalLeftEllipsisIcon } from '@heroicons/react/24/solid';
import { XMarkIcon } from '@heroicons/react/16/solid';

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
  const addChatMessage = useStoreChat((state) => state.addChatMessage);

  const getSessionId = useStoreChat((state) => state.getSessionId);
  const session_id = useStoreChat((state) => state.session_id);
  const setSessionId = useStoreChat((state) => state.setSessionId);
  const user = useStore((state) => state.user);

  const userId = user?.id;

  const queryClient = useQueryClient();

  const [showScrollDown, setShowScrollDown] = useState(false);

  const handleScroll = () => {
    const nearBottom = checkIfNearBottom();
    isNearBottomRef.current = nearBottom;
    setShowScrollDown(!nearBottom);
  };


  const groupId = useStoreChat((state) => state.currentGroup);
  const { data: videos = [] } = useVideos(groupId, session_id);
  const selectedVideosIds = videos.filter(video => video.selected).map(video => video._id);
  const [querying, setQuerying] = useState(false);
  useQuery({
    queryKey: ["chatMessages", session_id],
    queryFn: async () => {
      const session_id = getSessionId();
      if (!session_id) return [];
      const response = await api.get(`/api/user/chat-history/${session_id}`);
      const chat = response.data.chat;
      console.log("Fetched chat history:", chat);
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
    // staleTime: Infinity,
  })

  useEffect(() => {
    const handleStatus = (msg) => {
      setQuerying(true);
      if (!getSessionId()) setSessionId(msg.session_id);
      queryClient.invalidateQueries(['chatHistory']);
    };



    const handleResponse = (msg) => {
      const prev = useStoreChat.getState().chatMessages;

      const newBlock = parseChunkToBlock("text", msg.content_delta);
      if (!newBlock) return;

      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };

    const handleMedia = (media) => {
      const prev = useStoreChat.getState().chatMessages;
      if (!media || !media.media_type) return;
      const media_type = media.media_type;
      if (media_type !== 'image' && media_type !== 'video') return;

      const newBlocks = parseChunkToBlock(media_type, media.results)
      if (!newBlocks) return;

      const updated = addBlocksToMessages(prev, 'assistant', newBlocks);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };

    // handle session status
    socket.on('message_received', handleStatus);

    // handle stream thinking
    const handleThinking = (data) => {
      console.log("thinking ✅", data);
      const newBlock = parseChunkToBlock('thinking', data)
      console.log("newBlock thinking", newBlock);
      if (!newBlock) return;

      const prev = useStoreChat.getState().chatMessages;
      // Fix: actually update state with the result
      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };

    socket.on('thinking', handleThinking);
    // handle toolcall
    const handleToolCall = (data) => {
      const newBlock = parseChunkToBlock('tools', data)
      if (!newBlock) return;

      const prev = useStoreChat.getState().chatMessages;
      // Fix: actually update state with the result
      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
    }
    socket.on('tool_call', handleToolCall);
    //handle toolcallresult
    const handleToolCallResult = (data) => {
      // find the tool name
      const prev = useStoreChat.getState().chatMessages;
      const finished_tool_name = data.tool_name;
      const updated = updateToolCallBlock(prev, finished_tool_name);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
    }
    socket.on('tool_result', handleToolCallResult);
    // handle answer
    socket.on('response', handleResponse);

    // handle end
    socket.on('media', handleMedia);

    socket.on('stream_end', (msg) => {
      // console.log("stream end ✅✅✅✅✅✅✅✅✅", msg);
      setQuerying(false);
    })

    socket.on('continue_stream', (msg) => {
      const data = msg.content;
      const prev = useStoreChat.getState().chatMessages;
      const newBlocks = data.map((block) => parseChunkToBlock(block.block_type, block));
      const updated = addBlocksToMessages(prev, 'assistant', newBlocks);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
    })

    return () => {
      socket.off('message_received', handleStatus);
      socket.off('thinking', handleThinking);
      socket.off('response', handleResponse);
      socket.off('media', handleMedia);
      socket.off('tool_call', handleToolCall);
      socket.off('tool_result', handleToolCallResult);
      socket.off('stream_end');
    };
  }, []); // 

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

  const stopStreaming = () => {
    socket.emit('stop_stream', { sessionId: getSessionId() });
    setQuerying(false);
  };



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
    <div className='h-screen w-full flex flex-col justify-between bg-background'>
      <AppBar />
      <div
        ref={chatContainerRef}
        // onScroll={() => { isNearBottomRef.current = checkIfNearBottom(); }}
        onScroll={handleScroll}
        className="flex flex-col w-full h-[90vh] px-4 md:px-8 lg:px-16 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-scroll scrollbar-gutter-stable"
      >
        {/* Centered content container with max-width */}
        <div className="w-full max-w-3xl mx-auto flex flex-col gap-3 py-6">
          {chatMessages.map((m, i) => (
            <div key={i} className='w-full flex flex-col'>
              {m.blocks.map((block, j) => (
                <BlockRenderer key={`${i}-${j}`} block={block} role={m.role} />
              ))}
            </div>
          ))}

          {/* test video block */}
          {/* {
            <BlockRenderer block={{
              block_type: 'video',
              video_id: '2421946379',
              url: '/videos/testVideo.mp4',
              segments: [{ start_frame: 0, end_frame: 150 },
              { start_frame: 1000, end_frame: 2537 }], // in frames
              fps: 30,
            }} role={"assistant"} />
          } */}
          {/* test video block */}
          {/* {
            <BlockRenderer block={{
              block_type: 'text',
              text: 'This is a test message to demonstrate the text block rendering in the chat interface. It should properly display the text content sent by the assistant role.',
            }} role={"assistant"} />
          } */}
          {/* test ảnh block */}
          {/* {
            <BlockRenderer block={{
              block_type: 'image',
              url: ['/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png', '/images/testImage.png'],
            }} role={"assistant"} />
          } */}
          {/* test tool */}
          {/* {
            <BlockRenderer block={{
              block_type: 'tools',
              tools: [{
                tool_name: "Image Recognition",
                description: "Calling an image recognition model to analyze the provided image and extract relevant information."
              },
              {
                tool_name: "Voice Recognition",
                description: "Using speech-to-text to transcribe audio content from the video."
              },
              {
                tool_name: "OCR",
                description: "Extracting text from images using optical character recognition."
              },
              {
                tool_name: "CLIP Search",
                description: "Using CLIP model to find relevant video segments based on semantic image-text matching."
              }]
            }} role={"assistant"} />
          } */}
          {/* test thinking */}
          {/* {
            <BlockRenderer block={{
              block_type: 'thinking',
              thinking: "The assistant is currently processing the request and generating a response.",
            }} role={"assistant"} />
          } */}
          {showScrollDown && (
            <button
              onClick={() => {
                isNearBottomRef.current = true;
                bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
              }}
              className="
              fixed bottom-24 right-6 md:right-10
              z-20
              flex items-center justify-center
              w-10 h-10
              rounded-full
              bg-surface border border-surface-light
              shadow-lg
              cursor-pointer
              hover:bg-surface-light
              transition-all ease-out
            "
              aria-label="Scroll to bottom"
            >
              <ChevronDownIcon className="w-5 h-5 text-text" />
            </button>
          )}

          <div ref={bottomRef}></div>
        </div>
      </div>

      {/* Input area with centered max-width */}
      <div className="w-full px-4 md:px-8 lg:px-16 py-2 z-10 bg-background">
        <div className="max-w-3xl mx-auto">
          <div className={clsx(
            'flex flex-col w-full rounded-xl bg-surface border border-surface-light',
            'focus-within:ring-2 focus-within:ring-accent/50 focus-within:border-accent transition-all'
          )}>
            {/* Video thumbnails inside the input container */}
            {isOverrideMode() && overrideVideos.length > 0 && (
              <div className='flex flex-wrap gap-2 px-3 pt-3'>
                {overrideVideos.map((video, index) => (
                  <div
                    key={index}
                    className='relative group rounded-lg overflow-hidden border border-surface-light hover:border-accent/50 transition-colors'
                  >
                    <img
                      src={video.thumbnail || '/images/testImage.png'}
                      alt={video.title}
                      className='w-16 h-10 object-cover'
                    />
                    <button
                      onClick={() => setOverrideVideos(overrideVideos.filter((v) => v.video_id !== video.video_id))}
                      className='absolute top-0.5 right-0.5 p-0.5 rounded-full bg-black/60 hover:bg-red-500 text-white opacity-0 group-hover:opacity-100 transition-all cursor-pointer'
                      title='Remove video'
                    >
                      <XMarkIcon className='w-3 h-3' />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="flex  items-end gap-2 px-3 py-2">
              <Textarea
                {...register('prompt')}
                ref={(e) => {
                  register('prompt').ref(e);
                  chatRef.current = e;
                }}
                rows={1}
                className={clsx(
                  'block flex-1 border-none bg-transparent text-sm/6 text-text',
                  'focus:outline-none resize-none placeholder:text-text-muted',
                  'whitespace-pre-wrap leading-relaxed',
                  'max-h-[10rem] overflow-y-auto',
                  'self-center',
                  'scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent'
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
              <SendButton control={control} onSend={handlePrompt} querying={querying} onStop={stopStreaming} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
