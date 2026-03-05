import React, { useState, useEffect, useRef } from 'react'
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
  const isStreamingRef = useRef(false); // Track if we received continue_stream

  // chạy khi chuyển session
  useEffect(() => {
    isStreamingRef.current = false;
    setQuerying(false); // Reset querying state - will be set to true by continue_stream if needed
    setChatMessages([]); // Clear messages to prevent accumulation when switching
    // Scroll to bottom after messages are rendered
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: 'instant' });
    });
  }, [session_id]);

  // báo join session, mà hình như api bên be có tự động check on session hay sao
  useEffect(() => {
    const handleConnect = () => {
      const currentSessionId = getSessionId();
      if (currentSessionId) {
        socket.emit('join_session', { session_id: currentSessionId });
      }
    };

    // Join immediately if already connected
    if (socket.connected) {
      handleConnect();
    }

    socket.on('connect', handleConnect);
    return () => {
      socket.off('connect', handleConnect);
    };
  }, []);

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
      if (isStreamingRef.current) {
        setChatMessages([...data, ...chatMessages]);
      } else setChatMessages(data);
      requestAnimationFrame(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'instant' });
      });
    },
    enabled: !!session_id,
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  })

  // handle socket
  useEffect(() => {
    // response stream bằng text
    const handleResponse = (msg) => {
      // Ignore messages from other sessions
      if (msg.session_id && msg.session_id !== getSessionId()) return;

      const prev = useStoreChat.getState().chatMessages;

      const newBlock = parseChunkToBlock("text", msg.content_delta);
      if (!newBlock) return;

      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };
    // response stream bằng hình/video

    const handleMedia = (media) => {
      // Ignore messages from other sessions
      if (media.session_id && media.session_id !== getSessionId()) return;

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


    // handle stream thinking
    const handleThinking = (data) => {
      // Ignore messages from other sessions
      if (data.session_id && data.session_id !== getSessionId()) return;

      const newBlock = parseChunkToBlock('thinking', data)
      if (!newBlock) return;

      const prev = useStoreChat.getState().chatMessages;
      // Fix: actually update state with the result
      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);

      scrollToBottomIfNeeded();
    };

    // handle toolcall
    const handleToolCall = (data) => {
      // Ignore messages from other sessions
      if (data.session_id && data.session_id !== getSessionId()) return;

      const newBlock = parseChunkToBlock('tools', data)
      if (!newBlock) return;

      const prev = useStoreChat.getState().chatMessages;
      // Fix: actually update state with the result
      const updated = addBlockToMessages(prev, 'assistant', newBlock);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
    }
    //handle toolcallresult
    const handleToolCallResult = (data) => {
      // Ignore messages from other sessions
      if (data.session_id && data.session_id !== getSessionId()) return;

      // find the tool name
      const prev = useStoreChat.getState().chatMessages;
      const finished_tool_name = data.tool_name;
      const updated = updateToolCallBlock(prev, finished_tool_name);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
    }

    const handleMessageReceived = (msg) => {
      // maybe set a flag to set the send button to cancel button
      setQuerying(true);
    }
    const handleContinueStream = (msg) => {
      // Ignore messages from other sessions
      if (msg.session_id && msg.session_id !== getSessionId()) return;
      const data = msg.content;
      console.log("continue_stream data", data);
      if (!data || !Array.isArray(data) || data.length === 0) return;

      // Mark that we're streaming - prevents query from overwriting
      isStreamingRef.current = true;

      const prev = useStoreChat.getState().chatMessages;

      // Check if last message is already an assistant streaming message
      // If so, replace it instead of appending to avoid duplicates
      let baseMessages = prev;
      if (prev.length > 0) {
        const lastMsg = prev[prev.length - 1];
        // If last message is assistant and has similar block types, it's likely the same streaming message
        if (lastMsg.role === 'assistant') {
          // Remove the last assistant message - we'll replace it with fresh data
          baseMessages = prev.slice(0, -1);
        }
      }

      // Add the streaming blocks as a new assistant message
      const updated = addBlocksToMessages(baseMessages, 'assistant', data);
      console.log("continue_stream updated", updated);
      setChatMessages(updated);
      scrollToBottomIfNeeded();
      setQuerying(true);
    }
    const handleStreamEnd = (msg) => {
      isStreamingRef.current = false; // Reset streaming flag
      setQuerying(false);
    }

    socket.on('message_received', handleMessageReceived);
    // handle answer
    socket.on('response', handleResponse);
    socket.on('thinking', handleThinking);
    socket.on('media', handleMedia);
    socket.on('tool_call', handleToolCall);
    socket.on('tool_result', handleToolCallResult);
    socket.on('stream_end', handleStreamEnd);

    socket.on('continue_stream', handleContinueStream);

    return () => {
      socket.off('message_received', handleMessageReceived);
      // socket.off('message_received', handleStatus);
      socket.off('response', handleResponse);
      socket.off('media', handleMedia);
      socket.off('thinking', handleThinking);
      socket.off('tool_call', handleToolCall);
      socket.off('tool_result', handleToolCallResult);
      socket.off('stream_end', handleStreamEnd);
      socket.off('continue_stream', handleContinueStream);
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
  const scrollToBottomIfNeeded = (behavior = 'instant') => {
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
    socket.emit('cancel_stream', { session_id: getSessionId() });
    setQuerying(false);
  };

  // handle optimistic UI update when send prompt
  const handlePrompt = async () => {
    const prompt = getValues('prompt').trim();
    if (!prompt) return;
    const data = { userId, sessionId: getSessionId(), text: prompt, videos: selectedVideosIds }
    socket.emit('stream_chat', data);
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
      <div className='sticky top-0 border-b flex-shrink-0 max-md:hidden border-surface-light h-14'>
      </div>
      <div
        ref={chatContainerRef}
        // onScroll={() => { isNearBottomRef.current = checkIfNearBottom(); }}
        onScroll={handleScroll}
        className="flex flex-col w-full h-full px-4 md:px-8 lg:px-16 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-scroll scrollbar-gutter-stable"
      >
        {/* Centered content container with max-width */}
        <div className="w-full max-w-3xl mx-auto flex flex-col gap-3 py-6">
          {chatMessages.map((m, i) => (
            <div key={i} className='w-full flex flex-col'>
              {m.blocks.map((block, j) => (
                <BlockRenderer key={`${i}-${j}`} block={block} role={m.role} isLastMessage={i === chatMessages.length - 1} />
              ))}
            </div>
          ))}

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

            <div className="flex items-end gap-2 px-3 py-2">
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
                  if (querying) return;
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
