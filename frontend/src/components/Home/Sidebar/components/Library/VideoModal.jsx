import VideoJS from '@/components/common/components/VideoPlayer/VideoJS'
import Modal from '@/components/Modal/modal'
import { useStore as useChatStore } from '@/stores/chat'
import { PlusCircleIcon, CheckCircleIcon } from '@heroicons/react/20/solid'
import React, { useMemo, useState, useEffect, useRef } from 'react'

export default function VideoModal({ isModalOpen, closeModal, video, startTime }) {
  // ALL hooks must run unconditionally — before any early return
  const overrideVideos = useChatStore((s) => s.overrideVideos);
  const setOverrideVideos = useChatStore((s) => s.setOverrideVideos);
  const playerRef = useRef(null);

  // Keep a cached copy of video so the exit animation isn't cut short
  // (close() sets video=null immediately, which would unmount before the fade-out)
  const [displayVideo, setDisplayVideo] = useState(null);
  useEffect(() => {
    if (video) setDisplayVideo(video);
  }, [video]);

  // Seek to startTime whenever the modal opens with a new startTime
  useEffect(() => {
    if (isModalOpen && startTime != null && playerRef.current) {
      playerRef.current.currentTime(startTime);
    }
  }, [isModalOpen, startTime]);

  // Normalise: chip videos use _id/id; VideoPlayer uses video_id
  const videoId = displayVideo?._id ?? displayVideo?.id ?? displayVideo?.video_id;
  const isAdded = overrideVideos.some(v => (v.video_id ?? v._id ?? v.id) === videoId);

  // Memoised so VideoJS never sees a new options object on re-render (prevents reload)
  const videoJsOptions = useMemo(() => ({
    autoplay: false,
    controls: true,
    responsive: true,
    fluid: true,
    aspectRatio: '16:9',
    controlBar: {
      children: [
        'playToggle',
        'volumePanel',
        'currentTimeDisplay',
        'timeDivider',
        'durationDisplay',
        'progressControl',
        'fullscreenToggle',
      ],
    },
    sources: [{ src: displayVideo?.url, type: 'video/mp4' }],
  }), [displayVideo?.url]); // only rebuild if the URL changes

  // Guard after hooks - only block render if we've never had a video
  if (!displayVideo) return null;

  const toggleContext = () => {
    if (isAdded) {
      setOverrideVideos(overrideVideos.filter(v => (v.video_id ?? v._id ?? v.id) !== videoId));
    } else {
      setOverrideVideos([...overrideVideos, {
        video_id: videoId,
        url: video.url,
        title: video.name ?? video.title ?? videoId,
        thumbnail: video.thumbnail ?? null,
      }]);
    }
  };

  const handlePlayerReady = (player) => {
    playerRef.current = player;
    // Seek once the player has loaded enough metadata
    if (startTime != null) {
      player.on('loadedmetadata', () => {
        player.currentTime(startTime);
      });
    }
  };

  return (
    <Modal isOpen={isModalOpen} onClose={closeModal} title={displayVideo.name} zIndex='z-60'>
      <VideoJS options={videoJsOptions} onReady={handlePlayerReady} />

      {/* Add-to-context button */}
      <div className="flex items-center justify-end mt-3">
        <button
          onClick={toggleContext}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors cursor-pointer hover:bg-surface-light"
          title={isAdded ? 'Remove from chat context' : 'Add to chat context'}
        >
          {isAdded ? (
            <>
              <CheckCircleIcon className="w-5 h-5 text-accent" />
              <span className="text-accent">Added to context</span>
            </>
          ) : (
            <>
              <PlusCircleIcon className="w-5 h-5 text-text-muted hover:text-accent transition-colors" />
              <span className="text-text-muted">Add to context</span>
            </>
          )}
        </button>
      </div>
    </Modal>
  )
}