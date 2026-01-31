import React, { useState } from 'react'
import { useVideos } from '@/api/services/hooks/query';
import { useStore } from '@/stores/chat';
import LibraryModal from '../Library/LibraryModal';
import { PlusIcon } from '@heroicons/react/20/solid';

export default function VideosInConversation() {
  const groupId = useStore((state) => state.currentGroup);
  const sessionId = useStore((state) => state.session_id);
  const { data: videos = [] } = useVideos(groupId, sessionId);
  const selectedVideos = videos.filter(video => video.selected);

  // Mock data for testing - remove or set to [] in production
  const mockVideos = [
    { _id: '692ad412086ada3a309334ff', name: 'Introduction to AI', thumbnail: '/images/testImage.png', length: '12:34', selected: true },
    { _id: '692ad412086ada3a30933500', name: 'React Tutorial Part 1', thumbnail: '/images/testImage.png', length: '8:22', selected: true },
    { _id: '692ad412086ada3a30933501', name: 'Building Modern UIs', thumbnail: '/images/testImage.png', length: '15:00', selected: true },
    { _id: '692ad412086ada3a30933503', name: 'Video Editing Basics', thumbnail: '/images/testImage.png', length: '20:15', selected: true },
  ];

  // Use mock if selectedVideos is empty
  const displayVideos = groupId === '65c2f9a4e8b13d7c0a4f92be' ? [...mockVideos] : selectedVideos;

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [focusVideoId, setFocusVideoId] = useState(null);

  const closeModal = () => {
    setIsModalOpen(false);
    setFocusVideoId(null);
  };
  const openModal = () => setIsModalOpen(true);

  const handleVideoClick = (videoId) => {
    setFocusVideoId(videoId);
    setIsModalOpen(true);
  };

  return (
    <div className='relative flex flex-col h-full'>
      {/* Add Videos Button */}
      <div className='sticky top-0 px-2 py-2 border-b border-surface-light'>
        <button
          onClick={openModal}
          className='flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium transition-colors cursor-pointer'
        >
          <PlusIcon className="w-5 h-5" />
          <span>Add Videos</span>
        </button>
      </div>

      {/* Active Videos Section */}
      <div className='flex flex-col flex-1 overflow-hidden'>
        <p className='text-xs text-text-muted uppercase tracking-wide px-4 py-2'>
          Active Videos {displayVideos.length > 0 && `(${displayVideos.length})`}
        </p>
        <div className='columns-2 gap-2 px-2 flex-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-auto'>
          {displayVideos.length === 0 ? (
            <p className='text-sm text-text-dim col-span-2 text-center py-4'>No videos selected</p>
          ) : (
            displayVideos.map((video, idx) => (
              <div
                key={idx}
                className='pb-2 break-inside-avoid group cursor-pointer'
                onClick={() => handleVideoClick(video._id)}
              >
                <div className='relative overflow-hidden rounded-lg border border-white/10 group-hover:border-accent/50 transition-colors'>
                  <img
                    src={video.thumbnail || "/images/testImage.png"}
                    alt="thumbnail"
                    className='w-full aspect-video object-cover'
                  />
                  {/* Duration badge */}
                  <span className='absolute bottom-1 right-1 px-1.5 py-0.5 text-xs font-medium bg-black/70 text-white rounded'>
                    {video.length || "1:00"}
                  </span>
                </div>
                <h3 className='mt-1.5 text-sm font-medium truncate text-text-muted group-hover:text-text transition-colors'>
                  {video.name}
                </h3>
              </div>
            ))
          )}
        </div>
      </div>

      <LibraryModal isModalOpen={isModalOpen} closeModal={closeModal} focusVideoId={focusVideoId} />
    </div>
  )
}
