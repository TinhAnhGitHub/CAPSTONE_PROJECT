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
  const [isModalOpen, setIsModalOpen] = useState(false);
  const closeModal = () => setIsModalOpen(false);
  const openModal = () => setIsModalOpen(true);

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
          Active Videos {selectedVideos.length > 0 && `(${selectedVideos.length})`}
        </p>
        <div className='columns-2 gap-2 px-2 flex-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-auto'>
          {selectedVideos.length === 0 ? (
            <p className='text-sm text-text-dim col-span-2 text-center py-4'>No videos selected</p>
          ) : (
            selectedVideos.map((video, idx) => (
              <div key={idx} className='pb-2 break-inside-avoid'>
                <img src={video.thumbnail || "/images/testImage.png"} alt="thumbnail" className='w-full h-auto rounded-lg' />
                <div className='mt-1'>
                  <h3 className='text-sm font-medium truncate text-text'>{video.name}</h3>
                  <p className='text-xs text-text-dim'>{video.length || "1:00"}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <LibraryModal isModalOpen={isModalOpen} closeModal={closeModal} />
    </div>
  )
}
