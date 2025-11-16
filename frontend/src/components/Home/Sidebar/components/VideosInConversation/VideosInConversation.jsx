import React, { useState } from 'react'
import { useVideos } from '@/api/services/hooks/query';
import { useStore } from '@/stores/chat';
import LibraryModal from '../Library/LibraryModal';

export default function VideosInConversation() {
  const groupId = useStore((state) => state.currentGroup);
  const sessionId = useStore((state) => state.session_id);
  const { data: videos = [] } = useVideos(groupId, sessionId);
  const selectedVideos = videos.filter(video => video.selected);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const closeModal = () => setIsModalOpen(false);
  const openModal = () => setIsModalOpen(true);

  return (
    <div className='relative flex flex-col text-sm text-gray-400/60 h-full'>
      <div className=' flex justify-between items-center bg-black border-b border-gray-800 px-2 py-1'>
        <p className=''>Workspace videos</p>
        <div className='z-30 hover:bg-gray-800 p-2 rounded-full cursor-pointer transition' onClick={openModal}>
          <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="size-6">
            <path strokeLinecap="round" strokeLinejoin="round" d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z" />
          </svg>
        </div>
      </div>
      <div className='columns-2 gap-2 p-2 scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-300 overflow-y-auto '>
        {
          selectedVideos.map((video, idx) => (
            <div key={idx} className='pb-1 cursor-pointer break-inside-avoid'>
              <img src={video.thumbnail || "/images/testImage.png"} alt="thumbnail" className='w-full h-auto rounded-lg' />
              <div>
                <h3 className='text-sm font-semibold truncate'>{video.name}</h3>
                <p className='text-xs text-gray-500'>{video.length || "1:00"}</p>
              </div>
            </div>
          ))
        }
      </div>
      <LibraryModal isModalOpen={isModalOpen} closeModal={closeModal} />
    </div>
  )
}
