import React from 'react'
import { useVideos } from '@/api/services/hooks/query';
import { useStore } from '@/stores/chat';

export default function VideosInConversation() {
  const groupId = useStore((state) => state.currentGroup);
  const sessionId = useStore((state) => state.session_id);
  const { data: videos = [] } = useVideos(groupId, sessionId);
  const selectedVideos = videos.filter(video => video.selected);

  return (
    <div className='relative flex flex-col text-sm text-gray-400/60 h-full'>
      <div className=' flex justify-between items-center bg-black border-b border-gray-800 p-3'>
        <p className=''>Workspace videos</p>
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
    </div>
  )
}
