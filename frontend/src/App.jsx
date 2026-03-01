import React from 'react'
import Home from './pages/Home'
import { Outlet } from 'react-router-dom'
import { Toaster } from 'react-hot-toast';
import VideoModal from './components/Home/Sidebar/components/Library/VideoModal';
import { useVideoModalStore } from './stores/videoModal';
export default function App() {
  // isModalOpen, closeModal, title="Video name", video
  const { isOpen, video, close } = useVideoModalStore();

  return (
    <div>
      <Outlet />
      <Toaster reverseOrder={false} />

      {/* video modal */}
      <VideoModal isModalOpen={isOpen}
        video={video}
        closeModal={close}
      />
    </div>
  )
}
