import VideoJS from '@/components/common/components/VideoPlayer/VideoJS'
import Modal from '@/components/Modal/modal'
import React from 'react'

export default function VideoModal({ isModalOpen, closeModal, title="Video name", video }) {
  if (!video) return null;
      const videoJsOptions = {
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
        sources: [{
          src: video.url,
          type: 'video/mp4'
        }]
      }
    
  return (
    <Modal isOpen={isModalOpen} onClose={closeModal} title={title} zIndex='z-60'>
      <div className=''>
        <VideoJS options={videoJsOptions} />
      </div>
    </Modal>
  )
}
