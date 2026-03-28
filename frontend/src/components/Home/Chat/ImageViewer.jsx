import React, { useState, useMemo, memo } from 'react'
import Modal from '../../Modal/modal'
import { Gallery } from '../../common/components/ImagePreview/Gallery'

export default memo(function ImageViewer({ image, images = [], startIndex = 0, className = '' }) {
  const [isModalOpen, setIsModalOpen] = useState(false)

  const openImageGallery = () => {
    setIsModalOpen(true)
  }

  const closeImageGallery = () => {
    setIsModalOpen(false)
  }

  // Memoize gallery images to prevent recalculation
  const galleryImages = useMemo(() =>
    images.length > 0
      ? images.map(img => ({
        src: img.url,
        thumb: img.url,
        subHtml: img.title ? `<div class="lightGallery-captions"><h4>${img.title}</h4></div>` : ''
      }))
      : [{
        src: image.url,
        thumb: image.url,
        subHtml: image.title ? `<div class="lightGallery-captions"><h4>${image.title}</h4></div>` : ''
      }]
    , [images, image.url, image.title])

  return (
    <>
      <div
        className="border border-white/10 rounded-lg overflow-hidden hover:cursor-pointer hover:opacity-80 h-full"
        onClick={openImageGallery}
      >
        <img src={image.url} alt={image.title} className={className || 'w-full h-auto'} />
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeImageGallery}
        size="xl"
        showCloseButton={true}
      >
        <div className="h-full overflow-hidden">
          <Gallery images={galleryImages} startIndex={startIndex} />
        </div>
      </Modal>
    </>
  )
})
