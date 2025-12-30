import React from 'react'

export default function ImageViewer({image}) {
  return (
    <div className="border border-white/10 rounded-lg overflow-hidden">
        <img src={image.url} alt={image.title} />
    </div>
  )
}
