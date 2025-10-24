import React from 'react'

export default function ImageViewer({image}) {
  return (
    <div>
        <img src={image.url} alt={image.title} />
    </div>
  )
}
