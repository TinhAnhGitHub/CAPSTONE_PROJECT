import React from 'react'

export default function VideoPlayer({video}) {
  return (
    <div>
        {/* thumbnail here */}
        <video src={video.url} controls></video>
        <h2>{video.title}</h2>
    </div>
  )
}
