import React, { useState, useRef, useMemo, useCallback } from 'react'
import Modal from '../../Modal/modal'
import VideoJS from '../../common/components/VideoPlayer/VideoJS'
import { useStore } from '@/stores/chat'

export default function VideoPlayer({ video }) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const playerRef = useRef(null)

  const setOverrideVideos = useStore((state) => state.setOverrideVideos)
  const overrideVideos = useStore((state) => state.overrideVideos)
  const openVideoModal = () => {
    setIsModalOpen(true)
  }

  const closeVideoModal = () => {
    setIsModalOpen(false)
  }

  const fps = video.fps || 30
  const frameDuration = 1 / fps

  // Memoize markers with colors embedded
  const markers = useMemo(() =>
    video.segments?.flatMap((segment, i) => {
      const markers = []
      const hue = (i * 137.5) % 360
      const color = `hsl(${hue}, 80%, 55%)`

      if (segment.start_frame !== undefined) {
        markers.push({
          time: segment.start_frame * frameDuration,
          text: `Segment ${i + 1} Start`,
          class: `marker-segment-${i}`,
          color: color,
        })
        if (segment.end_frame !== undefined) {
          markers.push({
            time: segment.end_frame * frameDuration,
            text: `Segment ${i + 1} End`,
            class: `marker-segment-${i}`,
            color: color,
          })
        }
      } else if (segment.start_time !== undefined) {
        markers.push({
          time: segment.start_time,
          text: `Segment ${i + 1} Start`,
          class: `marker-segment-${i}`,
          color: color,
        })
        if (segment.end_time !== undefined) {
          markers.push({
            time: segment.end_time,
            text: `Segment ${i + 1} End`,
            class: `marker-segment-${i}`,
            color: color,
          })
        }
      }
      return markers
    }) || []
    , [video.segments, frameDuration])

  // Memoize video options to prevent VideoJS from re-initializing
  const videoJsOptions = useMemo(() => ({
    autoplay: false,
    controls: true,
    responsive: true,
    fluid: true,
    sources: [{
      src: video.url,
      type: 'video/mp4'
    }]
  }), [video.url])

  // Memoize the callback to prevent VideoJS from re-initializing
  const handlePlayerReady = useCallback((player) => {
    playerRef.current = player

    // Add markers if segments exist
    if (markers.length > 0) {
      player.markers({
        markers: markers,
        markerStyle: {
          'width': '8px',
          'border-radius': '2px',
        },
        onMarkerClick(marker) {
          // Jump to marker time when clicked
          player.currentTime(marker.time)
        },
        onMarkerReached(marker) {
          console.log('Marker reached:', marker.text)
        }
      })

      // Apply colors to markers after they're created
      const applyColors = () => {
        markers.forEach((marker) => {
          const markerEls = document.querySelectorAll(`.${marker.class}`)
          markerEls.forEach(el => {
            el.style.backgroundColor = marker.color
          })
        })
      }

      // Try multiple times to ensure markers are rendered
      setTimeout(applyColors, 50)
      setTimeout(applyColors, 200)
    }

    player.on('waiting', () => {
      console.log('player is waiting')
    })

    player.on('dispose', () => {
      console.log('player will dispose')
    })
  }, [markers])

  const addVideoToChatSession = (e) => {
    e.stopPropagation()
    // check if video already in overrideVideos
    if (overrideVideos.find(v => v.video_id === video.video_id)) {
      return
    }
    setOverrideVideos([...overrideVideos, video])    
  }

  return (
    <>
      <div
        className="border border-white/10 rounded-lg overflow-hidden hover:cursor-pointer hover:opacity-80"
        onClick={openVideoModal}
      >
        {/* Video thumbnail */}
        <div className="relative">
          <video
            src={video.url}
            className="w-full h-auto"
            preload="metadata"
          />
          <div className="absolute inset-0 flex items-center justify-center bg-black/30">
            <div className="w-12 h-12 rounded-full bg-white/80 flex items-center justify-center">
              <svg className="w-6 h-6 text-gray-800" fill="currentColor" viewBox="0 0 24 24">
                <path d="M8 5v14l11-7z" />
              </svg>
            </div>
          </div>
        </div>
        <div className="flex items-center justify-between ">
          {video.title && <h2 className="p-2 text-sm text-white">{video.title}</h2>}
          <div className="p-2 text-sm cursor-pointer" onClick={addVideoToChatSession}>Add</div>
        </div>
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeVideoModal}
        title={video.title || "Video Player"}
        size="xl"
      >
        <div className="h-full flex items-center justify-center">
          <div className="w-full max-w-5xl">
            <VideoJS options={videoJsOptions} onReady={handlePlayerReady} />
          </div>
        </div>
      </Modal>
    </>
  )
}
