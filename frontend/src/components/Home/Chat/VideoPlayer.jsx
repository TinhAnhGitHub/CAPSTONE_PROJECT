import React, { useState, useRef, useMemo, useCallback } from 'react'
import Modal from '../../Modal/modal'
import VideoJS from '../../common/components/VideoPlayer/VideoJS'
import { useStore } from '@/stores/chat'
import { PlusCircleIcon, CheckCircleIcon } from '@heroicons/react/20/solid'
import Markdown from 'react-markdown'

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

      if (segment.start !== undefined) {
        markers.push({
          time: segment.start * frameDuration,
          // text: `Segment ${i + 1} Start`,
          text: segment.caption,
          class: `marker-segment-${i}`,
          color: color,
        })
        if (segment.end !== undefined) {
          markers.push({
            time: segment.end * frameDuration,
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
  }), [video.url])

  // Memoize the callback to prevent VideoJS from re-initializing
  const handlePlayerReady = useCallback((player) => {
    playerRef.current = player

    // Force show time controls (workaround for responsive hiding)
    const controlBar = player.controlBar
    if (controlBar) {
      const timeControls = player.el().querySelectorAll('.vjs-time-control, .vjs-current-time, .vjs-duration, .vjs-time-divider')
      timeControls.forEach(el => {
        el.style.display = 'flex'
        el.style.paddingLeft = '0.5em'
        el.style.paddingRight = '0.5em'
      })
    }

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
      // console.log('player is waiting')
    })

    player.on('dispose', () => {
      // console.log('player will dispose')
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
        <div className="flex items-center justify-between p-2">
          {video.title && <h2 className="text-sm text-text truncate flex-1">{video.title}</h2>}
          <button
            onClick={addVideoToChatSession}
            className="p-1 rounded-full hover:bg-surface-light transition-colors cursor-pointer"
            title={overrideVideos.find(v => v.video_id === video.video_id) ? "Already added" : "Add to chat"}
          >
            {overrideVideos.find(v => v.video_id === video.video_id) ? (
              <CheckCircleIcon className="w-6 h-6 text-accent" />
            ) : (
              <PlusCircleIcon className="w-6 h-6 text-text-muted hover:text-accent transition-colors" />
            )}
          </button>
        </div>
      </div>

      <Modal
        isOpen={isModalOpen}
        onClose={closeVideoModal}
        title={video.title || "Video Player"}
        size="lg"
      >
        <div className='flex flex-col lg:flex-row gap-3'>
          {/* 3 parts */}
          <div className='flex-3'>
            <VideoJS options={videoJsOptions} onReady={handlePlayerReady} />
          </div>
          {/* 1 part */}
          <div className='flex-1'>
            <h3 className="text-xl font-medium text-text mb-2">{video.segments.length} Matches</h3>
            <div className="max-h-96 overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent">
              {video.segments?.map((segment, i) => {
                const hue = (i * 137.5) % 360
                const color = `hsl(${hue}, 80%, 55%)`
                const startTime = segment.start !== undefined
                  ? segment.start * frameDuration
                  : segment.start_time

                return (
                  <div
                    key={i}
                    className="border border-white/10 rounded-lg p-2 my-2 cursor-pointer hover:bg-surface-light transition-colors w-full flex flex-col"
                    onClick={() => playerRef.current?.currentTime(startTime)}
                    style={{ borderLeftColor: color, borderLeftWidth: '3px' }}
                  >
                    <div className="text-sm font-medium text-text border rounded-md p-1 self-start w-7 h-7 flex items-center justify-center my-1 font-mono">{i + 1}</div>
                    {segment.caption && <div className="text-text-muted text-sm mb-1"><Markdown>{segment.caption}</Markdown></div>}
                    <span className="text-text-muted text-xs border rounded-md mt-1 p-1 self-end font-mono">
                      {(segment.start !== undefined ? frameToTimeStr(segment.start, frameDuration) : segment.start_time !== undefined ? segment.start_time.toFixed(2) + 's' : 'N/A') + " - " +
                        (segment.end !== undefined ? frameToTimeStr(segment.end, frameDuration) : segment.end_time !== undefined ? segment.end_time.toFixed(2) + 's' : 'N/A')}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </Modal>
    </>
  )
}

function frameToTimeStr(frame, frameDuration) {
  const totalSeconds = frame * frameDuration
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = (totalSeconds % 60).toFixed(2)
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(5, '0')}`
}