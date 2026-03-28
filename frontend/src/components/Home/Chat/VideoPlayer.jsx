import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react'
import Modal from '../../Modal/modal'
import VideoJS from '../../common/components/VideoPlayer/VideoJS'
import { useStore } from '@/stores/chat'
import { PlusCircleIcon, CheckCircleIcon } from '@heroicons/react/20/solid'
import Markdown from 'react-markdown'
import SaveKeyframes from './Savekeyframes.jsx'

const MAX_RETRIES = 20
const RETRY_DELAY = 15000 // 15s between retries, gives up after ~5min

function PreviewImage({ src, alt, className }) {
  const [status, setStatus] = useState('loading') // 'loading' | 'loaded' | 'failed'
  const [retries, setRetries] = useState(0)
  const [imgSrc, setImgSrc] = useState(src)

  useEffect(() => {
    setStatus('loading')
    setRetries(0)
    setImgSrc(src)
  }, [src])

  const handleLoad = () => setStatus('loaded')

  const handleError = () => {
    if (retries < MAX_RETRIES) {
      setStatus('loading')
      setTimeout(() => {
        setRetries(r => r + 1)
        // cache-bust the retry
        setImgSrc(`${src}${src.includes('?') ? '&' : '?'}retry=${retries + 1}`)
      }, RETRY_DELAY)
    } else {
      setStatus('failed')
    }
  }

  if (status === 'failed') {
    return (
      <div className={`${className} bg-surface-light flex items-center justify-center`} title="Image unavailable">
        <svg className="w-4 h-4 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
    )
  }

  return (
    <div className={`${className} relative`}>
      {status === 'loading' && (
        <div className="absolute inset-0 bg-surface-light flex items-center justify-center rounded-md">
          <svg className="w-4 h-4 text-text-muted animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
      )}
      <img
        src={imgSrc}
        alt={alt}
        className={`rounded-md h-full w-full object-cover ${status === 'loading' ? 'invisible' : ''}`}
        onLoad={handleLoad}
        onError={handleError}
      />
    </div>
  )
}

export default function VideoPlayer({ video }) {
  const [isModalOpen, setIsModalOpen] = useState(false)
  const playerRef = useRef(null)

  console.log(video)

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

  const videoJsOptions = useMemo(() => ({
    autoplay: false,
    controls: true,
    responsive: true,
    fluid: true,
    poster: video.thumbnail,
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

  // toggle override videos
  const addVideoToChatSession = (e) => {
    e.stopPropagation()
    if (overrideVideos.find(v => v.video_id === video.video_id)) {
      // already added, remove it
      setOverrideVideos(overrideVideos.filter(v => v.video_id !== video.video_id))
    } else {
      // not added, add it
      setOverrideVideos([...overrideVideos, video])
    }
  }

  return (
    <>
      <div
        className="border border-white/10 rounded-lg overflow-hidden hover:cursor-pointer hover:opacity-80"
        onClick={openVideoModal}
      >
        {/* Video thumbnail */}
        <div className="relative">
          {/* <video
            src={video.url}
            className="w-full h-auto"
            preload="metadata"
          /> */}
          <img src={video.thumbnail || '/images/video_placeholder.png'} alt={video.title} className="w-full h-auto object-cover" />
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
        size="xl"
      >
        <div className='flex flex-col lg:flex-row h-full lg:items-center gap-3'>
          {/* 3 parts */}
          <div className='flex-1 lg:flex-2 max-lg:max-h-[50%] lg:h-full'>
            <VideoJS options={videoJsOptions} onReady={handlePlayerReady} />
            {/* settings like, save allkeyframes, save all images from keyframe */}
            <div className='m-2'>
              <SaveKeyframes segments={video.segments} videoId={video.video_id} videoName={video.title} />
            </div>
          </div>
          {/* 1 part */}
          <div className='flex flex-col flex-1 max-lg:min-h-[50%] lg:h-full'>
            <h3 className="text-xl font-medium text-text mb-2">{video.segments.length} Matches</h3>
            <div className="h-[90%] overflow-y-auto flex-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent">
              {
                video.segments?.map((segment, i) => {
                  const hue = (i * 137.5) % 360
                  const color = `hsl(${hue}, 80%, 55%)`
                  const startTime = segment.start !== undefined
                    ? segment.start * frameDuration
                    : segment.start_time

                  const images = segment.preview_images.length === 0
                    ? Array(5).fill('/images/testImage.png')
                    : segment.preview_images

                  return (
                    <div
                      key={i}
                      className=" flex flex-col gap-2 border border-white/10 rounded-lg p-2 my-2 cursor-pointer hover:bg-black/10 transition-colors w-full"
                      onClick={() => playerRef.current?.currentTime(startTime)}
                      style={{ borderLeftColor: color, borderLeftWidth: '3px' }}
                    >
                      <div className='flex items-center justify-between'>
                        <div className="text-sm font-medium text-text-muted border rounded-md p-1 self-start min-w-7 flex items-center justify-center my-1 font-mono">{i + 1}</div>
                        <span className="text-text-muted text-xs border rounded-md p-1 font-mono">
                          {(segment.start !== undefined ? frameToTimeStr(segment.start, frameDuration) : segment.start_time !== undefined ? segment.start_time.toFixed(2) + 's' : 'N/A') + " - " +
                            (segment.end !== undefined ? frameToTimeStr(segment.end, frameDuration) : segment.end_time !== undefined ? segment.end_time.toFixed(2) + 's' : 'N/A')}
                        </span>
                      </div>
                      <div className='w-full justify-between flex gap-1 bg-black/70 rounded-lg '>
                        {
                          images.map((img, idx) => (
                            <PreviewImage
                              key={idx}
                              src={img}
                              alt={`Segment ${i + 1} preview ${idx + 1}`}
                              className="rounded-md h-12 min-w-0 flex-1"
                            />
                          ))
                        }
                      </div>
                      {segment.caption && <div className="text-text-muted text-sm mb-1"><Markdown>{segment.caption}</Markdown></div>}
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