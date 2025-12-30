import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Modal from '@mui/material/Modal';
import VideoJS from './VideoJS';
import React, { useContext } from 'react';
import { getFps, getImageKey, videoPath } from '@/utils/imagePath';
import { useStore } from '@/stores/questions';
import { get, set } from 'idb-keyval';
import { useStoreImages } from '@/stores/blobs';

export default function VideoModal({ image, open, onClose }) {
    // const { undoRef, redoRef } = useContext(AppContext)
    
    const { updateQuestionField, getCurrentQuestion, currentQuestionId } = useStore();
    const images = getCurrentQuestion().images;

    const playerRef = React.useRef(null);
    const intervalRef = React.useRef(null);
    const timeoutRef = React.useRef(null);
    // console.log(image, "sfjikahfuia")
    const videoid = `${image?.group_id}_${image?.video_id}.mp4`;
    const fps = getFps(videoid) || 25;
    const frameDuration = 1 / fps;

    const keyframe_id = Number(image?.keyframe_id);
    
    const allTimeStamps = images?.map(img => img.video_id == image?.video_id && img.group_id == image?.group_id ? img.keyframe_id : null).filter(Boolean);
    const markers = allTimeStamps?.map((keyframe_id) => ({
        time: keyframe_id * frameDuration,
        text: `Frame ${keyframe_id}`,
    }));

    // const startTime = Math.max(frameDuration * image?.key - 10, 0);
    // const endTime = frameDuration * image?.keyframe_id + 10;
    const [showFullTimeline, setShowFullTimeline] = React.useState(true);
    const videoJsOptions = {
        autoplay: false,
        controls: true,
        responsive: true,
        fluid: true,
        sources: [{
            src: videoPath(image?.video_id, image?.group_id),
            type: 'video/mp4'
        }],
        plugins: {
            offset: {
                start: showFullTimeline ? 0 : Math.max(playerRef.current.currentTime() - 10, 0),
                end: showFullTimeline ? undefined : playerRef.current.currentTime() + 10,
                restartBeginning: true
            }
        }
    };

    const handlePlayerReady = (player) => {
        playerRef.current = player;
        const currentTime = keyframe_id * frameDuration;
        player.addClass('vjs-has-started');
        player.currentTime(currentTime);
        player.markers({
            markers: markers,
            markerStyle: {
                'width': '8px',
                'background-color': 'red'
            },
            onMarkerReached(marker) {
            }
        });
    };
    const stepFrame = (direction) => {
        if (!playerRef.current) return;
        const currentTime = playerRef.current.currentTime();
        playerRef.current.currentTime(currentTime + direction * frameDuration);
    };

    const startHoldStep = (direction) => {
        stepFrame(direction);
        timeoutRef.current = setTimeout(() => {
            intervalRef.current = setInterval(() => {
                stepFrame(direction);
            }, 100);
        }, 500);

    };

    const stopHoldStep = () => {
        clearTimeout(timeoutRef.current);
        clearInterval(intervalRef.current);
    };

    const { setBlob } = useStoreImages();

    const cutFrame = async () => {
        if (!playerRef.current) return;

        const currentTime = playerRef.current.currentTime();
        const keyframe_id = Math.round(currentTime / frameDuration);

        // Ensure video element is ready
        const video = playerRef.current.el().querySelector("video");
        if (!video) return;

        // Create canvas to draw frame
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert to WebP Blob
        canvas.toBlob(async (blob) => {
            if (!blob) return;
            const blobKey = getImageKey(keyframe_id, image.video_id, image.group_id);
            await set(blobKey, blob);
            // Update frontend state with blob URL
            const newImage = {
                keyframe_id: keyframe_id,
                video_id: image.video_id,
                group_id: image.group_id,
                blobKey,
                fps: fps,
                score: 1.0, // because we picked this frame
            };

            setBlob(blobKey, blob);
            // setSortedImages(prev => [...prev, newImage]);
            updateQuestionField({'images': [...images, newImage]});
            // undoRef.current.push(images);
            // onClose();
        }, "image/webp", 0.9);
    };


    React.useEffect(() => {
        const handleKeyDown = (e) => {
            if (!playerRef.current) return;
            if (e.key === 'ArrowLeft') stepFrame(-1);
            if (e.key === 'ArrowRight') stepFrame(1);
            // if (e.code === 'Space') {
            //     e.preventDefault(); // Prevent page scroll
            //     if (!playerRef.current) return;
            //     const player = playerRef.current;
            //     player.paused() ? player.play() : player.pause();
            // }
        };
        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);

    }, [image]);

    return (
        <div>
            <Modal
                open={open}
                onClose={() => { onClose(); setShowFullTimeline(true); }}
                aria-labelledby="modal-modal-title"
                aria-describedby="modal-modal-description"
            >
                <Box sx={style}>
                    <VideoJS options={videoJsOptions} onReady={handlePlayerReady} key={showFullTimeline ? 'full' : 'clipped'} />
                    <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2, mb: 2 }}>
                        <Button variant="outlined"
                            onMouseDown={() => startHoldStep(-1)}
                            onMouseUp={stopHoldStep}
                            onMouseLeave={stopHoldStep}
                        >
                            -1 Frame (<kbd>←</kbd>)</Button>
                        <Button variant="outlined"
                            onMouseDown={() => startHoldStep(1)}
                            onMouseUp={stopHoldStep}
                            onMouseLeave={stopHoldStep}
                        >
                            +1 Frame (<kbd>→</kbd>)</Button>
                        <Button variant="outlined" onClick={() => setShowFullTimeline(!showFullTimeline)}>
                            {showFullTimeline ? 'Show Clip' : 'Show Full Timeline'}
                        </Button>
                        <Button variant="outlined" onClick={cutFrame}>Cut frame</Button>
                    </Box>

                </Box>
            </Modal>
        </div>
    );
}


const style = {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: "75vw",
    bgcolor: 'background.paper',
    border: '1px solid #000',
    borderRadius: 2,
    boxShadow: 24,
    p: 4,
};


