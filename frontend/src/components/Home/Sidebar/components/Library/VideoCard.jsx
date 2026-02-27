import React from 'react'
import { useMutation, useQueryClient } from 'react-query';
import api from '@/api/api';
import VideoDropdownList from './VideoDropdownList';
import SelectedIcon from './SelectedIcon';
import { useStore } from '@/stores/chat';
import clsx from 'clsx';
import { ingested, errorIngested } from '@/utils/library';
import IngestedStatus from './IngestedStatus/IngestedStatus';
import useEdit from '@/api/services/hooks/edit';
import { ArrowPathIcon } from '@heroicons/react/20/solid';
import VideoModal from './VideoModal';

export default function VideoCard({ video, isHighlighted = false, onEdit }) {
    const {
        isEditing,
        editValue,
        setEditValue,
        startEditing,
        saveEdit,
        cancelEdit,
    } = useEdit({
        initialValue: video.name,
        onSave: (value) => onEdit?.(video._id, value),
    });

    const queryClient = useQueryClient();
    const session_id = useStore((state) => state.session_id);
    const selectMutation = useMutation({
        mutationFn: async ({ video_id, session_id }) => {
            return await api.post('/api/user/videos/select', {
                video_ids: [video_id],
                session_id: session_id,
            })
        },
        onSettled: () => {
            queryClient.invalidateQueries(['videos']);
        }
    })
    const handleToggleSelect = async (video_id, session_id) => {
        selectMutation.mutate({ video_id, session_id });
    }
    const retryMutation = useMutation({
        mutationFn: async (video_id) => {
            return await api.post('/api/user/ingestion/retry', {
                video_ids: [video_id],
            })
        },
        onSettled: () => {
            queryClient.invalidateQueries(['videos']);
        }
    })
    const handleFailed = (e) => {
        // call api to re-ingest
        e.stopPropagation();
        console.log("Re-ingest video:", video._id);
        retryMutation.mutate(video._id);
    }

    const failed = video.ingested_status === -1;

    const [isOpen, setIsOpen] = React.useState(false);
    const openModal = () => setIsOpen(true);
    const closeModal = () => {setIsOpen(false); console.log("Close modal")};
    return (
        <div className={clsx(
            "group relative rounded-xl p-2 cursor-pointer transition-all",
            "hover:bg-white/5",
            (errorIngested(video.ingested_status) || !ingested(video.ingested_status)) && "opacity-50",
            isHighlighted && "animate-highlight-pulse ring-2 ring-accent ring-offset-2 ring-offset-background rounded-xl")}
            onClick={openModal}>
            {/* Thumbnail */}
            <div className='relative aspect-video rounded-lg overflow-hidden bg-black'>
                <img
                    src={video.thumbnail || "/images/testImage.png"}
                    alt="thumbnail"
                    className="w-full h-full object-cover"
                />
                {!failed && <div className='absolute inset-0 flex items-center justify-center'>
                    <IngestedStatus percentage={video.ingested_status} />
                </div>}
                {/* Selection indicator */}
                <div className={
                    clsx('absolute top-1 right-1 rounded-md p-1 bg-black/50 backdrop-blur-sm cursor-pointer hover:bg-black/70  transition-colors',
                        (!ingested(video.ingested_status) || failed) && "!cursor-not-allowed opacity-50")
                }

                    onClick={(e) => {
                        e.stopPropagation();
                        if (!ingested(video.ingested_status)) return;
                        handleToggleSelect(video._id, session_id)
                    }}>
                    <div className='rounded-md p-1 hover:bg-white/10 '>
                        <SelectedIcon selected={video.selected} />
                    </div>
                </div>
                {failed && (
                    <div className='absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-500/10 backdrop-blur-sm text-white text-xs font-medium px-4 py-2 rounded-lg hover:bg-red-500 transition-colors cursor-pointer'
                        onClick={handleFailed}
                    >
                        <ArrowPathIcon className='inline-block size-4' />
                    </div>
                )}
            </div>

            {/* Info */}
            <div className="relative mt-2">
                <div className="pr-6">
                    {isEditing ? (
                        <input
                            className="w-full bg-surface rounded-md px-2 py-1 text-sm text-text outline-none focus:ring-2 focus:ring-accent/50"
                            autoFocus
                            value={editValue}
                            onClick={(e) => e.stopPropagation()}
                            onChange={(e) => setEditValue(e.target.value)}
                            onBlur={saveEdit}
                            onKeyDown={(e) => {
                                if (e.key === "Enter") saveEdit();
                                if (e.key === "Escape") cancelEdit();
                            }}
                        />
                    ) : (
                        <h3
                            className="text-sm font-medium text-text-muted group-hover:text-text truncate transition-colors"
                        >
                            {video.name}
                        </h3>
                    )}

                    <p className="text-xs text-text-dim">
                        {formatVideoLength(video.length || 60)}
                    </p>
                </div>

                {/* 3-dots */}
                <div
                    className="absolute top-0 right-0 rounded-md p-1 hover:bg-white/10 cursor-pointer block md:hidden md:group-hover:block has-data-open:block"
                    onClick={(e) => e.stopPropagation()}
                >
                    <VideoDropdownList
                        video={video}
                        onStartEdit={startEditing}
                    />
                </div>
            </div>

            {/* modal */}
            <VideoModal isModalOpen={isOpen} closeModal={closeModal} video={video} />

        </div>
    )
}

const formatVideoLength = (lengthInSeconds) => {
    const hours = Math.floor(lengthInSeconds / 3600);
    const minutes = Math.floor((lengthInSeconds % 3600) / 60);
    const seconds = Math.floor(lengthInSeconds % 60);
    // show like youtube, if hours > 0, show hh:mm:ss, else show mm:ss, round seconds
    if (hours > 0) {    
        return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    } else {
        return `${minutes}:${String(seconds).padStart(2, '0')}`;
    }
}