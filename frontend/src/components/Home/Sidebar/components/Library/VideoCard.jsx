import React from 'react'
import { useMutation, useQueryClient } from 'react-query';
import api from '@/api/api';
import VideoDropdownList from './VideoDropdownList';
import SelectedIcon from './SelectedIcon';
import { useStore } from '@/stores/chat';
import clsx from 'clsx';
import { ingested } from '@/utils/library';
import IngestedStatus from './IngestedStatus/IngestedStatus';

export default function VideoCard({ video, isHighlighted = false }) {
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
    const handleFailed = () => {
        // call api to re-ingest
    }
    return (
        <div className={clsx(
            "group relative rounded-xl p-2 cursor-pointer transition-all",
            "hover:bg-white/5",
            !ingested(video.ingested_status) && "opacity-50 !cursor-not-allowed",
            isHighlighted && "animate-highlight-pulse ring-2 ring-accent ring-offset-2 ring-offset-background rounded-xl"
        )}>
            {/* Thumbnail */}
            <div className='relative aspect-video rounded-lg overflow-hidden bg-black'>
                <img
                    src={video.thumbnail || "/images/testImage.png"}
                    alt="thumbnail"
                    className="w-full h-full object-cover"
                />
                <div className='absolute inset-0 flex items-center justify-center'>
                    <IngestedStatus percentage={video.ingested_status} />
                </div>
                {/* Selection indicator */}
                <div className='absolute top-2 right-2 rounded-md p-1 bg-black/50 backdrop-blur-sm cursor-pointer hover:bg-black/70 transition-colors'
                    onClick={(e) => {
                        e.stopPropagation();
                        if (!ingested(video.ingested_status)) return;
                        handleToggleSelect(video._id, session_id)
                    }}>
                    <SelectedIcon selected={video.selected} />
                </div>
            </div>

            {/* Info */}
            <div className="relative mt-2">
                <div className='pr-6'>
                    <h3 className="text-sm font-medium text-text-muted group-hover:text-text truncate transition-colors">{video.name}</h3>
                    <p className="text-xs text-text-dim">{video.length || "1:00"}</p>
                </div>
                {/* 3-dots: always visible on mobile, hover on desktop */}
                <div className="absolute top-0 right-0 rounded-md p-1 hover:bg-white/10 cursor-pointer block md:hidden md:group-hover:block has-data-open:block">
                    <VideoDropdownList video={video} />
                </div>
            </div>

            {video.failed && (
                <div className='absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-red-500/90 backdrop-blur-sm text-white text-xs font-medium px-3 py-1.5 rounded-lg'
                    onClick={handleFailed}
                >
                    Retry
                </div>
            )}
        </div>
    )
}
