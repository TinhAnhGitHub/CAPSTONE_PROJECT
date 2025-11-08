import React from 'react'
import { useMutation, useQueryClient } from 'react-query';
import api from '@/api/api';
import VideoDropdownList from './VideoDropdownList';
import SelectedIcon from './SelectedIcon';
import { useStore } from '@/stores/chat';
import clsx from 'clsx';
import { ingested } from '@/utils/library';
import IngestedStatus from './IngestedStatus/IngestedStatus';

export default function VideoCard({ video }) {    
    const queryClient = useQueryClient();
    const session_id = useStore((state) => state.session_id);
    const selectMutation = useMutation({
        mutationFn: async ({video_id, session_id}) => {
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
        selectMutation.mutate({video_id, session_id});
    }
    
    return (
        <div className={clsx("relative mb-4 rounded-lg hover:shadow-lg p-2 cursor-pointer transition",
            !ingested(video.ingested_status) && "opacity-50 !cursor-not-allowed hover:shadow-none"
            )}
            onClick={() => {
                if (!ingested(video.ingested_status)) return;
                handleToggleSelect(video._id, session_id)}}>
            <div className='relative'>
                <img
                    src={video.thumbnail || "/images/testImage.png"}
                    alt="thumbnail"
                    className="w-full h-auto rounded-lg border border-gray-300"
                />
                <div className='absolute top-0 left-0 w-full h-full flex items-center justify-center'>
                    <IngestedStatus percentage={video.ingested_status} />
                </div>
            </div>
            <div className="relative">
                <div className='pr-8'>
                    <h3 className="text-sm font-semibold truncate">{video.name}</h3>
                    <p className="text-xs text-gray-500">{video.length || "1:00"}</p>
                </div>
                <div className="absolute top-2 right-0 rounded-full p-1 hover:bg-gray-200 cursor-pointer">
                    <VideoDropdownList video={video} />
                </div>
            </div>
            <div className='absolute top-2 right-2 rounded-full p-1 '>
                <SelectedIcon selected={video.selected} />
            </div>
        </div>
    )
}
