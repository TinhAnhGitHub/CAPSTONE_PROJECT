import api from '@/api/api';
import Upload from '../../../../common/components/Upload';
import Modal from '@/components/Modal/modal';
import { useQuery, useQueryClient } from 'react-query';
import VideoCard from './VideoCard';
import Group from './Group';
import AddGroupButton from './AddGroupButton';
import { useStore } from "@/stores/chat";
import { useGroups, useRenameGroup, useRenameVideo, useVideos } from '@/api/services/hooks/query';
import { useEffect, useRef, useState } from 'react';
import socket from '@/api/socket';

export default function LibraryModal({ isModalOpen, closeModal, focusVideoId }) {
    const group = useStore((state) => state.currentGroup);
    const { data: groups = [] } = useGroups();
    
    const sessionId = useStore((state) => state.session_id);
    const [highlightedVideoId, setHighlightedVideoId] = useState(null);
    const videoRefs = useRef({});
    // get groups
    const queryClient = useQueryClient();

    const { data: videos = [] } = useVideos(group, sessionId);

    useEffect(() => {
        socket.on('ingestion_status', (data) => {
            // video_id and run_id
            // invalidate videos query to refetch updated status
            queryClient.invalidateQueries(['videos']);
        })
        return () => {
            socket.off('ingestion_status');
        }
    }, [queryClient])

    // Handle focus video - scroll to it and highlight
    useEffect(() => {
        if (focusVideoId && isModalOpen) {
            setHighlightedVideoId(focusVideoId);

            // Small delay to ensure DOM is ready
            const scrollTimeout = setTimeout(() => {
                const videoElement = videoRefs.current[focusVideoId];
                if (videoElement) {
                    videoElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }, 100);

            // Remove highlight after animation
            const highlightTimeout = setTimeout(() => {
                setHighlightedVideoId(null);
            }, 2000);

            return () => {
                clearTimeout(scrollTimeout);
                clearTimeout(highlightTimeout);
            };
        }
    }, [focusVideoId, isModalOpen]);

    const renameGroup = useRenameGroup();
    function handleEditGroup(groupId, newName) {
        // TODO: Call API to persist the change
        renameGroup.mutate({ groupId, newName });
    }

    const renameVideo = useRenameVideo();
    function handleEditVideo(videoId, newName) {
        renameVideo.mutate({ videoId, newName });
    }

    return (
        <Modal isOpen={isModalOpen} onClose={closeModal} title="Library">
            <div className='flex max-md:flex-col'>
                <div className='max-md:flex max-md:flex-col md:w-[20%]  md:border-r border-surface-light p-2  scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent '>
                    <div className='flex flex-col max-md:flex-row-reverse max-md:items-center items-right max-md:justify-between mb-2'>
                        <AddGroupButton />
                        <div className='border-b border-surface-light my-2 max-md:hidden'></div>
                        <p className=' text-xs text-text-muted uppercase tracking-wide px-2'>GROUPS</p>
                    </div>
                    <div className='groups max-md:flex max-md:overflow-x-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent overflow-y-auto md:max-h-[55vh]'>
                        {
                            groups.map((group, idx) => (
                                <Group key={idx} groups={groups} group={group} onEdit={handleEditGroup} />
                            ))
                        }
                    </div>
                </div>
                <div className="relative md:w-[80%] h-[70vh] flex flex-col">
                    <div className="flex-1 overflow-y-auto px-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent">
                        <div className=" grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2 p-2">
                            {videos.map((video, idx) => (
                                <div
                                    key={idx}
                                    className='break-inside-avoid'
                                    ref={el => videoRefs.current[video._id] = el}
                                >
                                    <VideoCard video={video} isHighlighted={highlightedVideoId === video._id} onEdit={handleEditVideo} />
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Sticky Upload at bottom */}
                    <div className="ml-2 pr-2 py-2 border-t border-surface-light">
                        <Upload />
                    </div>
                </div>
            </div>
        </Modal>
    )
}
