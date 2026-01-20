import api from '@/api/api';
import Upload from '../../../../common/components/Upload';
import Modal from '@/components/Modal/modal';
import { useQuery, useQueryClient } from 'react-query';
import VideoCard from './VideoCard';
import Group from './Group';
import { PlusIcon } from '@heroicons/react/16/solid';
import AddGroupButton from './AddGroupButton';
import { useStore } from "@/stores/chat";
import { useCreateGroup, useGroups, useVideos } from '@/api/services/hooks/query';
import { useEffect, useRef, useState } from 'react';
import socket from '@/api/socket';
import { ensureGroupId } from '@/utils/ensure/ensureGroupId';

export default function LibraryModal({ isModalOpen, closeModal, focusVideoId }) {
    const group = useStore((state) => state.currentGroup);
    const setCurrentGroup = useStore((state) => state.setCurrentGroup);
    const sessionId = useStore((state) => state.session_id);
    const [highlightedVideoId, setHighlightedVideoId] = useState(null);
    const videoRefs = useRef({});
    // get groups
    const { data: groups = [] } = useGroups();
    const queryClient = useQueryClient();
    const createNewGroupMutation = useCreateGroup();

    // Mock data for testing - remove or set to [] in production
    const mockGroups = [
        { _id: 'g1', name: 'Marketing Videos' },
        { _id: 'g2', name: 'Tutorial Series' },
        { _id: 'g3', name: 'Product Demos' },
        { _id: 'g4', name: 'Training Materials' },
    ];

    const mockVideos = [
        { _id: 'v1', name: 'Introduction to AI', thumbnail: '/images/testImage.png', length: '12:34', ingested_status: 100, selected: false },
        { _id: 'v2', name: 'React Tutorial Part 1', thumbnail: '/images/testImage.png', length: '8:22', ingested_status: 100, selected: true },
        { _id: 'v3', name: 'Building Modern UIs', thumbnail: '/images/testImage.png', length: '15:00', ingested_status: 75, selected: false },
        { _id: 'v4', name: 'Video Editing Basics', thumbnail: '/images/testImage.png', length: '20:15', ingested_status: 100, selected: false },
        { _id: 'v5', name: 'CSS Grid Layout', thumbnail: '/images/testImage.png', length: '10:45', ingested_status: 100, selected: true },
        { _id: 'v6', name: 'Node.js Crash Course', thumbnail: '/images/testImage.png', length: '25:00', ingested_status: 50, selected: false },
    ];

    // Use mock if data is empty
    const displayGroups = groups.length > 0 ? groups : mockGroups;

    useEffect(() => {
        socket.on('ingestion_status', (data) => {
            // video_id and run_id
            // invalidate videos query to refetch updated status
            queryClient.invalidateQueries(['videos']);
        })
    }, [queryClient])

    useEffect(() => {
        const result = ensureGroupId(groups, group, setCurrentGroup);
        // console.log("Ensure group result:", result);
        if (result.status === "create") {
            createNewGroupMutation.mutate();
        }
    }, [group])

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

    const { data: videos = [] } = useVideos(group, sessionId);

    // Use mock if videos is empty
    const displayVideos = videos.length > 0 ? videos : mockVideos;

    return (
        <Modal isOpen={isModalOpen} onClose={closeModal} title="Library">
            <div className='flex max-md:flex-col'>
                <div className='max-md:flex max-md:flex-col md:w-[20%]  md:border-r border-surface-light p-2 overflow-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent'>
                    <div className='flex flex-col max-md:flex-row-reverse max-md:items-center items-right max-md:justify-between mb-2'>
                        <AddGroupButton />
                        <div className='border-b border-surface-light my-2 max-md:hidden'></div>
                        <p className=' text-xs text-text-muted uppercase tracking-wide px-2'>GROUPS</p>
                    </div>
                    <div className='groups max-md:flex max-md:overflow-x-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent'>
                        {
                            displayGroups.map((group, idx) => (
                                <Group key={idx} group={group} />
                            ))
                        }
                    </div>
                </div>
                <div className="relative w-[80%] h-[70vh] flex flex-col">
                    <div className="flex-1 overflow-y-auto px-1 scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent">
                        <div className="columns-2 md:columns-4 gap-4 p-2">
                            {displayVideos.map((video, idx) => (
                                <div
                                    key={idx}
                                    className='break-inside-avoid'
                                    ref={el => videoRefs.current[video._id] = el}
                                >
                                    <VideoCard video={video} isHighlighted={highlightedVideoId === video._id} />
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
