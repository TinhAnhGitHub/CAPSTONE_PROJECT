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
import { useEffect } from 'react';
import socket from '@/api/socket';
import { ensureGroupId } from '@/utils/ensure/ensureGroupId';

export default function LibraryModal({ isModalOpen, closeModal }) {
    const group = useStore((state) => state.currentGroup);
    const setCurrentGroup = useStore((state) => state.setCurrentGroup);
    const sessionId = useStore((state) => state.session_id);
    // get groups
    const { data: groups = [] } = useGroups();
    const queryClient = useQueryClient();
    const createNewGroupMutation = useCreateGroup();

    useEffect(() => {
        socket.on('ingestion_status', (data)=> {
            // video_id and run_id
            // invalidate videos query to refetch updated status
            queryClient.invalidateQueries(['videos']);
        })
    }, [queryClient])

    useEffect(() => {
        const result = ensureGroupId(groups, group, setCurrentGroup);
        console.log("Ensure group result:", result);
        if (result.status === "create") {
            createNewGroupMutation.mutate();
        }
    }, [group])

    const { data: videos = [] } = useVideos(group, sessionId);
    return (
        <Modal isOpen={isModalOpen} onClose={closeModal} title="Library">
            <div className='flex'>

                <div className='w-[20%] border-r border-gray-200 p-2 overflow-auto'>
                    <div className='flex items-center mb-2'>
                        <p className='font-semibold'>Groups</p>
                        <div className='ml-auto'>
                            <AddGroupButton />
                        </div>
                    </div>
                    {
                        groups.map((group, idx) => (
                            <Group key={idx} group={group} />
                        ))
                    }
                </div>
                <div className="relative w-[80%] h-[70vh] flex flex-col">
                    <div className="flex-1 overflow-y-auto px-1">
                        <div className="columns-4 gap-4 p-2">
                            {videos.map((video, idx) => (
                                <div key={idx} className='break-inside-avoid'><VideoCard video={video} /></div>
                            ))}
                        </div>
                    </div>

                    {/* Sticky Upload at bottom */}
                    <div className="p-2 border-t border-gray-200 dark:border-gray-700">
                        <Upload />
                    </div>
                </div>
            </div>
        </Modal>
    )
}
