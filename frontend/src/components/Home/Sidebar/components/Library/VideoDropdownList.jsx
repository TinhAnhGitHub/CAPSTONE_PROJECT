import api from '@/api/api';
import { useStore } from '@/stores/chat';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react'
import {
    ArchiveBoxXMarkIcon,
    ChevronDownIcon,
    EllipsisVerticalIcon,
    PencilIcon,
    Square2StackIcon,
    TrashIcon,
} from '@heroicons/react/16/solid'
import { useMutation, useQueryClient } from 'react-query'

export default function VideoDropdownList({ video, onStartEdit }) {
    const queryClient = useQueryClient();
    const session_id = useStore((state) => state.session_id);
    const group = useStore((state) => state.currentGroup);
    const deleteVideoMutation = useMutation({
        mutationFn: (video) => {
            return api.delete('/api/user/videos/delete', {
                data: {
                    video_ids: [video._id],
                    video_run_ids: [video.video_run_id],
                }
            })
        },
        onMutate: async (video) => {
            const queryKey = ['videos', group, session_id];
            await queryClient.cancelQueries(queryKey);
            const previousVideos = queryClient.getQueryData(queryKey);
            queryClient.setQueryData(queryKey, old => old.filter(v => v._id !== video._id));
            return { previousVideos };
        },
        onError: (err, video, context) => {
            console.log(err);
            queryClient.setQueryData(queryKey, context.previousVideos);
        },
        onSettled: () => {
            console.log('invalidating videos');
            queryClient.invalidateQueries(queryKey);
        }
    })
    function handleDelete(video) {
        deleteVideoMutation.mutate(video);
    }
    return (
        <div className="h-5 w-5" onClick={(e) => { e.stopPropagation(); }}>
            <Menu>
                <MenuButton className="">
                    <EllipsisVerticalIcon className="h-5 w-5 text-text-muted hover:text-text transition-colors" />
                </MenuButton>

                <MenuItems
                    transition
                    anchor="bottom start"
                    className="w-52 origin-top-right rounded-xl border border-surface-light bg-surface shadow-xl p-1 text-sm/6 text-text transition duration-100 ease-out [--anchor-gap:--spacing(1)] focus:outline-none data-closed:scale-95 data-closed:opacity-0 z-50"
                >
                    <MenuItem>
                        <button 
                            onClick={() => onStartEdit(video)}
                        className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <PencilIcon className="size-4 text-text-muted" />
                            Edit
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘E</kbd>
                        </button>
                    </MenuItem>
                    {/* <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <Square2StackIcon className="size-4 text-text-muted" />
                            Duplicate
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem> */}
                    <div className="my-1 h-px bg-surface-light" />
                    {/* <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <ArchiveBoxXMarkIcon className="size-4 text-text-muted" />
                            Archive
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘A</kbd>
                        </button>
                    </MenuItem> */}
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 text-red-400 data-focus:bg-white/10"
                            onClick={() => handleDelete(video)}
                        >
                            <TrashIcon className="size-4" />
                            Delete
                            <kbd className="ml-auto hidden font-sans text-xs text-red-400/50 group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem>
                </MenuItems>
            </Menu>
        </div>
    )
}
