import api from '@/api/api';
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

export default function VideoDropdownList({ video }) {
    const queryClient = useQueryClient();
    const deleteVideoMutation = useMutation({
        mutationFn: (video) => {
            console.log("Deleting video:", video);
            return api.delete('/api/user/videos/delete', {
                data: {
                    video_ids: [video._id],
                    video_run_ids: [video.video_run_id],
                }
            })
        },
        onSettled: () => {
            queryClient.invalidateQueries('userVideos');
        }
    })
    function handleDelete(video) {
        confirm("Are you sure to delete this video?") &&
            deleteVideoMutation.mutate(video);
    }
    return (
        <div className="h-5 w-5" onClick={(e) => { e.stopPropagation(); }}>
            <Menu>
                <MenuButton className="">
                    <EllipsisVerticalIcon className="h-5 w-5 text-gray-500" />
                </MenuButton>

                <MenuItems
                    transition
                    anchor="bottom start"
                    className="w-52 origin-top-right rounded-xl border border-white/5 bg-black/50 backdrop-blur-md p-1 text-sm/6 text-white transition duration-100 ease-out [--anchor-gap:--spacing(1)] focus:outline-none data-closed:scale-95 data-closed:opacity-0 z-50"
                >
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <PencilIcon className="size-4 fill-white/30" />
                            Edit
                            <kbd className="ml-auto hidden font-sans text-xs text-white/50 group-data-focus:inline">⌘E</kbd>
                        </button>
                    </MenuItem>
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <Square2StackIcon className="size-4 fill-white/30" />
                            Duplicate
                            <kbd className="ml-auto hidden font-sans text-xs text-white/50 group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem>
                    <div className="my-1 h-px bg-white/5" />
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <ArchiveBoxXMarkIcon className="size-4 fill-white/30" />
                            Archive
                            <kbd className="ml-auto hidden font-sans text-xs text-white/50 group-data-focus:inline">⌘A</kbd>
                        </button>
                    </MenuItem>
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10"
                            onClick={() => handleDelete(video)}
                        >
                            <TrashIcon className="size-4 fill-white/30" />
                            Delete
                            <kbd className="ml-auto hidden font-sans text-xs text-white/50 group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem>
                </MenuItems>
            </Menu>
        </div>
    )
}
