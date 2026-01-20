import api from '@/api/api';
import { useDeleteGroup } from '@/api/services/hooks/query';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react'
import {
    ArchiveBoxXMarkIcon,
    ChevronDownIcon,
    EllipsisVerticalIcon,
    PencilIcon,
    Square2StackIcon,
    TrashIcon,
} from '@heroicons/react/16/solid'

export default function GroupDropdownList({ group }) {
    const deleteGroupMutation = useDeleteGroup();
    function handleDelete(groupId) {
        deleteGroupMutation.mutate(groupId);
    }
    return (
        <div className="h-5 w-5" onClick={(e) => { e.stopPropagation() }}>
            <Menu>
                <MenuButton className="">
                    <EllipsisVerticalIcon className="h-5 w-5 text-text-muted hover:text-text transition-colors" />
                </MenuButton>

                <MenuItems
                    transition
                    anchor="bottom start"
                    className="w-52 origin-top-right rounded-xl border border-surface-light bg-surface p-1 text-sm/6 text-text transition duration-100 ease-out [--anchor-gap:--spacing(1)] focus:outline-none data-closed:scale-95 data-closed:opacity-0 z-50 shadow-lg"
                >
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <PencilIcon className="size-4 text-text-muted" />
                            Edit
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘E</kbd>
                        </button>
                    </MenuItem>
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <Square2StackIcon className="size-4 text-text-muted" />
                            Duplicate
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem>
                    <div className="my-1 h-px bg-surface-light" />
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10">
                            <ArchiveBoxXMarkIcon className="size-4 text-text-muted" />
                            Archive
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘A</kbd>
                        </button>
                    </MenuItem>
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10 text-red-400"
                            onClick={() => handleDelete(group._id)}
                        >
                            <TrashIcon className="size-4" />
                            Delete
                            <kbd className="ml-auto hidden font-sans text-xs text-text-dim group-data-focus:inline">⌘D</kbd>
                        </button>
                    </MenuItem>
                </MenuItems>
            </Menu>
        </div>
    )
}
