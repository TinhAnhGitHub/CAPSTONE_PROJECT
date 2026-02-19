import api from '@/api/api';
import { useCreateNewChat, useDeleteSession } from '@/api/services/hooks/query';
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
import { DocumentTextIcon, PhotoIcon } from '@heroicons/react/24/outline';
import JSZip from 'jszip';

export default function SaveKeyframes({ segments, videoId, videoName }) {

    const handleSaveImages = () => {
        const imageUrls = segments.map(segment => segment.preview_images?.[2])
        const zip = new JSZip();
        const imgFolder = zip.folder("keyframe_images");
        imageUrls.forEach((url, index) => {
            if (url) {
                imgFolder.file(`keyframe_${index + 1}.jpg`, url.split(',')[1], { base64: true });
            }
        });
        zip.generateAsync({ type: "blob" }).then((content) => {
            const link = document.createElement("a");
            link.href = URL.createObjectURL(content);
            link.download = "keyframe_images.zip";
            link.click();
        });
    }

    const handleSaveTextFiles = () => {
        // create a text file with the keyframe information, by dumping the segments data into a text file
        const textContent = JSON.stringify(segments, null, 2); 
        const blob = new Blob([textContent], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'keyframe_info.txt';
        link.click();
    }

    return (
        <div className="h-5 w-5" onClick={(e) => { e.stopPropagation(); }}>
            <Menu>
                <MenuButton className="inline-flex items-center gap-2 rounded-md bg-accent px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-white/10 focus:not-data-focus:outline-none data-focus:outline data-focus:outline-white data-hover:bg-accent/90 data-open:bg-accent/90 whitespace-nowrap hover:cursor-pointer">
                    Download Keyframes
                    <ChevronDownIcon className="size-4 fill-white/60" />
                </MenuButton>

                <MenuItems
                    transition
                    anchor="bottom start"
                    className="w-52 origin-top-right rounded-xl border border-surface-light bg-surface p-1 text-sm/6 text-text transition duration-100 ease-out [--anchor-gap:--spacing(1)] focus:outline-none data-closed:scale-95 data-closed:opacity-0 z-50 shadow-lg"
                >
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10 hover:cursor-pointer"
                            onClick={handleSaveImages}>
                            <PhotoIcon className="size-4 text-text-muted" />
                            Images
                        </button>
                    </MenuItem>
                    <MenuItem>
                        <button className="group flex w-full items-center gap-2 rounded-lg px-3 py-1.5 data-focus:bg-white/10 hover:cursor-pointer"
                            onClick={handleSaveTextFiles}>
                        
                            <DocumentTextIcon className="size-4 text-text-muted" />
                            Text File
                        </button>
                    </MenuItem>
                </MenuItems>
            </Menu>
        </div>
    )
}
