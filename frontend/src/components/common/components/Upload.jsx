import React from 'react'
import { useMutation, useQueryClient } from 'react-query';
import Dropzone from 'react-dropzone';
import api from '@/api/api';
import toast from 'react-hot-toast';
import { PlusIcon } from '@heroicons/react/24/solid'
import { useStore } from '@/stores/chat';

export default function Upload() {
    const group = useStore((state) => state.currentGroup);
    const sessionId = useStore((state) => state.session_id);

    const queryClient = useQueryClient();
    const uploadMutation = useMutation(
        async (files) => {
            setIsUploading(true);
            setProgress(0);

            const formData = new FormData();
            files.forEach(file => {
                formData.append('files', file);
            });
            formData.append('group', group);
            formData.append('session_id', sessionId);
            const res = await api.post('/api/user/uploads', formData, {
                onUploadProgress: (event) => {
                    const percent = Math.round((event.loaded * 100) / event.total);
                    setProgress(percent);
                }
            });
            return res.data;
        },
        {
            onSuccess: () => {
                toast.success('Uploaded!');
                queryClient.invalidateQueries('videos');
            },
            onError: () => {
                toast.error('Error uploading file')
            },
            onSettled: () => {
                setIsUploading(false);
                setProgress(0);
            }
        }
    );
    const [isUploading, setIsUploading] = React.useState(false);
    const [progress, setProgress] = React.useState(0);
    return (
        <>
            <Dropzone
                accept={{ 'video/*': [] }}
                onDrop={(files) => uploadMutation.mutate(files)}>
                {({ getRootProps, getInputProps }) => (
                    <div {...getRootProps()} className="bg-black text-white rounded-md p-2 cursor-pointer">
                        <input {...getInputProps()} />
                        <div className='flex items-center gap-1'>
                            {
                                isUploading ?
                                    `Uploading... ${progress}%`
                                    :
                                    <div><PlusIcon className='h-5 w-5 inline-block' /> Upload Video</div>

                            }
                        </div>
                    </div>
                )}
            </Dropzone>
        </>
    )
}
