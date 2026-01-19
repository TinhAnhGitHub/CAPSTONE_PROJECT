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

    const [isUploading, setIsUploading] = React.useState(false);
    const [progress, setProgress] = React.useState(0);


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
            },
            onError: () => {
                toast.error('Error uploading file')
            },
            onSettled: () => {
                queryClient.invalidateQueries(['videos']);
                setIsUploading(false);
                setProgress(0);
            }
        }
    );
    return (
        <>
            <Dropzone
                accept={{ 'video/*': [] }}
                onDrop={(files) => uploadMutation.mutate(files)}>
                {({ getRootProps, getInputProps }) => (
                    <div {...getRootProps()} className="bg-accent hover:bg-accent-hover text-white rounded-md p-2 cursor-pointer transition-colors ">
                        <input {...getInputProps()} />
                        <div className='flex items-center justify-center gap-1 font-medium'>
                            {
                                isUploading ?
                                    `Uploading... ${progress}%`
                                    :
                                    <>
                                        <PlusIcon className='h-5 w-5' /> Upload Video</>
                            }
                        </div>
                    </div>
                )}
            </Dropzone>
        </>
    )
}
