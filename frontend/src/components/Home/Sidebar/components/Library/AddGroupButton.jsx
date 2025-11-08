import api from '@/api/api';
import { PlusIcon } from '@heroicons/react/16/solid'
import React from 'react'
import { useMutation, useQueryClient } from 'react-query'

export default function AddGroupButton() {
    const queryClient = useQueryClient();
    const addGroupMutation = useMutation({
        mutationFn: (groupName) => {
            return api.post('/api/user/groups/create', { group_name: groupName })
        },
        onSettled: () => {
            queryClient.invalidateQueries('groups');
        }
    })
    const handleAddGroup = () => {
        // now give current time string
        const groupName = "New Group " + new Date().toLocaleTimeString();
        addGroupMutation.mutate(groupName);
    }
  return (
    <div>
          <PlusIcon className='size-5 text-gray-500 hover:text-gray-700 cursor-pointer' onClick={handleAddGroup} />
    </div>
  )
}
