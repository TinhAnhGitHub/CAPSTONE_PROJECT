import { useCreateGroup } from '@/api/services/hooks/query';
import { PlusIcon } from '@heroicons/react/20/solid'
import React from 'react'

export default function AddGroupButton() {
  const addGroupMutation = useCreateGroup();

  const handleAddGroup = () => {
    // now give current time string
    const groupName = "New Group " + new Date().toLocaleTimeString();
    addGroupMutation.mutate(groupName);
  }
  return (
    <button
      onClick={handleAddGroup}
      className='flex items-center gap-2 w-full px-3 py-2 rounded-lg bg-accent hover:bg-accent-hover text-white text-sm font-medium transition-colors cursor-pointer'
    >
      <PlusIcon className="w-5 h-5" />
      <span>Add Group</span>
    </button>
  )
}
