import React from 'react'
import { useStore } from "@/stores/chat";
import clsx from 'clsx';
import GroupDropdownList from './GroupDropdownList';

export default function Group({ group }) {
  const currentGroup = useStore((state) => state.currentGroup);

  const setCurrentGroup = useStore((state) => state.setCurrentGroup);

  const handleSelectGroup = () => {
    setCurrentGroup(group._id);
  }

  return (
    <div className={clsx(
      'relative mx-1 my-0.5 py-2 px-3 rounded-lg cursor-pointer transition-colors',
      'text-text-muted hover:text-text hover:bg-white/5',
      currentGroup === group._id && 'bg-white/10 text-text',
      'group'
    )}
      onClick={handleSelectGroup}
    >
      <div className='text-sm truncate pr-6'>{group.name}</div>
      {/* Ellipsis: always visible on mobile, hover on desktop */}
      <div className="absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 hover:bg-white/10 cursor-pointer block md:hidden md:group-hover:block has-data-open:block">
        <GroupDropdownList group={group} />
      </div>
    </div>
  )
}
