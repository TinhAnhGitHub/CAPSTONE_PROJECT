import React, { useState } from 'react'
import SessionDropdownList from './SessionDropdownList'
import clsx from 'clsx'
import { useStore as useStoreChat } from "@/stores/chat";
import useEdit from '@/api/services/hooks/edit';
export default function ChatHistory({ conv, session_id, onEdit }) {
    const setSessionId = useStoreChat((state) => state.setSessionId);

    const { isEditing, editValue, setEditValue, startEditing, saveEdit, cancelEdit } = useEdit({
        initialValue: conv.name || conv._id,
        onSave: (newName) => {
            onEdit?.(conv._id, newName);
        }
    });

    function onSelect(session_id) {
        setSessionId(session_id);
    }

    return (
        isEditing ? (
            <input
                className="relative mx-2 my-1 py-2 px-3 bg-surface rounded-lg text-sm text-text outline-none focus:ring-2 focus:ring-accent/50"
                autoFocus
                value={editValue}
                onChange={(e) => setEditValue(e.target.value)}
                onBlur={saveEdit}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') saveEdit();
                    if (e.key === 'Escape') cancelEdit();
                }}
            />
        ) :

            <div
                className={
                    clsx('relative mx-2 my-0.5 py-2 px-3 rounded-lg cursor-pointer transition-colors',
                        'text-text-muted hover:text-text hover:bg-white/5',
                        session_id === conv._id && 'bg-white/10 text-text',
                        "group"
                    )
                }
                onClick={() => onSelect(conv._id)
                }>
                <div className='text-sm truncate pr-6'>{conv.name || conv._id}</div>
                {/* Ellipsis: always visible on mobile (md:hidden), hover on desktop (md:group-hover:block) */}
                <div className='absolute right-2 top-1/2 -translate-y-1/2 rounded-md p-1 hover:bg-white/10 cursor-pointer block md:hidden md:group-hover:block has-data-open:block'>
                    <SessionDropdownList session={conv} onStartEdit={startEditing} />
                </div>
            </div>
    );
}
