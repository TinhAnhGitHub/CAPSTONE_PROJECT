import React, { useState } from 'react'
import SessionDropdownList from './SessionDropdownList'
import clsx from 'clsx'
import { useStore as useStoreChat } from "@/stores/chat";

export default function ChatHistory({ conv, session_id, onEdit }) {
    const [isEditing, setIsEditing] = useState(false);
    const [editValue, setEditValue] = useState(conv._id);
    const setSessionId = useStoreChat((state) => state.setSessionId);


    function onSelect(session_id) {
        // set session id 
        setSessionId(session_id);
    }


    function saveEdit() {
        if (editValue.trim()) {
            // Call parent handler to save the edit
            onEdit?.(conv._id, editValue.trim());
        }
        setIsEditing(false);
    }

    function cancelEdit() {
        setEditValue(conv.name || conv._id); // Reset to original
        setIsEditing(false);
    }

    function startEditing() {
        setEditValue(conv.name || conv._id);
        setIsEditing(true);
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
