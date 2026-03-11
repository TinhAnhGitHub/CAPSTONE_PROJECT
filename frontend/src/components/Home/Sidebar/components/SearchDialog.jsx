import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import { Dialog, DialogBackdrop, DialogPanel } from '@headlessui/react'
import { MagnifyingGlassIcon, ChatBubbleLeftIcon } from '@heroicons/react/20/solid'
import { useSearchChatHistory } from '@/api/services/hooks/query'
import { useStore as useStoreChat } from '@/stores/chat'
import clsx from 'clsx'

function HighlightedSnippet({ text, query }) {
    if (!query) return <span>{text}</span>
    const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
    const parts = text.split(regex)
    return (
        <span className="truncate">
            {parts.map((part, i) =>
                regex.test(part)
                    ? <mark key={i} className="bg-accent/30 text-text rounded-sm px-0.5">{part}</mark>
                    : <span key={i}>{part}</span>
            )}
        </span>
    )
}

export default function SearchDialog({ isOpen, onClose }) {
    const [query, setQuery] = useState('')
    const [debouncedQuery, setDebouncedQuery] = useState('')
    const inputRef = useRef(null)
    const setSessionId = useStoreChat((state) => state.setSessionId)
    const setTargetMessageId = useStoreChat((state) => state.setTargetMessageId)
    const [selectedIndex, setSelectedIndex] = useState(0)

    const { data: results, isLoading } = useSearchChatHistory(debouncedQuery)

    // Debounce the search query
    useEffect(() => {
        const timer = setTimeout(() => {
            setDebouncedQuery(query.trim())
        }, 300)
        return () => clearTimeout(timer)
    }, [query])

    // Reset state when dialog opens/closes
    useEffect(() => {
        if (isOpen) {
            setQuery('')
            setDebouncedQuery('')
            setSelectedIndex(0)
        }
    }, [isOpen])

    // Reset selected index when results change
    useEffect(() => {
        setSelectedIndex(0)
    }, [results])

    const selectResult = useCallback((sessionId, messageId) => {
        setSessionId(sessionId)
        setTargetMessageId(messageId)
        onClose()
    }, [setSessionId, setTargetMessageId, onClose])

    function handleKeyDown(e) {
        if (!results?.length) return

        if (e.key === 'ArrowDown') {
            e.preventDefault()
            setSelectedIndex((prev) => (prev + 1) % results.length)
        } else if (e.key === 'ArrowUp') {
            e.preventDefault()
            setSelectedIndex((prev) => (prev - 1 + results.length) % results.length)
        } else if (e.key === 'Enter') {
            e.preventDefault()
            selectResult(results[selectedIndex].session_id, results[selectedIndex].message_id)
        }
    }

    return (
        <Dialog
            open={isOpen}
            onClose={onClose}
            className="relative z-50 focus:outline-none"
        >
            <DialogBackdrop className="fixed inset-0 bg-black/50" />
            <div className="fixed inset-0 flex items-start justify-center pt-[15vh]">
                <DialogPanel className="w-full max-w-xl rounded-xl bg-surface shadow-2xl border border-surface-light overflow-hidden">
                    {/* Search input */}
                    <div className="flex items-center gap-3 px-4 py-3 border-b border-surface-light">
                        <MagnifyingGlassIcon className="w-5 h-5 text-text-muted shrink-0" />
                        <input
                            ref={inputRef}
                            autoFocus
                            type="text"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Search conversations..."
                            className="flex-1 bg-transparent text-text text-sm placeholder:text-text-dim outline-none"
                        />
                        <kbd className="hidden sm:inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium text-text-dim bg-surface-light border border-surface-light">
                            ESC
                        </kbd>
                    </div>

                    {/* Results */}
                    <div className="max-h-80 overflow-y-auto scrollbar-thin scrollbar-thumb-surface-light scrollbar-track-transparent">
                        {isLoading && debouncedQuery && (
                            <div className="px-4 py-8 text-center text-sm text-text-muted">
                                Searching...
                            </div>
                        )}

                        {!isLoading && debouncedQuery && results?.length === 0 && (
                            <div className="px-4 py-8 text-center text-sm text-text-muted">
                                No conversations found for "{debouncedQuery}"
                            </div>
                        )}

                        {!debouncedQuery && (
                            <div className="px-4 py-8 text-center text-sm text-text-dim">
                                Start typing to search your conversations
                            </div>
                        )}

                        {results?.map((item, index) => (
                            <button
                                key={item.message_id}
                                onClick={() => selectResult(item.session_id, item.message_id)}
                                onMouseEnter={() => setSelectedIndex(index)}
                                className={clsx(
                                    'w-full flex items-center gap-3 px-4 py-3 text-left transition-colors cursor-pointer',
                                    index === selectedIndex
                                        ? 'bg-accent/15 text-text'
                                        : 'text-text-muted hover:bg-white/5'
                                )}
                            >
                                <ChatBubbleLeftIcon className="w-4 h-4 shrink-0 text-text-dim" />
                                <div className="min-w-0 flex-1">
                                    <div className="text-xs text-text-dim mb-0.5">
                                        {item.name}
                                    </div>
                                    <div className="text-sm">
                                        <HighlightedSnippet text={item.snippet} query={debouncedQuery} />
                                    </div>
                                </div>
                            </button>
                        ))}
                    </div>

                    {/* Footer hint */}
                    {results?.length > 0 && (
                        <div className="flex items-center gap-4 px-4 py-2 border-t border-surface-light text-[11px] text-text-dim">
                            <span className="flex items-center gap-1">
                                <kbd className="px-1 py-0.5 rounded bg-surface-light border border-surface-light font-mono">↑</kbd>
                                <kbd className="px-1 py-0.5 rounded bg-surface-light border border-surface-light font-mono">↓</kbd>
                                navigate
                            </span>
                            <span className="flex items-center gap-1">
                                <kbd className="px-1 py-0.5 rounded bg-surface-light border border-surface-light font-mono">↵</kbd>
                                open
                            </span>
                        </div>
                    )}
                </DialogPanel>
            </div>
        </Dialog>
    )
}
