import React, { useState, useEffect, useRef } from 'react'
import HistoryConversations from './components/HistoryConversations'
import VideosInConversation from './components/VideosInConversation/VideosInConversation'
import UserBar from './components/UserBar'
import { useStore } from '@/stores/chat'

export default function Sidebar() {
    const sidebarOpen = useStore((state) => state.sidebarOpen)
    const setSidebarOpen = useStore((state) => state.setSidebarOpen)

    // Track if animation should be enabled (only on user interaction, not resize)
    const [shouldAnimate, setShouldAnimate] = useState(false)
    const prevOpenRef = useRef(sidebarOpen)

    useEffect(() => {
        // Only animate when sidebarOpen changes (user clicked toggle)
        if (prevOpenRef.current !== sidebarOpen) {
            setShouldAnimate(true)
            prevOpenRef.current = sidebarOpen
        }
    }, [sidebarOpen])

    // Disable animation on resize
    useEffect(() => {
        let resizeTimeout
        const handleResize = () => {
            setShouldAnimate(false)
            clearTimeout(resizeTimeout)
        }
        window.addEventListener('resize', handleResize)
        return () => {
            window.removeEventListener('resize', handleResize)
            clearTimeout(resizeTimeout)
        }
    }, [])

    return (
        <>
            {/* Overlay for mobile - closes sidebar when clicking outside */}
            {sidebarOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-40 md:hidden"
                    onClick={() => setSidebarOpen(false)}
                />
            )}

            {/* Sidebar */}
            <div
                className={`
                    h-screen min-w-[300px] max-w-[300px] bg-darker-background text-gray-400 flex flex-col
                    fixed md:relative z-50
                    shadow-[1px_0_0_0_rgba(255,255,255,0.03)]
                    ${shouldAnimate ? 'transition-transform duration-300 ease-in-out' : ''}
                    ${sidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
                `}
            >
                {/* Close button for mobile */}
                {/* <button
                    onClick={() => setSidebarOpen(false)}
                    className="md:hidden absolute top-2 right-2 p-2 text-gray-400 hover:text-white"
                    aria-label="Close sidebar"
                >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                </button> */}

                <div className='flex-[10] overflow-y-auto'>
                    <HistoryConversations />
                </div>
                <div className='flex-[10] overflow-y-auto'>
                    <VideosInConversation />
                </div>
                <div className='border-t border-white/10 p-2'>
                    <UserBar />
                </div>
            </div>
        </>
    )
}
