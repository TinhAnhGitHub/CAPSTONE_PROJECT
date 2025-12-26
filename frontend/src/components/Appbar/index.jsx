import React from 'react'
import { useStore } from '@/stores/chat'

export default function AppBar() {
  const toggleSidebar = useStore((state) => state.toggleSidebar)

  return (
    <header className="flex flex-wrap gap-10 items-center text-sm tracking-normal text-white uppercase max-md:py-2 sticky top-0 z-10 bg-[#0a1242]">
      {/* Hamburger button - only visible on mobile (md and below) */}
      <button
        onClick={toggleSidebar}
        className="md:hidden p-2 rounded-lg hover:bg-white/10 transition-colors"
        aria-label="Toggle sidebar"
      >
        <svg
          className="w-6 h-6"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 6h16M4 12h16M4 18h16"
          />
        </svg>
      </button>
    </header>
  )
}
