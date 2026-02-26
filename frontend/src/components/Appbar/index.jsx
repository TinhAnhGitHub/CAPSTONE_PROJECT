import React from 'react'
import { useStore } from '@/stores/chat'
import { Bars3Icon } from '@heroicons/react/24/outline'

export default function AppBar() {
  const toggleSidebar = useStore((state) => state.toggleSidebar)

  return (
    <header className="flex items-center gap-4 px-4 py-2 md:hidden sticky top-0 z-10 bg-background border-b border-surface-light">
      {/* Hamburger button - only visible on mobile (md and below) */}
      <button
        onClick={toggleSidebar}
        className="p-2 rounded-lg text-text-muted hover:text-text hover:bg-surface-light transition-colors cursor-pointer"
        aria-label="Toggle sidebar"
      >
        <Bars3Icon className="w-6 h-6" />
      </button>
    </header>
  )
}
