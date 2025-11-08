import { Button } from '@headlessui/react'
import React from 'react'
import { useWatch } from 'react-hook-form'

export default function SendButton({control, handlePrompt}) {
    const prompt = useWatch({control, name: 'prompt'});
  return (
      <Button
          onClick={handlePrompt}
          disabled={!prompt?.trim()}
          className="inline-flex items-center gap-2 rounded-md bg-gray-700 px-3 py-1.5 text-sm/6 font-semibold text-white shadow-inner shadow-white/10 focus:not-data-focus:outline-none data-focus:outline data-focus:outline-white data-hover:bg-gray-600 data-open:bg-gray-700 self-end disabled:cursor-not-allowed">
          <svg className="w-6 h-6 text-white dark:text-white" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" viewBox="0 0 24 24">
              <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v13m0-13 4 4m-4-4-4 4" />
          </svg>
      </Button>

  )
}
