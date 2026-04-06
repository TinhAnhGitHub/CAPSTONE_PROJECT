import { Button } from '@headlessui/react'
import React from 'react'
import { useWatch } from 'react-hook-form'
import { ArrowUpIcon, StopIcon } from '@heroicons/react/20/solid'

export default function SendButton({ control, querying, onSend, onStop }) {
    const prompt = useWatch({ control, name: 'prompt' });
    const hasContent = prompt?.trim();

    return (
        <Button
            onClick={querying ? onStop : onSend}
            disabled={!querying && !hasContent}
            className={`
                flex items-center justify-center w-8 h-8 rounded-full
                transition-all duration-200
                ${querying || hasContent
                    ? 'bg-accent hover:bg-accent-hover text-white cursor-pointer'
                    : 'bg-surface-light text-text-dim cursor-not-allowed'
                }
                focus:outline-none focus:ring-2 focus:ring-accent/50
            `}
        >
            {
                querying ?
                    <StopIcon className="w-5 h-5" /> :
                    <ArrowUpIcon className="w-5 h-5" />
            }
        </Button>
    )
}
