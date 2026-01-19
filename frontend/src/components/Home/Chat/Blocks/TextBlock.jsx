import { useState } from "react";
import clsx from "clsx";
import Markdown from "react-markdown";
import { ClipboardDocumentIcon, CheckIcon } from "@heroicons/react/24/outline";
import toast from "react-hot-toast";

function handleCopyText(text, setCopied) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text);
    } else {
        // Fallback for non-secure contexts
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
    }
    setCopied(true);
    toast.success("Copied to clipboard!");
    setTimeout(() => setCopied(false), 2000);
}

export default function TextBlock({ block, role }) {
    const [copied, setCopied] = useState(false);

    return (
        <div
            className={clsx(
                'max-w-[75%] px-4 py-2 my-2 rounded-lg text-sm whitespace-pre-wrap break-words hyphens-auto',
                role === 'user'
                    ? 'bg-accent/80 text-white self-end backdrop-blur-md shadow-lg border border-white/10'
                    : "text-text self-start"
            )}
        >
            <Markdown key={block.text}>{block.text || ''}</Markdown>
            {role === 'assistant' && (
                <button
                    onClick={() => handleCopyText(block?.text, setCopied)}
                    className="mt-3 p-1.5 rounded-md text-text-muted hover:text-text hover:bg-surface-light transition-all duration-200 ease-in-out cursor-pointer"
                    title="Copy to clipboard"
                >
                    {copied ? (
                        <CheckIcon className="w-5 h-5 text-accent" />
                    ) : (
                        <ClipboardDocumentIcon className="w-5 h-5" />
                    )}
                </button>
            )}
        </div>
    );
}
