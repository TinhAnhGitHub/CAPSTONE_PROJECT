import { useState, useEffect } from "react";
import { CheckIcon, ChevronDownIcon } from "@heroicons/react/24/solid";
import {
    DocumentTextIcon,      // OCR
    MicrophoneIcon,        // Voice Recognition
    PhotoIcon,             // CLIP / Image Search
    EyeIcon,               // Image Analyzing
    MagnifyingGlassIcon,   // Search
    CpuChipIcon            // Default/AI
} from "@heroicons/react/24/outline";

// Map tool names to their icons
const toolIcons = {
    'ocr': DocumentTextIcon,
    'text_recognition': DocumentTextIcon,
    'voice': MicrophoneIcon,
    'voice_recognition': MicrophoneIcon,
    'speech': MicrophoneIcon,
    'asr': MicrophoneIcon,
    'clip': PhotoIcon,
    'image_search': PhotoIcon,
    'image_analyzing': EyeIcon,
    'image_analysis': EyeIcon,
    'image_recognition': EyeIcon,
    'analyze': EyeIcon,
    'search': MagnifyingGlassIcon,
    'default': CpuChipIcon
};

const getToolIcon = (toolName) => {
    const normalizedName = toolName.toLowerCase().replace(/\s+/g, '_');
    for (const [key, Icon] of Object.entries(toolIcons)) {
        if (normalizedName.includes(key)) {
            return Icon;
        }
    }
    return toolIcons.default;
};

export default function ToolCallsteps({ block = [{
    tool_name: "Image Recognition",
    description: "Calling an image recognition model to analyze the provided image and extract relevant information.",
},
{
    tool_name: "Voice Recognition",
    description: "Using speech-to-text to transcribe audio content from the video."
},
{
    tool_name: "OCR",
    description: "Extracting text from images using optical character recognition."
},
{
    tool_name: "CLIP Search",
    description: "Using CLIP model to find relevant video segments based on semantic image-text matching."
}] }) {
    const [expandedSteps, setExpandedSteps] = useState(new Set());

    useEffect(() => {
        setExpandedSteps(new Set([block.length - 1]));
    }, [block.length]);

    const toggleStep = (index) => {

        setExpandedSteps((prev) => {
            const next = new Set(prev);
            if (next.has(index)) {
                next.delete(index);
            } else {
                next.add(index);
            }
            return next;
        });
    };

    return (
        <div className="w-full px-4">
            <div className="w-full max-w-lg divide-y divide-surface-light rounded-r-xl">
                <ol className="relative border-l py-2 pr-2 border-surface-light border-dashed">
                    {block.map((tool, i) => {
                        {/* const isLast = i === block.length - 1; */ }
                        const isExpanded = expandedSteps.has(i);
                        const hasDescription = !!tool.description;
                        const ToolIcon = getToolIcon(tool.tool_name);
                        const status = tool.status == "finished"; // true: done, false: in-progress
                        return (
                            <li key={i} className="ml-6 pb-8 last:pb-0">
                                {/* Icon in circle - always shows tool icon, styling indicates status */}
                                <span
                                    className={`absolute -left-3 flex h-6 w-6 items-center justify-center rounded-full border ${status
                                        ? 'bg-accent border-accent'
                                        : 'bg-surface-light border-surface-light'
                                        }`}
                                >
                                        {/* <ToolIcon className="h-3 w-3 text-white" /> */ }
                                    {status ? 
                                    (
                                        <CheckIcon className="h-3 w-3 text-white" />
                                    ) : (
                                        <svg className="h-3 w-3 text-white animate-spin" viewBox="0 0 24 24" fill="none">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                        </svg>
                                    )}
                                </span>

                                {/* Content - no duplicate icon */}
                                <div>
                                    <button
                                        type="button"
                                        onClick={() => toggleStep(i)}
                                        className={`flex items-center gap-1 text-left ${hasDescription ? "cursor-pointer" : "cursor-default"}`}
                                        disabled={!hasDescription}
                                    >
                                        <h3 className="text-sm font-medium text-text">{tool.tool_name}</h3>
                                        {hasDescription && (
                                            <ChevronDownIcon
                                                className={`h-4 w-4 text-text-muted transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
                                            />
                                        )}
                                    </button>
                                    {hasDescription && (
                                        <div
                                            className={`overflow-hidden transition-all duration-200 ${isExpanded ? "max-h-40 opacity-100" : "max-h-0 opacity-0"}`}
                                        >
                                            <p className="mt-1 text-sm text-text-muted">
                                                {tool.description}
                                            </p>
                                        </div>
                                    )}
                                </div>
                            </li>
                        );
                    })}
                </ol>
            </div>
        </div>
    );
}
