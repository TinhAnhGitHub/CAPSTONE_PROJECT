// type StepStatus = "done" | "active" | "pending" | "error";

// type Step = {
//     id: string;
//     title: string;
//     description?: string;
//     status: StepStatus;
//     timestamp?: string;
// };

import { useState, useEffect } from "react";
import { CheckIcon, ChevronDownIcon } from "@heroicons/react/24/solid";
import { Disclosure, DisclosureButton, DisclosurePanel } from "@headlessui/react";
import Markdown from "react-markdown";
import { ChatBubbleLeftEllipsisIcon } from "@heroicons/react/20/solid";

// Inline code component (for `code`) - matches TextBlock style
function InlineCode({ children, ...props }) {
    return (
        <code
            className="!bg-white/10 !text-text-muted !px-1.5 !py-0.5 !rounded !text-[13px] !font-mono !font-normal before:!content-none after:!content-none"
            {...props}
        >
            {children}
        </code>
    );
}

export default function VerticalStepper({ steps, autoExpand = true }) {
    // Track which steps are expanded
    const [expandedSteps, setExpandedSteps] = useState(new Set());

    // Only auto-expand the last step when autoExpand is true
    useEffect(() => {
        if (autoExpand) {
            setExpandedSteps(new Set([steps.length - 1]));
        } else {
            setExpandedSteps(new Set());
        }
    }, [steps.length, autoExpand]);

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
        <ol className="relative border-l py-2 pr-2 border-surface-light border-dashed">
            {steps.map((step, i) => {
                const isFirst = i === 0;
                const isLast = i === steps.length - 1;
                const isExpanded = expandedSteps.has(i);
                const hasDescription = !!step.description;

                return (
                    <li key={i} className="ml-6 pb-4 last:pb-0">
                        {/* Dot/Icon */}
                        {isFirst && <span
                            className={`absolute -left-3 flex h-6 w-6 items-center justify-center rounded-full border 
                            bg-accent border-accent
                            `}
                        >
                            {/* thinking bubble icon */}
                            <ChatBubbleLeftEllipsisIcon className="h-3 w-3 text-white" />
                            {/* )} */}
                        </span>}

                        {/* Content */}
                        <div>
                            <button
                                type="button"
                                onClick={() => toggleStep(i)}
                                className={`flex items-center gap-1 text-left ${hasDescription ? "cursor-pointer" : "cursor-default"}`}
                                disabled={!hasDescription}
                            >
                                <h3 className="text-sm font-medium text-text">{step.title}</h3>
                                {hasDescription && (
                                    <ChevronDownIcon
                                        className={`h-4 w-4 text-text-muted transition-transform duration-200 ${isExpanded ? "rotate-180" : ""}`}
                                    />
                                )}
                            </button>
                            {hasDescription && (
                                <div
                                    className={`overflow-hidden transition-all duration-200 ${isExpanded ? " opacity-100" : "max-h-0 opacity-0"}`}
                                >
                                    <div className="mt-1 text-sm text-text-muted prose prose-sm prose-invert max-w-full prose-code:before:content-none prose-code:after:content-none">
                                        <Markdown
                                            components={{
                                                code: ({ className, children, ...props }) => {
                                                    if (className) {
                                                        return <code className={className} {...props}>{children}</code>;
                                                    }
                                                    return <InlineCode {...props}>{children}</InlineCode>;
                                                },
                                            }}
                                        >
                                            {step.description}
                                        </Markdown>
                                    </div>
                                </div>
                            )}
                        </div>
                    </li>
                );
            })}
        </ol>
    );
}
